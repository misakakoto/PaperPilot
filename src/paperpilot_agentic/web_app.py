"""Focused local Web application for the slim PaperPilot runtime."""

from __future__ import annotations

import os
import socket
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from .bootstrap import build_runtime
from .config import RuntimeSettings, resolve_provider_api_key
from .web_api import create_fastapi_router
from .build_info import build_status


def probe_provider_connectivity(
    base_url: str,
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    """Check service-process reachability without making a billable API call."""

    parsed = urlparse(str(base_url or ""))
    host = parsed.hostname
    if not host:
        return {
            "reachable": False,
            "probe": "tcp_connect_v1",
            "error_type": "InvalidProviderURL",
        }
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    started = time.perf_counter()
    try:
        with socket.create_connection(
            (host, port),
            timeout=max(0.1, min(float(timeout_seconds), 10.0)),
        ):
            pass
    except OSError as exc:
        return {
            "reachable": False,
            "probe": "tcp_connect_v1",
            "error_type": type(exc).__name__,
            "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
        }
    return {
        "reachable": True,
        "probe": "tcp_connect_v1",
        "error_type": None,
        "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
    }


def create_app(settings: RuntimeSettings | None = None):
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from starlette.middleware.trustedhost import TrustedHostMiddleware
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install PaperPilot with the web extra") from exc

    settings = settings or RuntimeSettings.from_env()
    static_dir = Path(__file__).resolve().parent / "web_static"
    app = FastAPI(title="PaperPilot", version="1.0.0")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost','127.0.0.1','[::1]','testserver'])

    @app.middleware('http')
    async def local_browser_boundary(request, call_next):
        # Localhost is privileged. A remote web page must not approve/run jobs via CSRF.
        if request.method not in {'GET','HEAD','OPTIONS'}:
            origin=request.headers.get('origin')
            same_origin=f'{request.url.scheme}://{request.headers.get("host","")}'
            if (origin and origin != same_origin) or request.headers.get('sec-fetch-site')=='cross-site':
                return JSONResponse({'detail':'Cross-origin research actions are not allowed'},status_code=403)
        response=await call_next(request)
        # Windows registry associations can map .js to text/plain. With nosniff,
        # browsers then correctly refuse every ES module and the UI never boots.
        if request.url.path.startswith('/assets/') and request.url.path.endswith('.js'):
            response.headers['Content-Type']='application/javascript; charset=utf-8'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Content-Security-Policy']=("default-src 'self'; script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; "
            "connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        if request.url.path=='/' or request.url.path.startswith('/assets/'):
            response.headers['Cache-Control']='no-store'
        return response

    runtime = build_runtime(settings)

    def factory():
        return runtime

    app.include_router(
        create_fastapi_router(
            runtime_factory=factory,
            prefix="/api/agentic",
            default_engine="answer_first_text",
            default_visual_policy="auto",
        )
    )

    def qwen_vision_configured() -> bool:
        enabled = os.getenv("PAPERPILOT_QWEN_VISION_ENABLED", "auto").strip().casefold()
        if enabled in {"0", "false", "no", "off", "disabled"}:
            return False
        try:
            return bool(resolve_provider_api_key("qwen"))
        except (OSError, RuntimeError, ValueError):
            return False

    def indexed_paper_count() -> int | None:
        database = settings.memory_db
        if database is None or not database.is_file():
            return None
        try:
            uri = f"file:{database.resolve().as_posix()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                row = connection.execute(
                    "SELECT COUNT(*) FROM papers WHERE namespace=? AND active=1",
                    (settings.memory_namespace,),
                ).fetchone()
            return int(row[0]) if row else 0
        except (OSError, sqlite3.Error):
            return None

    @app.get("/api/bootstrap")
    def bootstrap() -> dict[str, object]:
        return {
            "name": "PaperPilot",
            "build": build_status(),
            "capabilities": {"workbench": True, "research_design": True,
                "paired_diagnostic": True, "arbitrary_code_execution": False,
                "training_checkpoint_resume": False, "research_loop": True,
                "research_loop_checkpoint_resume": True},
            "provider": settings.provider,
            "model": settings.model,
            "source_root": str(Path(__file__).resolve().parents[2]),
            "process_id": os.getpid(),
            "launcher_id": os.getenv("PAPERPILOT_LAUNCHER_ID"),
            "llm_configured": bool(settings.api_key),
            "vision_provider": "qwen",
            "vision_model": os.getenv(
                "PAPERPILOT_QWEN_VISION_MODEL",
                os.getenv("PAPERPILOT_QWEN_MODEL", "qwen3-vl-flash-2025-10-15"),
            ),
            "vision_configured": qwen_vision_configured(),
            "memory_database_configured": bool(
                settings.memory_db and settings.memory_db.is_file()
            ),
            "indexed_paper_count": indexed_paper_count(),
            "memory_namespace": settings.memory_namespace,
            "dense_retrieval_configured": bool(
                getattr(runtime.search_adapter, "embedder", None)
            ),
            'retrieval_status': getattr(runtime.search_adapter, 'retrieval_status', {}),
            "reranker_configured": bool(
                getattr(runtime.search_adapter, "reranker", None)
            ),
            "paper_roots": [str(value) for value in settings.paper_roots],
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/provider-connectivity")
    def provider_connectivity() -> dict[str, object]:
        return {
            **probe_provider_connectivity(
                settings.base_url,
                timeout_seconds=min(settings.timeout_seconds, 5.0),
            ),
            "provider": settings.provider,
            "model": settings.model,
            "billable_requests": 0,
        }

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    class LocalAssets(StaticFiles):
        def is_not_modified(self, response_headers, request_headers):
            # Avoid reusing a cached response with a wrong Windows MIME type.
            return False

    app.mount("/assets", LocalAssets(directory=static_dir), name="assets")
    return app


def main() -> None:
    load_dotenv(override=False)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install PaperPilot with the web extra") from exc
    uvicorn.run(
        "paperpilot_agentic.web_app:create_app",
        factory=True,
        host=os.getenv("PAPERPILOT_WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("PAPERPILOT_WEB_PORT", "8014")),
        reload=False,
    )


if __name__ == "__main__":
    main()
