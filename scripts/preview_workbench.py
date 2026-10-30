"""Start an isolated workbench DB against the configured local paper library.

No source database is copied or historical run executed. The chosen workspace
must be outside the production workspace, and the API credential is never printed.
"""

import argparse
from dataclasses import replace
from pathlib import Path

from paperpilot_agentic.config import RuntimeSettings, resolve_provider_api_key
from paperpilot_agentic.web_app import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--provider", choices=("qwen", "deepseek"), default="qwen")
    parser.add_argument("--model", default="qwen-turbo")
    args = parser.parse_args()
    base = RuntimeSettings.from_env()
    folder = args.workspace.resolve()
    if folder == base.workspace or folder.is_relative_to(base.workspace):
        raise ValueError("Use a separate acceptance workspace, not the production state")
    settings = replace(
        base,
        workspace=folder,
        runtime_db=folder / "runtime.sqlite3",
        provider=args.provider,
        api_key=resolve_provider_api_key(args.provider),
        model=args.model,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        if args.provider == "qwen"
        else "https://api.deepseek.com",
        thinking_enabled=False,
    )
    import uvicorn

    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
