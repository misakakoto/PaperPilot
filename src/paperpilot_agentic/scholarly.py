from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


_USER_AGENT = "PaperPilot/2.0 (local evidence research agent)"
_ARXIV_HOSTS = {"arxiv.org", "export.arxiv.org"}


class ScholarlySearchAdapter:
    """Traceable academic discovery through public APIs, never generic scraping."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        timeout_seconds: float = 15.0,
        max_download_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = max(2.0, float(timeout_seconds))
        self.max_download_bytes = max(1_000_000, int(max_download_bytes))

    def search(self, query: str, top_k: int = 8, *, published_from: str | None = None,
               published_to: str | None = None, sort: str = 'relevance',
               require_full_text: bool = False) -> dict[str, Any]:
        for value in (published_from, published_to):
            if value is not None and (not isinstance(value, str) or len(value) != 10
                                      or date.fromisoformat(value).isoformat() != value):
                raise ValueError('Publication bounds must be ISO YYYY-MM-DD dates')
        if published_from and published_to and published_from > published_to:
            raise ValueError('Publication date range is reversed')
        if sort not in {'relevance', 'newest'} or type(require_full_text) is not bool:
            raise ValueError('Invalid scholarly sort or full-text requirement')
        top_k = max(1, min(20, int(top_k)))
        clean = re.sub(r"\s+", " ", str(query)).strip()
        if not clean:
            return {"query": clean, "items": [], "providers": [], "errors": []}
        scope = {'policy': 'dated-discovery-v2-conjunctive-arxiv', 'published_from': published_from,
                 'published_to': published_to, 'sort': sort, 'require_full_text': require_full_text,
                 'full_text_scope': 'arxiv_allowlist_only' if require_full_text else 'metadata_and_available_arxiv',
                 'coverage': 'BOUNDED_NOT_EXHAUSTIVE; arxiv dates are submission dates; PDF links are not verified downloads'}
        cache_key = hashlib.sha256(json.dumps([clean.casefold(), top_k, scope], sort_keys=True).encode()).hexdigest()
        cache_path = self.cache_dir / f"search-{cache_key}.json"
        cached = self._read_cache(cache_path, maximum_age=3600 if sort == 'newest' else 7 * 24 * 3600)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

        items: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        providers: list[str] = []
        rejected: list[dict[str, Any]] = []
        for provider, operation in (
            ("openalex", self._openalex),
            ("semantic_scholar", self._semantic_scholar),
            ("arxiv", self._arxiv),
            ("crossref", self._crossref),
        ):
            try:
                dated = bool(published_from or published_to or sort != 'relevance')
                if dated and provider == 'semantic_scholar':
                    errors.append({'provider': provider, 'error': 'Requested date/sort protocol not supported by this adapter; not silently downgraded'})
                    continue
                options = {'published_from': published_from, 'published_to': published_to, 'sort': sort} if dated else {}
                rows = operation(clean, max(3, min(10, top_k)), **options)
                providers.append(provider)
                for row in rows:
                    when = row.get('publication_date')
                    reason = None
                    if published_from or published_to:
                        try:
                            when = date.fromisoformat(when).isoformat() if isinstance(when, str) else None
                        except ValueError:
                            when = None
                        if when is None:
                            reason = 'missing_exact_date'
                        elif (published_from and when < published_from) or (published_to and when > published_to):
                            reason = 'outside_date_range'
                    if reason is None and require_full_text and not self._safe_arxiv_url(row.get('full_text_url')):
                        reason = 'no_supported_full_text_url'
                    if reason:
                        rejected.append({'source_id': row.get('source_id'), 'title': row.get('title'),
                                         'publication_date': row.get('publication_date'), 'reason': reason})
                    else:
                        items.append(row)
            except Exception as exc:  # API failure is a declared boundary, not fatal.
                errors.append({"provider": provider, "error": f"{type(exc).__name__}: {exc}"[:500]})
            if len(self._deduplicate(items)) >= top_k:
                break
        selected = self._deduplicate(items)
        if sort == 'newest':
            selected.sort(key=lambda row: row.get('publication_date') or '', reverse=True)
        result = {
            "query": clean,
            "items": selected[:top_k],
            "providers": providers,
            "errors": errors,
            "cache_hit": False,
            "retrieved_at": time.time(),
            "search_scope": scope,
            "rejected_items": rejected,
        }
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def fetch_full_text(self, url: str) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(str(url))
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or host not in _ARXIV_HOSTS:
            raise PermissionError(
                "Full-text fetch is limited to allowlisted HTTPS scholarly hosts"
            )
        key = hashlib.sha256(str(url).encode("utf-8")).hexdigest()
        pdf_path = self.cache_dir / f"fulltext-{key}.pdf"
        if not pdf_path.is_file():
            request = urllib.request.Request(str(url), headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                content_type = str(response.headers.get("Content-Type", "")).lower()
                data = response.read(self.max_download_bytes + 1)
            if len(data) > self.max_download_bytes:
                raise ValueError("Scholarly full text exceeds the download limit")
            if not data.startswith(b"%PDF") and "pdf" not in content_type:
                raise ValueError("Allowlisted scholarly URL did not return a PDF")
            pdf_path.write_bytes(data)
        if PdfReader is None:
            raise RuntimeError("pypdf is required for scholarly PDF extraction")
        reader = PdfReader(str(pdf_path))
        pages: list[dict[str, Any]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if not text:
                continue
            for offset in range(0, len(text), 5_000):
                chunk = text[offset : offset + 5_000].strip()
                if chunk:
                    pages.append({"page": page_number, "content": chunk})
            if len(pages) >= 24:
                break
        return {
            "url": str(url),
            "cache_path": str(pdf_path),
            "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            "pages": pages,
        }

    def _request_json(self, url: str) -> Mapping[str, Any]:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            data = response.read(5_000_001)
        if len(data) > 5_000_000:
            raise ValueError("Scholarly API response exceeds the size limit")
        value = json.loads(data.decode("utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("Scholarly API did not return an object")
        return value

    def _openalex(self, query: str, limit: int, *, published_from=None, published_to=None, sort='relevance') -> list[dict[str, Any]]:
        args: dict[str, Any] = {"search": query, "per-page": limit}
        filters = []
        if published_from:
            filters.append('from_publication_date:'+published_from)
        if published_to:
            filters.append('to_publication_date:'+published_to)
        if filters:
            args['filter'] = ','.join(filters)
        if sort == 'newest':
            args['sort'] = 'publication_date:desc'
        params = urllib.parse.urlencode(args)
        value = self._request_json(f"https://api.openalex.org/works?{params}")
        result: list[dict[str, Any]] = []
        for row in value.get("results", []):
            if not isinstance(row, Mapping):
                continue
            doi = str(row.get("doi") or "").removeprefix("https://doi.org/") or None
            result.append(
                {
                    "source_id": str(row.get("id") or doi or ""),
                    "provider": "openalex",
                    "title": str(row.get("display_name") or ""),
                    "abstract": self._openalex_abstract(row.get("abstract_inverted_index")),
                    "year": row.get("publication_year"),
                    "publication_date": row.get('publication_date'),
                    "date_basis": 'publication_date',
                    "doi": doi,
                    "landing_url": row.get("doi") or row.get("id"),
                    "full_text_url": self._safe_arxiv_url(
                        ((row.get("best_oa_location") or {}).get("pdf_url") if isinstance(row.get("best_oa_location"), Mapping) else None)
                    ),
                }
            )
        return result

    def _semantic_scholar(self, query: str, limit: int) -> list[dict[str, Any]]:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "limit": limit,
                "fields": "title,abstract,year,url,externalIds,openAccessPdf",
            }
        )
        value = self._request_json(f"https://api.semanticscholar.org/graph/v1/paper/search?{params}")
        result: list[dict[str, Any]] = []
        for row in value.get("data", []):
            if not isinstance(row, Mapping):
                continue
            external = row.get("externalIds") if isinstance(row.get("externalIds"), Mapping) else {}
            oa = row.get("openAccessPdf") if isinstance(row.get("openAccessPdf"), Mapping) else {}
            result.append(
                {
                    "source_id": str(row.get("paperId") or ""),
                    "provider": "semantic_scholar",
                    "title": str(row.get("title") or ""),
                    "abstract": str(row.get("abstract") or ""),
                    "year": row.get("year"),
                    "doi": external.get("DOI"),
                    "landing_url": row.get("url"),
                    "full_text_url": self._safe_arxiv_url(oa.get("url")),
                }
            )
        return result

    def _arxiv(self, query: str, limit: int, *, published_from=None, published_to=None, sort='relevance') -> list[dict[str, Any]]:
        # Natural-language terms must each have an explicit field and Boolean
        # conjunction. A single `all:` before a space-separated query broadens
        # matching and can rank unrelated newest papers. Quoted phrases survive.
        terms = shlex.split(query)
        if not terms:
            return []
        search_query = ' AND '.join('all:"'+term.replace('\\', '\\\\').replace('"', '\\"')+'"' for term in terms)
        if published_from or published_to:
            lower = (published_from or '1991-01-01').replace('-', '')+'0000'
            upper = (published_to or date.today().isoformat()).replace('-', '')+'2359'
            search_query = f'({search_query}) AND submittedDate:[{lower} TO {upper}]'
        args: dict[str, Any] = {"search_query": search_query, "start": 0, "max_results": limit}
        if sort == 'newest':
            args.update(sortBy='submittedDate', sortOrder='descending')
        params = urllib.parse.urlencode(args)
        request = urllib.request.Request(
            f"https://export.arxiv.org/api/query?{params}",
            headers={"User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            data = response.read(5_000_001)
        if len(data) > 5_000_000:
            raise ValueError("arXiv response exceeds the size limit")
        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        result: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns):
            identifier = (entry.findtext("atom:id", default="", namespaces=ns) or "").rstrip("/").split("/")[-1]
            result.append(
                {
                    "source_id": f"arxiv:{identifier}",
                    "provider": "arxiv",
                    "title": re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=ns)).strip(),
                    "abstract": re.sub(r"\s+", " ", entry.findtext("atom:summary", default="", namespaces=ns)).strip(),
                    "year": (entry.findtext("atom:published", default="", namespaces=ns) or "")[:4] or None,
                    "publication_date": (entry.findtext("atom:published", default="", namespaces=ns) or "")[:10] or None,
                    "date_basis": 'arxiv_submission_date',
                    "doi": None,
                    "landing_url": f"https://arxiv.org/abs/{identifier}",
                    "full_text_url": self._safe_arxiv_url(f"https://arxiv.org/pdf/{identifier}") if identifier else None,
                }
            )
        return result

    def _crossref(self, query: str, limit: int, *, published_from=None, published_to=None, sort='relevance') -> list[dict[str, Any]]:
        args: dict[str, Any] = {"query.bibliographic": query, "rows": limit}
        filters = []
        if published_from:
            filters.append('from-pub-date:'+published_from)
        if published_to:
            filters.append('until-pub-date:'+published_to)
        if filters:
            args['filter'] = ','.join(filters)
        if sort == 'newest':
            args.update(sort='published', order='desc')
        params = urllib.parse.urlencode(args)
        value = self._request_json(f"https://api.crossref.org/works?{params}")
        message = value.get("message") if isinstance(value.get("message"), Mapping) else {}
        result: list[dict[str, Any]] = []
        for row in message.get("items", []):
            if not isinstance(row, Mapping):
                continue
            titles = row.get("title") or []
            result.append(
                {
                    "source_id": str(row.get("DOI") or row.get("URL") or ""),
                    "provider": "crossref",
                    "title": str(titles[0] if titles else ""),
                    "abstract": re.sub(r"<[^>]+>", " ", str(row.get("abstract") or "")).strip(),
                    "year": (((row.get("published") or {}).get("date-parts") or [[None]])[0] or [None])[0],
                    "publication_date": self._crossref_date(row),
                    "date_basis": 'publication_date',
                    "doi": row.get("DOI"),
                    "landing_url": row.get("URL"),
                    "full_text_url": None,
                }
            )
        return result

    @staticmethod
    def _crossref_date(row):
        parts = (((row.get('published') or {}).get('date-parts') or [[]])[0])
        if len(parts) < 3:
            return None  # A year-only record is not an invented January 1 date.
        try:
            return date(*parts[:3]).isoformat()
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _openalex_abstract(value: Any) -> str:
        if not isinstance(value, Mapping):
            return ""
        positions: list[tuple[int, str]] = []
        for token, indexes in value.items():
            if isinstance(indexes, Sequence) and not isinstance(indexes, (str, bytes)):
                positions.extend((int(index), str(token)) for index in indexes)
        return " ".join(token for _, token in sorted(positions))

    @staticmethod
    def _safe_arxiv_url(value: Any) -> str | None:
        if not value:
            return None
        url = str(value).replace("http://", "https://", 1)
        try:
            parsed = urllib.parse.urlparse(url)
            port = parsed.port
        except ValueError:
            return None
        if (parsed.scheme != 'https' or (parsed.hostname or '').lower() not in _ARXIV_HOSTS
                or parsed.username or parsed.password or port not in (None, 443)):
            return None
        match = re.fullmatch(r'/(?:abs|pdf)/((?:\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7})(?:v\d+)?)(?:\.pdf)?', parsed.path)
        return 'https://arxiv.org/pdf/'+match.group(1) if match else None

    @staticmethod
    def _deduplicate(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in items:
            item = dict(raw)
            key = str(item.get("doi") or item.get("source_id") or item.get("title") or "").casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    @staticmethod
    def _read_cache(path: Path, *, maximum_age: float) -> dict[str, Any] | None:
        if not path.is_file() or time.time() - path.stat().st_mtime > maximum_age:
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None
