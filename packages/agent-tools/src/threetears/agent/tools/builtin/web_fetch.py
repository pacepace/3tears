"""Web fetch tool for retrieving and extracting page content."""

from __future__ import annotations

import re
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.utils import tool_error

try:
    import trafilatura

    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

_MAX_CHARS = 15000
_MAX_RETRIES = 2


class WebFetchInput(BaseModel):
    """Input for the web fetch tool."""

    url: str = Field(description="URL to fetch and extract content from")


def _extract_meta_refresh(html: str) -> str | None:
    """Extract URL from meta refresh tag."""
    match = re.search(
        r'<meta[^>]*http-equiv=["\']refresh["\'][^>]*content=["\'][^"\']*url=([^"\'>\s]+)',
        html,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _extract_js_redirect(html: str) -> str | None:
    """Extract URL from simple JS redirects."""
    match = re.search(
        r'(?:window\.location|location\.href)\s*=\s*["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _extract_content_sync(html: str) -> tuple[str | None, str | None]:
    """Extract main content from HTML using trafilatura (synchronous)."""
    if not _HAS_TRAFILATURA:
        # Fallback: strip tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else None
        return text, title

    text_or_none: str | None = trafilatura.extract(html, include_comments=False, include_tables=True)
    metadata = trafilatura.extract_metadata(html)
    title = metadata.title if metadata and metadata.title else None
    return text_or_none, title


def _validate_content(text: str | None, title: str | None) -> bool:
    """Reject redirect stubs and empty content."""
    if not text or len(text.strip()) < 50:
        return False
    return True


def _create_fetch_fn(max_chars: int, credential_resolver: Any | None) -> Any:
    """Create a fetch function."""

    def _fetch(url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; 3tears/0.1)"}

        if credential_resolver is not None:
            try:
                extra_headers = credential_resolver(url)
                if extra_headers:
                    headers.update(extra_headers)
            except Exception:
                pass  # credential resolver failure is non-fatal

        try:
            with httpx.Client(timeout=30.0, follow_redirects=True, max_redirects=5) as client:
                resp = client.get(url, headers=headers)

                # Handle 429/403 with retry
                for _attempt in range(_MAX_RETRIES):
                    if resp.status_code not in (429, 403):
                        break
                    import time

                    time.sleep(1)
                    resp = client.get(url, headers=headers)

                resp.raise_for_status()
                html = resp.text

                # Check for meta refresh / JS redirects
                redirect_url = _extract_meta_refresh(html) or _extract_js_redirect(html)
                if redirect_url:
                    resp = client.get(redirect_url, headers=headers)
                    resp.raise_for_status()
                    html = resp.text

            text, title = _extract_content_sync(html)

            if not _validate_content(text, title):
                return tool_error("web_fetch", "fetch", "page content appears empty or is a redirect stub")

            result = ""
            if title:
                result = f"# {title}\n\n"
            result += text or ""

            if len(result) > max_chars:
                result = result[: max_chars - 20] + "\n\n[Content truncated]"

            return result

        except httpx.HTTPStatusError as exc:
            return tool_error("web_fetch", "fetch", f"HTTP {exc.response.status_code}")
        except Exception as exc:
            return tool_error("web_fetch", "fetch", str(exc))

    return _fetch


def create_web_fetch_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a web fetch tool.

    Optional config keys:
    - ``max_chars``: max output length (default 15000)
    - ``_credential_resolver``: callable(url) -> dict of extra headers
    """
    max_chars = config.get("max_chars", _MAX_CHARS)
    credential_resolver = config.get("_credential_resolver")

    return StructuredTool.from_function(
        func=_create_fetch_fn(max_chars, credential_resolver),
        name="web_fetch",
        description=description,
        args_schema=WebFetchInput,
    )
