"""Local Playwright adapter for the official Uni-TabRL HTML render template."""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path
from typing import Any

from reproduction.common.paths import project_path


OFFICIAL_TEMPLATE = project_path("StrucTab/code/Uni_TabRL/server/HTML_render/template.py")
DEFAULT_OUTPUT_DIR = project_path("reproduction/results/reward/rendered")
EMPTY_CELL_REPLACEMENTS = (
    (re.compile(r"<td>\s*</td>"), "<td>&nbsp;</td>"),
    (re.compile(r"<th>\s*</th>"), "<th>&nbsp;</th>"),
)


def _load_official_template():
    spec = importlib.util.spec_from_file_location("structab_official_render_template", OFFICIAL_TEMPLATE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official render template: {OFFICIAL_TEMPLATE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TEMPLATE = _load_official_template()


def fill_empty_cells(html: str) -> str:
    """Match the official renderer's empty-cell placeholder behavior."""
    for pattern, replacement in EMPTY_CELL_REPLACEMENTS:
        html = pattern.sub(replacement, html)
    return html


def stable_render_name(html: str) -> str:
    if not isinstance(html, str) or not html.strip():
        raise ValueError("render HTML must be a non-empty string")
    return f"table_render_{hashlib.sha256(html.encode('utf-8')).hexdigest()[:24]}.png"


def build_document(html: str, padding: int = 6, mathjax_url: str | None = None) -> str:
    """Build the official single-table document, optionally without MathJax."""
    html = fill_empty_cells(html)
    if mathjax_url:
        head = _TEMPLATE.HTML_HEAD.format(mathjax_url=mathjax_url)
    else:
        # The CSS is copied from the official template by removing only its two
        # MathJax scripts. This gives deterministic offline rendering for ordinary
        # text while making the missing formula engine explicit in diagnostics.
        head = """<head><meta charset="UTF-8"><style>
html { background: white; }
body { margin: 0; padding: 0; }
table { margin: 0; border-collapse: collapse; border: 2px solid black; }
td, th { border: 1px solid black; word-break: break-all; padding: 6px 12px; }
p { margin: 12px 0; line-height: 1.5; }
</style></head>"""
    return _TEMPLATE.SINGLE_TABLE_HTML_TEMPLATE.format(
        html_head=head, content=html, padding=padding
    )


class LocalHtmlRenderer:
    """Reusable local Chromium renderer with deterministic disk caching."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        *,
        padding: int = 6,
        viewport: tuple[int, int] = (2000, 2000),
        device_scale_factor: float = 2.0,
        mathjax_dir: Path | None = None,
    ):
        if padding < 0 or min(viewport) < 1 or device_scale_factor <= 0:
            raise ValueError("invalid render geometry")
        self.output_dir = output_dir.resolve()
        self.padding = padding
        self.viewport = viewport
        self.device_scale_factor = device_scale_factor
        self.mathjax_dir = mathjax_dir.resolve() if mathjax_dir else None
        self._playwright = None
        self._browser = None
        self._context = None

    @property
    def mathjax_url(self) -> str | None:
        if self.mathjax_dir is None:
            return None
        main = self.mathjax_dir / "es5" / "tex-svg.js"
        if not main.is_file():
            raise FileNotFoundError(f"MathJax entrypoint missing: {main}")
        return main.as_uri()

    def start(self) -> "LocalHtmlRenderer":
        if self._context is not None:
            return self
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("playwright is required for HTML rendering") from exc
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(
                headless=True,
                executable_path=self._playwright.chromium.executable_path,
                args=["--disable-dev-shm-usage", "--disable-gpu"],
            )
            self._context = self._browser.new_context(
                viewport={"width": self.viewport[0], "height": self.viewport[1]},
                device_scale_factor=self.device_scale_factor,
            )
        except Exception:
            self.close()
            raise
        return self

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._context = self._browser = self._playwright = None

    def __enter__(self) -> "LocalHtmlRenderer":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def render(self, html: str, *, image_name: str | None = None, reuse: bool = True) -> dict[str, Any]:
        self.start()
        name = Path(image_name or stable_render_name(html)).name
        if not name.lower().endswith(".png"):
            name += ".png"
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise ValueError("unsafe render image name")
        output = self.output_dir / name
        if reuse and output.is_file() and output.stat().st_size > 0:
            return {
                "status": "ok",
                "image_path": str(output),
                "cache_hit": True,
                "mathjax_enabled": self.mathjax_dir is not None,
            }

        document = build_document(html, self.padding, self.mathjax_url)
        page = self._context.new_page()
        try:
            page.set_content(document, wait_until="load")
            if self.mathjax_dir is not None:
                page.wait_for_function(
                    "window.MathJax && MathJax.startup && MathJax.startup.promise",
                    timeout=30_000,
                )
                page.evaluate("MathJax.startup.promise")
                page.evaluate("MathJax.typesetPromise()")
            page.evaluate("document.fonts ? document.fonts.ready : Promise.resolve()")
            target = page.locator("body > div").first
            target.wait_for(state="visible")
            target.screenshot(path=str(output))
        finally:
            page.close()
        return {
            "status": "ok",
            "image_path": str(output),
            "cache_hit": False,
            "mathjax_enabled": self.mathjax_dir is not None,
        }
