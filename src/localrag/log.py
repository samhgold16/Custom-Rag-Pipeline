"""Logging setup for the CLI. The library itself only creates loggers and never configures them."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(verbose: bool = False, console: Console | None = None) -> None:
    """Route library logs through Rich on stderr. ``verbose`` shows INFO from ``localrag``."""
    handler = RichHandler(
        console=console or Console(stderr=True), show_path=False, show_time=False, markup=False
    )
    logging.basicConfig(level=logging.WARNING, handlers=[handler], format="%(message)s", force=True)
    logging.getLogger("localrag").setLevel(logging.INFO if verbose else logging.WARNING)
    # One INFO line per HTTP request to Ollama is noise.
    logging.getLogger("httpx").setLevel(logging.WARNING)
