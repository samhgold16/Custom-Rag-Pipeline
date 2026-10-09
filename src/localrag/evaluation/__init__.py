"""Evaluation harness: gold datasets, deterministic metrics, a config grid, and reports."""

from localrag.evaluation.dataset import EvalItem, load_dataset
from localrag.evaluation.metrics import ItemResult, Summary, content_correct, summarize
from localrag.evaluation.report import render_markdown, write_report
from localrag.evaluation.runner import GRIDS, EvalConfig, EvalRun, run_eval

__all__ = [
    "GRIDS",
    "EvalConfig",
    "EvalItem",
    "EvalRun",
    "ItemResult",
    "Summary",
    "content_correct",
    "load_dataset",
    "render_markdown",
    "run_eval",
    "summarize",
    "write_report",
]
