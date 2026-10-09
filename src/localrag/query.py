"""Query-side processing: the scope guardrail and two-step query rewriting."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from langchain_core.language_models import BaseChatModel

from localrag.types import GuardrailDecision

logger = logging.getLogger(__name__)

# Four-digit years in a plausible range. A bare \d{4} would read "1000-qubit" as a year.
_YEAR = re.compile(r"(?<![\d,.])(1[89]\d\d|2[01]\d\d)(?!\d)")


def extract_years(text: str) -> list[int]:
    """Distinct years mentioned in ``text``, in order of appearance."""
    seen: dict[int, None] = {}
    for match in _YEAR.finditer(text):
        seen.setdefault(int(match.group(1)), None)
    return list(seen)


def _span(values: Iterable[int]) -> str:
    ordered = sorted(values)
    if not ordered:
        return "nothing"
    if ordered == list(range(ordered[0], ordered[-1] + 1)) and len(ordered) > 2:
        return f"{ordered[0]}–{ordered[-1]}"
    return ", ".join(map(str, ordered))


def scope_guardrail(question: str, field: str, available: Iterable[int]) -> GuardrailDecision:
    """Decide whether a question's explicit years fall inside the indexed corpus.

    * No year in the question: ``pass``.
    * Every year outside the corpus: ``abstain``. The caller answers without retrieval or an
      LLM call, so the refusal does not depend on the model's calibration.
    * Any year inside: ``filter`` to those years, a lightweight self-query that turns a
      temporal constraint in the text into an exact payload filter.
    """
    available_t = tuple(sorted(set(available)))
    requested = tuple(extract_years(question))
    if not requested:
        return GuardrailDecision("pass", field, requested, available_t, "No year in the question.")
    inside = tuple(y for y in requested if y in available_t)
    if not inside:
        outside = _span(requested)
        return GuardrailDecision(
            "abstain",
            field,
            requested,
            available_t,
            f"The indexed documents cover {field} {_span(available_t)}; there is nothing for "
            f"{outside}.",
        )
    return GuardrailDecision(
        "filter", field, requested, available_t, f"Scoped the search to {field} {_span(inside)}."
    )


def guardrail_filter(decision: GuardrailDecision) -> dict[str, list[int]]:
    """Payload filter implied by a ``filter`` decision (empty otherwise)."""
    if decision.action != "filter":
        return {}
    return {decision.field: [y for y in decision.requested if y in decision.available]}


REWRITE_PROMPT = (
    "Rewrite the user's question as a standalone search query for a document collection.\n"
    "Rules:\n"
    "- Fix spelling, expand abbreviations, and replace vague references with explicit terms.\n"
    "- Keep every constraint exactly as given: dates, years, numbers, names, and entities. "
    "Never replace one year or number with another, and never add a constraint that is not there.\n"
    "- Do not answer the question.\n"
    "Reply with the rewritten query only, on one line.\n\n"
    "Question: {question}\n"
    "Rewritten query:"
)


class QueryRewriter:
    """Rewrite a question into a clearer search query with a separate LLM call.

    The rewrite is used only for retrieval; the generator still answers the original question.
    A rewrite that drops or alters a year from the original is rejected, which guards against
    the query drift seen when a single prompt was asked to rewrite *and* answer.
    """

    def __init__(
        self, llm: BaseChatModel, prompt: str = REWRITE_PROMPT, max_chars: int = 300
    ) -> None:
        self.llm = llm
        self.prompt = prompt
        self.max_chars = max_chars

    def rewrite(self, question: str) -> str:
        reply = str(self.llm.invoke(self.prompt.format(question=question)).content).strip()
        candidate = reply.splitlines()[0].strip().strip("\"'") if reply else ""
        if not candidate or len(candidate) > self.max_chars:
            logger.warning("Rewrite rejected (empty or too long); using the original question")
            return question
        if set(extract_years(question)) != set(extract_years(candidate)):
            logger.warning("Rewrite rejected (changed the years): %r", candidate)
            return question
        return candidate
