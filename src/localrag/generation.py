"""Prompts, context formatting, and the answer chain. All prompt text lives in this module."""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from localrag.config import ContextFormat
from localrag.enrichers import format_header

# The original pipeline's system prompt, kept verbatim so the `plain` baseline is reproducible.
BASE_SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
)

# Added only for `tagged` context, where the labels exist. It asks for citations; it does not
# mention dates or out-of-scope questions, so any change in abstention behaviour comes from the
# labels themselves, not from a prompt written around the test case.
CITATION_RULE = (
    "Each passage starts with a bracketed label naming its source document. "
    "Cite the source of each fact you use in brackets."
)

PASSAGE_SEPARATOR = "\n\n"

ABSTENTION_PATTERN = re.compile(
    r"\b(?:"
    r"i don't know|i do not know|i'm not sure|i am not sure"
    r"|(?:is|are|was|were) not (?:mentioned|provided|available|stated|specified)"
    r"|(?:does not|doesn't|do not|don't) (?:mention|contain|provide|include|specify|say)"
    r"|no (?:relevant |specific )?(?:information|data)"
    r"|(?:cannot|can't|unable to) (?:answer|determine|be determined)"
    r")",
    re.IGNORECASE,
)


def is_abstention(text: str) -> bool:
    """Whether an answer declines to answer. Curly apostrophes are normalised first."""
    return bool(ABSTENTION_PATTERN.search(text.replace("’", "'")))


def passage_label(doc: Document, label_fields: Sequence[str]) -> str:
    """``[source: 2032.txt | year: 2032]``. Rendered at prompt time; never embedded."""
    parts = [f"source: {doc.metadata.get('source', 'unknown')}"]
    parts += [f"{f}: {doc.metadata[f]}" for f in label_fields if doc.metadata.get(f) is not None]
    return "[" + " | ".join(parts) + "]"


def _with_header(doc: Document, label_fields: Sequence[str]) -> str:
    header = format_header(doc.metadata, label_fields)
    # A chunk from a header-enriched collection already starts with its header.
    if not header or doc.page_content.startswith(header):
        return doc.page_content
    return f"{header}\n{doc.page_content}"


def format_context(
    documents: Sequence[Document], context_format: ContextFormat, label_fields: Sequence[str] = ()
) -> str:
    """Join passages for the prompt.

    ``plain`` shows stored text only. ``header`` prepends the same ``Metadata: year=2033`` line
    the ingest-time enricher writes, but without touching the embeddings. ``tagged`` prepends a
    ``[source: … | year: …]`` label.
    """
    if context_format == "plain":
        parts = [d.page_content for d in documents]
    elif context_format == "header":
        parts = [_with_header(d, label_fields) for d in documents]
    else:
        parts = [f"{passage_label(d, label_fields)}\n{d.page_content}" for d in documents]
    return PASSAGE_SEPARATOR.join(parts)


def system_prompt(context_format: ContextFormat) -> str:
    rules = (
        f"{BASE_SYSTEM_PROMPT} {CITATION_RULE}"
        if context_format == "tagged"
        else BASE_SYSTEM_PROMPT
    )
    return f"{rules}\n\n{{context}}"


class AnswerGenerator:
    """``prompt | llm | parser`` composed with LCEL, so the engine owns (and times) each stage."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def generate(
        self,
        question: str,
        documents: Sequence[Document],
        context_format: ContextFormat,
        label_fields: Sequence[str] = (),
    ) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt(context_format)), ("human", "{question}")]
        )
        chain = prompt | self.llm | StrOutputParser()
        context = format_context(documents, context_format, label_fields)
        return chain.invoke({"context": context, "question": question}).strip()
