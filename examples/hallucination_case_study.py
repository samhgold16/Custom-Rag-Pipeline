"""Reproduce the 2023 hallucination and the context changes that remove it.

The corpus covers 2030-2035; the question asks about 2023. Each variant changes one thing.

    uv run python examples/hallucination_case_study.py
"""

from localrag import RAGEngine, Settings

QUESTION = "what was the max qubits that Quantum mind reached in 2023?"

plain = RAGEngine(Settings(data_dir="data/quantummind"))
plain.ingest()
embedded = plain.variant(header_fields=("year",))  # separate collection, same Qdrant client
embedded.ingest()

variants = [
    ("raw chunks, plain context (original)", plain, "plain", False),
    ("year header embedded with the text", embedded, "plain", False),
    ("year header at prompt time only", plain, "header", False),
    ("[source | year] label + cite rule", plain, "tagged", False),
    ("scope guardrail (no LLM call)", plain, "header", True),
]

print(f"Q: {QUESTION}\n")
for label, engine, context, guardrail in variants:
    answer = engine.ask(QUESTION, context_format=context, guardrail=guardrail)
    years = [s.metadata.get("year") for s in answer.sources]
    verdict = "abstained" if answer.abstained else "ANSWERED"
    print(f"- {label}\n  retrieved years: {years or '-'}\n  [{verdict}] {answer.text}\n")
