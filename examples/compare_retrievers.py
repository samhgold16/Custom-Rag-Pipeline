"""Run one question through every retrieval strategy and show what each puts in the context.

uv run python examples/compare_retrievers.py "Did the company express frustration, and when?"
"""

import sys

from localrag import RAGEngine, RetrievalConfig, Settings

question = (
    sys.argv[1]
    if len(sys.argv) > 1
    else (
        "Did the company ever have a sense of frustration or aggression about the "
        "Quantum-AI space, in which year?"
    )
)
engine = RAGEngine(Settings(data_dir="data/quantummind"))
engine.ingest()

configs = {
    "dense": RetrievalConfig(strategy="dense"),
    "mmr (λ=0.5)": RetrievalConfig(strategy="mmr"),
    "hybrid BM25+dense (RRF)": RetrievalConfig(strategy="hybrid"),
    "dense + LLM re-rank": RetrievalConfig(rerank=True),
}

print(f"Q: {question}\n")
for name, config in configs.items():
    answer = engine.ask(question, retrieval=config, guardrail=False)
    ranked = ", ".join(f"{s.metadata.get('year')}({s.score:.3g})" for s in answer.sources)
    print(f"{name:<26} {ranked}\n{'':<26} → {answer.text}\n")
