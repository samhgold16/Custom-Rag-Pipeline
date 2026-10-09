"""Ingest the demo corpus and answer one question with hybrid retrieval.

uv run python examples/quickstart.py
"""

from localrag import RAGEngine, RetrievalConfig, Settings

engine = RAGEngine(Settings(data_dir="data/quantummind"))
report = engine.ingest()  # idempotent: re-runs embed nothing unless files changed
print(f"{report.collection}: {report.chunks} chunks ({report.added} embedded this run)\n")

answer = engine.ask(
    "What alternative approach is QuantumMind exploring to generate near-term revenue?",
    retrieval=RetrievalConfig(strategy="hybrid", k=4),
)
print(answer.text, "\n")
for src in answer.sources:
    print(f"  {src.rank}. {src.source}  year={src.metadata.get('year')}  score={src.score:.3f}")
