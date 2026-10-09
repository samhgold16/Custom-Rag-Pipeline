import httpx
import pytest

from localrag import providers
from localrag.config import Settings
from localrag.providers import (
    HealthReport,
    OllamaProvider,
    ProviderError,
    chat_template_problem,
    check_ollama,
    server_version,
)

TAGS = {
    "models": [
        {"name": "hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M", "digest": "50f17e0550ee"},
        {"name": "nomic-embed-text:latest", "digest": "0a109f422b47"},
    ]
}


def fake_http(monkeypatch, *, get=None, post=None):
    def respond(payload):
        def handler(url, **kwargs):
            if isinstance(payload, Exception):
                raise payload
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

        return handler

    if get is not None:
        monkeypatch.setattr(providers.httpx, "get", respond(get))
    if post is not None:
        monkeypatch.setattr(providers.httpx, "post", respond(post))


def test_check_ollama_accepts_implicit_latest_tag(monkeypatch):
    fake_http(monkeypatch, get=TAGS)
    report = check_ollama("http://x", ["hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M", "nomic-embed-text"])
    assert report.ok
    assert report.digests["nomic-embed-text"] == "0a109f422b47"


def test_check_ollama_reports_missing_models_with_pull_hint(monkeypatch):
    fake_http(monkeypatch, get=TAGS)
    report = check_ollama("http://x", ["llama-nope"])
    assert report.missing == ["llama-nope"]
    with pytest.raises(ProviderError, match="ollama pull llama-nope"):
        report.raise_for_status()


def test_unreachable_server(monkeypatch):
    fake_http(monkeypatch, get=httpx.ConnectError("refused"))
    report = check_ollama("http://x", ["m"])
    assert not report.reachable and report.error == "ConnectError"
    with pytest.raises(ProviderError, match="ollama serve"):
        report.raise_for_status()
    assert server_version("http://x") is None


def test_server_version(monkeypatch):
    fake_http(monkeypatch, get={"version": "0.35.1"})
    assert server_version("http://x") == "0.35.1"


@pytest.mark.parametrize(
    ("template", "problem"),
    [("{{ .Prompt }}", True), ("", True), ("{{- bos_token }}{%- for m in messages %}", False)],
)
def test_chat_template_problem(monkeypatch, template, problem):
    fake_http(monkeypatch, post={"template": template})
    assert (chat_template_problem("http://x", "m") is not None) is problem


def test_provider_builds_configured_models():
    provider = OllamaProvider(Settings(chat_model="c", embedding_model="e", temperature=0.3))
    chat = provider.chat_model()
    assert (chat.model, chat.temperature, chat.reasoning) == ("c", 0.3, False)
    assert provider.embeddings().model == "e"


def test_health_report_ok_property():
    assert HealthReport("h", True).ok
    assert not HealthReport("h", True, missing=["m"]).ok
