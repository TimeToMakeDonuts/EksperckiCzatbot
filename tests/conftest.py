import sys
import types
from pathlib import Path

import pytest

# Katalog główny projektu na ścieżce importów (testy uruchamiane poleceniem `pytest`)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Atrapa core.llm_engine: prawdziwy moduł przy imporcie łączy się z serwerem LLM
# i ładuje model embeddingów (~1 GB), a testowane funkcje są deterministyczne
# i nie potrzebują modelu. Atrapę trzeba wstawić przed importem modułów core.
_fake_engine = types.ModuleType("core.llm_engine")
_fake_engine.llm = None
_fake_engine.judge_llm = None
_fake_engine._embed_initialized = True
_fake_engine.get_judge = lambda: _fake_engine.judge_llm or _fake_engine.llm
_fake_engine.get_standard_llm_response = lambda prompt: ""
_fake_engine.init_or_update_llm = lambda *args, **kwargs: None
_fake_engine.init_or_update_judge_llm = lambda *args, **kwargs: None
sys.modules["core.llm_engine"] = _fake_engine

import core  # noqa: E402

core.llm_engine = _fake_engine


class FakeLLM:
    # Model zwracający z góry ustaloną odpowiedź (albo rzucający wyjątek)
    # i zapamiętujący otrzymane prompty.
    def __init__(self, text: str = "", error: Exception | None = None):
        self.text = text
        self.error = error
        self.prompts: list[str] = []

    def complete(self, prompt: str):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return types.SimpleNamespace(text=self.text)


@pytest.fixture
def fake_llm(monkeypatch):
    # Podmienia główny model (i sędziego) na FakeLLM; zwraca fabrykę do ustawienia odpowiedzi.
    def _set(text: str = "", error: Exception | None = None) -> FakeLLM:
        model = FakeLLM(text, error)
        monkeypatch.setattr(_fake_engine, "llm", model)
        monkeypatch.setattr(_fake_engine, "judge_llm", None)
        return model
    return _set
