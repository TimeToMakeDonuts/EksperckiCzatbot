from core.graph_rag import _extract_keywords, _parse_llm_output, _stem5


def test_stem5_truncates_long_words():
    assert _stem5("grafowych") == "grafo"
    assert _stem5("relacyjnych") == "relac"


def test_stem5_keeps_short_words():
    assert _stem5("rag") == "rag"
    assert _stem5("model") == "model"


def test_parse_strips_prefix_and_splits_commas():
    raw = "Search terms: SoilMind Pro 3.0, cena, parametry techniczne"
    assert _parse_llm_output(raw) == ["soilmind pro 3.0", "cena", "parametry techniczne"]


def test_parse_handles_numbered_and_bulleted_lists():
    raw = "1. Kwas Grid\n2) centrum danych\n- platforma"
    assert _parse_llm_output(raw) == ["kwas grid", "centrum danych", "platforma"]


def test_parse_drops_stopwords_short_and_long_phrases():
    raw = "jakie, ab, " + "x" * 70 + ", patent"
    assert _parse_llm_output(raw) == ["patent"]


def test_parse_returns_at_most_six_phrases():
    raw = ", ".join(f"fraza{i}" for i in range(10))
    assert len(_parse_llm_output(raw)) == 6


def test_extract_keywords_puts_llm_phrases_first(fake_llm):
    fake_llm("ZorbScript, autor")
    keywords, phrases = _extract_keywords("Co to jest ZorbScript i kto jest jego autorem?")
    assert phrases == ["zorbscript", "autor"]
    assert keywords[:2] == ["zorbscript", "autor"]
    # dalej stemy słów z pytania, bez słów z listy _STOPWORDS ("jest") i bez duplikatów
    assert "autor" in keywords and "jest" not in keywords
    assert len(keywords) == len(set(keywords)) <= 12


def test_extract_keywords_adds_short_tokens_from_phrases(fake_llm):
    fake_llm("numer NIP firmy")
    keywords, _ = _extract_keywords("Jaki jest NIP firmy Zorbex?")
    assert "nip" in keywords


def test_extract_keywords_falls_back_to_stems_when_llm_fails(fake_llm):
    fake_llm(error=ConnectionError("serwer niedostępny"))
    keywords, phrases = _extract_keywords("Jakie nagrody zdobyła firma?")
    assert phrases == []
    assert keywords == ["nagro", "zdoby", "firma"]
