from core.graph_rag import _rerank_triplets

PROMPT = "Kto jest głównym inwestorem i jaki grant otrzymała firma?"

TRIPLETS = [
    "Zorbex S.A. | ZALOZONO_W_ROKU | 2019",
    "Zorbex S.A. | NIP | 7842390156",
    "Zorbex S.A. | LICZBA_PRACOWNIKOW | 234",
    "Fundacja Przyszlosci Rolnictwa | JEST_INWESTOREM | Zorbex S.A.",
    "Fundacja Przyszlosci Rolnictwa | UDZIELILA_GRANTU_MLN_PLN | 12.4",
    "Fundacja Przyszlosci Rolnictwa | GRANT_W_ROKU | 2020",
]


def test_empty_list_is_returned_unchanged():
    assert _rerank_triplets([], ["zorbex"], PROMPT) == []


def test_hub_phrase_does_not_count_and_relevant_triplets_stay():
    # "zorbex s.a." występuje w 4 z 6 trójek (67% < 70%), a "fundacja" niesie sygnał
    result = _rerank_triplets(TRIPLETS, ["fundacja", "inwestor", "grant"], PROMPT)
    assert all("Fundacja" in t for t in result)
    assert len(result) == 3


def test_phrase_above_70_percent_is_treated_as_hub():
    # Gdyby "zorbex s.a." nie była hubem (5 z 6 trójek), dawałaby +2 trójkom firmy
    triplets = TRIPLETS[:3] + ["SoilMind Pro 3.0 | JEST_PRODUKTEM | Zorbex S.A."] + TRIPLETS[3:]
    result = _rerank_triplets(triplets, ["zorbex s.a.", "fundacja", "grant"], PROMPT)
    assert all("Fundacja" in t for t in result)


def test_returns_original_list_when_fewer_than_three_remain():
    result = _rerank_triplets(TRIPLETS, ["udzielila_grantu"], "grant")
    assert result == TRIPLETS


def test_returns_original_list_when_nothing_scores():
    result = _rerank_triplets(TRIPLETS, ["kwas grid"], "Gdzie jest centrum danych?")
    assert result == TRIPLETS
