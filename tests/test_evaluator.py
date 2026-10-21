from evaluation.evaluator import (
    classify_error,
    judge_faithfulness,
    parse_faithfulness_output,
    score_citation_precision,
)

CONTEXT = (
    "Zorbex S.A. | ZALOZONO_W_ROKU | 2019\n"
    "Zorbex S.A. | ODDZIAL_CENTRALA_RD | Gniezno\n"
    "Zorbex S.A. | NIP | 7842390156"
)


def test_parse_reads_score_line():
    raw = "- Firma powstała w 2019 -> YES\n- Siedziba w Poznaniu -> NO\nSCORE: 50"
    assert parse_faithfulness_output(raw) == (50, "score")


def test_parse_counts_yes_no_when_score_line_missing():
    raw = "- twierdzenie 1 -> YES\n- twierdzenie 2 -> YES\n- twierdzenie 3 -> NO\n- twierdzenie 4 -> YES"
    assert parse_faithfulness_output(raw) == (75, "yes_no")


def test_parse_falls_back_to_last_number():
    # Np. odpowiedź ucięta przez limit tokenów, z innymi strzałkami niż "->"
    raw = "- Firma ma 234 pracowników → YES\n- EBITDA wynosi 8"
    assert parse_faithfulness_output(raw) == (8, "last_number")


def test_parse_clamps_score_to_100():
    assert parse_faithfulness_output("SCORE: 140") == (100, "score")


def test_parse_returns_none_without_numbers():
    assert parse_faithfulness_output("Nie potrafię ocenić.") == (None, "none")


def test_judge_faithfulness_returns_raw_output_and_layer(fake_llm):
    fake_llm("- Firma powstała w 2019 -> YES\nSCORE: 100")
    result = judge_faithfulness("Firma powstała w 2019 roku.", CONTEXT)
    assert result == {"score": 100, "raw": "- Firma powstała w 2019 -> YES\nSCORE: 100", "layer": "score"}


def test_judge_faithfulness_skips_empty_context(fake_llm):
    model = fake_llm("SCORE: 100")
    assert judge_faithfulness("Odpowiedź", "")["score"] is None
    assert model.prompts == []


def test_judge_faithfulness_handles_judge_error(fake_llm):
    fake_llm(error=TimeoutError())
    assert judge_faithfulness("Odpowiedź", CONTEXT) == {"score": None, "raw": "", "layer": "error"}


def test_citation_precision_counts_triplets_with_two_parts_in_answer():
    answer = "Zorbex S.A. została założona w 2019 roku, a centrala R&D mieści się w Gnieźnie."
    # zacytowana tylko pierwsza trójka: "Gnieźnie" to inna forma niż "Gniezno" w grafie
    assert score_citation_precision(answer, CONTEXT) == round(1 / 3, 3)


def test_citation_precision_none_for_empty_input():
    assert score_citation_precision("", CONTEXT) is None
    assert score_citation_precision("Odpowiedź", "  ") is None


def test_classify_error_labels():
    assert classify_error(90, 30, 100, "") == "Brak kontekstu w grafie"
    assert classify_error(90, 30, 20, CONTEXT) == "Halucynacja RAG"
    assert classify_error(None, 30, 90, CONTEXT) == "Błąd obliczenia metryki"
    assert classify_error(80, 30, 90, CONTEXT) == "OK"
    assert classify_error(60, 30, 90, CONTEXT) == "Odpowiedź częściowa"
    assert classify_error(20, 40, 90, CONTEXT) == "RAG gorszy od LLM"
    assert classify_error(30, 40, 90, CONTEXT) == "Zła odpowiedź"
