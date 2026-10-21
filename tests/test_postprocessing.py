from core import graph_rag, vector_rag
from core.graph_rag import _deduplicate_sentences, _strip_brak_contradiction, postprocess_answer


def test_deduplicate_removes_repeated_sentence():
    text = "Firma powstała w 2019 roku. Siedziba jest w Gnieźnie. Firma powstała w 2019 roku."
    assert _deduplicate_sentences(text) == "Firma powstała w 2019 roku. Siedziba jest w Gnieźnie."


def test_deduplicate_ignores_case_and_whitespace():
    text = "Prezesem jest Marta Klejnot-Wasiak.\nPREZESEM  jest Marta Klejnot-Wasiak."
    assert _deduplicate_sentences(text) == "Prezesem jest Marta Klejnot-Wasiak."


def test_deduplicate_keeps_list_lines():
    text = "Parametry:\n- Stopień ochrony: IP68\n- Protokół: LoRaWAN"
    assert _deduplicate_sentences(text) == text


def test_strip_removes_brak_sentence_after_real_answer():
    text = "Prezesem jest Marta Klejnot-Wasiak od 2021 roku. Brak informacji o tym, kiedy objęła funkcję."
    assert _strip_brak_contradiction(text) == "Prezesem jest Marta Klejnot-Wasiak od 2021 roku."


def test_strip_keeps_answer_when_everything_is_brak():
    text = "Brak wystarczających danych w bazie wiedzy na ten temat."
    assert _strip_brak_contradiction(text) == text


def test_strip_handles_text_without_polish_diacritics():
    text = "Patent dotyczy algorytmu kalibracji. Brak wystarczajacych danych o urzedzie."
    assert _strip_brak_contradiction(text) == "Patent dotyczy algorytmu kalibracji."


def test_strip_removes_brak_line_from_list():
    # Regresja (R8, Q3): w starej wersji deduplikacja sklejała linie spacją, więc zdanie
    # "Brak informacji..." doklejone do listy bez kropki nie było usuwane
    text = "Parametry czujnika:\n- Cena netto: 1890 PLN\nBrak informacji o liczbie czujników."
    assert postprocess_answer(text) == "Parametry czujnika:\n- Cena netto: 1890 PLN"


def test_strip_keeps_paragraph_breaks():
    text = "Pierwszy akapit.\n\nDrugi akapit. Brak danych o trzecim."
    assert _strip_brak_contradiction(text) == "Pierwszy akapit.\n\nDrugi akapit."


def test_postprocess_combines_both_steps():
    text = "Grant wyniósł 12,4 mln PLN. Grant wyniósł 12,4 mln PLN. Brak informacji o roku grantu."
    assert postprocess_answer(text) == "Grant wyniósł 12,4 mln PLN."


def test_vector_and_graph_rag_share_postprocessing():
    # Warunek rzetelnego porównania: obie metody różnią się wyłącznie wyszukiwaniem trójek
    assert vector_rag.postprocess_answer is graph_rag.postprocess_answer
