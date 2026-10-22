import pandas as pd

from core import graph_builder
from core.graph_builder import (TRIPLET_COLUMNS, count_incomplete, extract_triplets_with_llm, parse_triplet_lines,
                                save_edited_triplets_to_neo4j)

S, R, O = TRIPLET_COLUMNS


def test_parse_basic_line():
    rows = parse_triplet_lines('Kosmokwaki | POWSTAŁ_W_ROKU | 2000')
    assert rows == [{S: 'Kosmokwaki', R: 'POWSTAŁ_W_ROKU', O: '2000'}]


def test_parse_skips_header_and_text_without_separator():
    response = 'Oto trójki:\nPodmiot | RELACJA | Obiekt\nKosmokwaki | ZADEBIUTOWAŁ_W | telewizja Polsat'
    assert parse_triplet_lines(response) == [{S: 'Kosmokwaki', R: 'ZADEBIUTOWAŁ_W', O: 'telewizja Polsat'}]


def test_parse_strips_list_markers_and_quotes():
    response = '1. Kosmokwaki | wykonuje utwór | „Brum brum”\n- Kosmokwaki | WYKONUJE_UTWÓR | "Ene due"'
    rows = parse_triplet_lines(response)
    assert [r[O] for r in rows] == ['Brum brum', 'Ene due']
    assert {r[R] for r in rows} == {'WYKONUJE_UTWÓR'}


def test_parse_replaces_underscores_in_nodes():
    rows = parse_triplet_lines('Kosmokwaki | ŚPIEWA | Kaczym_głosem')
    assert rows[0][O] == 'Kaczym głosem'


def test_parse_handles_markdown_table_rows():
    response = '| Podmiot | Relacja | Obiekt |\n|---|---|---|\n| Kwakot | JEST_CZŁONKIEM | Kosmokwaki |'
    assert parse_triplet_lines(response) == [{S: 'Kwakot', R: 'JEST_CZŁONKIEM', O: 'Kosmokwaki'}]


def test_parse_keeps_incomplete_triplet_for_editing():
    # niepełna trójka zostaje w tabeli, żeby użytkownik mógł ją uzupełnić
    rows = parse_triplet_lines('Piosenki | "BRUM, BRUM" |')
    assert rows == [{S: 'Piosenki', R: 'BRUM,_BRUM', O: ''}]
    assert count_incomplete(rows) == 1


def test_parse_drops_lines_with_wrong_number_of_parts():
    assert parse_triplet_lines('A | B\nA | B | C | D') == []


def test_parse_splits_quoted_title_list():
    # rzeczywista odpowiedź modelu 3B: trzy tytuły w jednym obiekcie
    rows = parse_triplet_lines('Kosmokwaki | WYKONUJE_UTWÓR | Brum, brum", "Ene due", "Misio komputerowy')
    assert [r[O] for r in rows] == ['Brum, brum', 'Ene due', 'Misio komputerowy']
    assert {r[S] for r in rows} == {'Kosmokwaki'}


def test_parse_splits_short_comma_list():
    rows = parse_triplet_lines('Kosmokwaki | DOMINUJĄCY_KOLOR | czerwony, niebieski i żółty')
    assert [r[O] for r in rows] == ['czerwony', 'niebieski', 'żółty']


def test_parse_does_not_split_address_or_description():
    response = ('Zorbex S.A. | SIEDZIBA | ul. Mieszka I 17, 62-200 Gniezno\n'
                'Kosmokwaki | POPULARNY_WŚRÓD | wśród najmłodszych, rodziców dzieci\n'
                'Kosmokwaki | EMANUJĄ | radość i optymizm, które z nich emanują')
    assert [r[O] for r in parse_triplet_lines(response)] == [
        'ul. Mieszka I 17, 62-200 Gniezno', 'wśród najmłodszych, rodziców dzieci',
        'radość i optymizm, które z nich emanują']


def test_count_incomplete_treats_none_as_empty():
    rows = [{S: 'A', R: 'R', O: 'B'}, {S: 'A', R: None, O: 'B'}]
    assert count_incomplete(rows) == 1


def test_extract_uses_parser_and_inserts_text(fake_llm):
    model = fake_llm('Firma {X} | MA_SIEDZIBĘ_W | Gdańsk')
    rows = extract_triplets_with_llm('Tekst z {klamrami}.')
    assert rows == [{S: 'Firma {X}', R: 'MA_SIEDZIBĘ_W', O: 'Gdańsk'}]
    assert 'Tekst z {klamrami}.' in model.prompts[0]


def test_extract_uses_given_model_and_token_limit(fake_llm):
    from conftest import FakeLLM
    default_model = fake_llm('A | R | B')
    judge = FakeLLM('Kosmokwaki | POWSTAŁ_W_ROKU | 2000')
    rows = extract_triplets_with_llm('tekst', model=judge, max_tokens=2048)
    assert rows == [{S: 'Kosmokwaki', R: 'POWSTAŁ_W_ROKU', O: '2000'}]
    assert judge.used_limits == [2048]          # wywołana kopia modelu z limitem 2048
    assert judge.max_tokens is None             # oryginalny model sędziego bez zmian
    assert default_model.prompts == []          # główny model nie był wywołany


def test_extract_without_limit_uses_model_as_is(fake_llm):
    model = fake_llm('A | R | B')
    extract_triplets_with_llm('tekst')
    assert model.used_limits == [None]


def test_extract_returns_empty_list_on_llm_error(fake_llm):
    fake_llm(error=ConnectionError())
    assert extract_triplets_with_llm('tekst') == []


class _FakeSession:
    def __init__(self, calls):
        self.calls = calls

    def run(self, query, **params):
        self.calls.append((query, params))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeDriver:
    def __init__(self, calls):
        self.calls = calls

    def session(self, database=None):
        return _FakeSession(self.calls)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_save_counts_saved_and_skipped_rows(monkeypatch):
    calls = []
    monkeypatch.setattr(graph_builder.GraphDatabase, 'driver', lambda *a, **k: _FakeDriver(calls))
    df = pd.DataFrame([
        {S: 'Kosmokwaki', R: 'POWSTAŁ_W_ROKU', O: '2000'},
        {S: 'Piosenki', R: 'BRUM_BRUM', O: None},       # puste pole -> pominięta
        {S: None, R: None, O: None},                     # całkiem pusty wiersz -> nie liczony
        {S: 'Kwakot', R: 'JEST CZŁONKIEM!', O: 'Kosmokwaki'},
    ])
    assert save_edited_triplets_to_neo4j(df) == (2, 1)
    assert [c[1] for c in calls] == [{'n1': 'Kosmokwaki', 'n2': '2000'}, {'n1': 'Kwakot', 'n2': 'Kosmokwaki'}]
    assert '`JESTCZŁONKIEM`' in calls[1][0]


def test_save_returns_none_on_connection_error(monkeypatch):
    def broken(*a, **k):
        raise ConnectionError()
    monkeypatch.setattr(graph_builder.GraphDatabase, 'driver', broken)
    assert save_edited_triplets_to_neo4j(pd.DataFrame([{S: 'A', R: 'R', O: 'B'}])) is None
