import os
import re
import pandas as pd
from PyPDF2 import PdfReader
from neo4j import GraphDatabase
from core import llm_engine
from core.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(uploaded_file) -> str:
    # Czyta wgrany plik PDF i zwraca cały jego tekst (ze wszystkich stron)
    # jako jeden ciąg znaków. Tabele i układ strony nie są zachowywane.
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"  # strona-skan bez tekstu zwraca None
    return text


TRIPLET_COLUMNS = ("Obiekt 1 (Start)", "Relacja", "Obiekt 2 (Koniec)")

# Zasady i przykład zamiany wyliczenia na kilka trójek — bez nich mały model (3B) wpisywał
# kategorie zamiast nazw ("Piosenki"), całe zdania jako relacje i listy w jednym obiekcie.
# Przykład celowo dotyczy innej dziedziny niż typowe teksty wejściowe.
_EXTRACTION_PROMPT = """Jesteś analitykiem danych grafowych. Zamień poniższy tekst na fakty w postaci trójek:
Podmiot | RELACJA | Obiekt

Zasady:
1. Każda linia to dokładnie jedna trójka z trzema NIEPUSTYMI elementami.
2. Podmiot to konkretna nazwa własna z tekstu (osoba, firma, zespół, produkt), a nie ogólna kategoria, np. "Piosenki" albo "Postacie".
3. Relacja to krótka nazwa czynności lub cechy (2-4 słowa) pisana WIELKIMI_LITERAMI z podkreślnikami, np. POWSTAŁ_W_ROKU, MA_SIEDZIBĘ_W, WYKONUJE_UTWÓR. Nie wpisuj do relacji wartości ani całego zdania.
4. Obiekt to jedna wartość: nazwa, liczba, data lub krótkie określenie. Nie pisz "Tak" ani "Nie".
5. Wyliczenie kilku elementów zapisz jako kilka trójek – po jednej na każdy element.
6. Dla faktów tego samego rodzaju używaj tej samej nazwy relacji.
7. Zapisuj tylko fakty z tekstu, bez ocen i własnych domysłów.

Przykład.
Tekst: Firma Alfa z Gdańska, założona w 2015 roku, produkuje rowery i hulajnogi. Prezesem firmy jest Anna Nowak.
Trójki:
Firma Alfa | MA_SIEDZIBĘ_W | Gdańsk
Firma Alfa | ZAŁOŻONA_W_ROKU | 2015
Firma Alfa | PRODUKUJE | rowery
Firma Alfa | PRODUKUJE | hulajnogi
Anna Nowak | JEST_PREZESEM | Firma Alfa

Tekst do analizy:
{text}

Trójki (tylko linie w formacie Podmiot | RELACJA | Obiekt, bez innego tekstu):"""

_QUOTES = '"\'„”“«»'
_LIST_MARKER_RE = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s+')
_HEADER_ROWS = {("podmiot", "relacja", "obiekt"), ("węzeł 1", "relacja", "węzeł 2")}


def _clean_node(value: str) -> str:
    # Węzeł: bez cudzysłowów na brzegach i bez podkreślników, które model wstawia zamiast spacji.
    value = value.strip().strip(_QUOTES).replace("_", " ")
    return re.sub(r'\s+', ' ', value).strip()


def _clean_relation(value: str) -> str:
    # Relacja: WIELKIE_LITERY z pojedynczymi podkreślnikami zamiast spacji.
    value = value.strip().strip(_QUOTES).upper().replace(" ", "_")
    return re.sub(r'_+', '_', value).strip("_")


_QUOTED_LIST_SEP = re.compile(r'["„“”]\s*(?:,|\bi\b|\boraz\b|\bczy\b)\s*["„“”]')


def _split_list_object(obj: str) -> list[str]:
    # Model często wpisuje wyliczenie do jednego obiektu mimo zasady 5 w prompcie.
    # Rozbijamy je deterministycznie tylko w dwóch bezpiecznych przypadkach:
    #   - lista tytułów w cudzysłowach: "Brum, brum", "Ene due" czy "Misio komputerowy",
    #   - co najmniej 3 krótkie elementy (do 3 słów) po przecinku: czerwony, niebieski, żółty.
    # Adresy i opisy ("ul. Mieszka I 17, 62-200 Gniezno") zostają bez zmian.
    if _QUOTED_LIST_SEP.search(obj):
        items = _QUOTED_LIST_SEP.split(obj)
    else:
        items = re.split(r',|\s+i\s+|\s+oraz\s+', obj)
        if len(items) < 3 or any(len(i.split()) > 3 or not i.strip() for i in items):
            return [_clean_node(obj)]
    items = [_clean_node(i) for i in items]
    return [i for i in items if i] or [_clean_node(obj)]


def parse_triplet_lines(response: str) -> list[dict]:
    # Zamienia odpowiedź modelu na wiersze tabeli edycji (słowniki z kolumnami TRIPLET_COLUMNS).
    # Pomija nagłówki i linie bez trzech części; usuwa numerację i wypunktowanie.
    # Trójki z pustym elementem są ZACHOWYWANE, żeby użytkownik mógł je uzupełnić
    # w tabeli (aplikacja ostrzega o nich, a zapis do Neo4j je pomija).
    rows = []
    for line in response.split('\n'):
        line = _LIST_MARKER_RE.sub('', line.strip())
        if '|' not in line or set(line) <= set('|-: '):  # brak separatora albo linia "---" tabeli Markdown
            continue
        if line.startswith('|'):  # wiersz tabeli Markdown: "| A | B | C |"
            line = line[1:]
            if line.endswith('|') and line.count('|') == 3:
                line = line[:-1]
        parts = line.split('|')
        if len(parts) != 3:
            continue
        subject, relation, obj = _clean_node(parts[0]), _clean_relation(parts[1]), _clean_node(parts[2])
        if (subject.lower(), relation.lower().replace("_", " "), obj.lower()) in _HEADER_ROWS:
            continue
        objects = _split_list_object(parts[2].strip()) if obj else [obj]
        for single in objects:
            rows.append(dict(zip(TRIPLET_COLUMNS, (subject, relation, single))))
    return rows


def count_incomplete(rows: list[dict]) -> int:
    # Liczba trójek, w których co najmniej jeden element jest pusty.
    return sum(1 for row in rows if not all(str(row.get(col) or "").strip() for col in TRIPLET_COLUMNS))


def extract_triplets_with_llm(text: str, model=None, max_tokens: int | None = None) -> list:
    # Prosi model LLM o wyciągnięcie z tekstu faktów w formacie "Podmiot | RELACJA | Obiekt"
    # i zamienia odpowiedź na listę słowników — wierszy tabeli edycji w zakładce "Baza Wiedzy".
    # model: który model ma wykonać ekstrakcję (domyślnie główny llm; np. większy model sędziego
    #        daje wyraźnie lepsze trójki, a czas nie jest tu krytyczny),
    # max_tokens: limit długości odpowiedzi tylko dla tego wywołania – domyślny limit odpowiedzi
    #        (512) ucina listę trójek dla dłuższych tekstów.
    # Przy błędzie modelu zwraca pustą listę.
    prompt = _EXTRACTION_PROMPT.replace("{text}", text)  # replace zamiast format(): tekst może zawierać klamry
    llm = model if model is not None else llm_engine.llm
    try:
        if max_tokens:
            # Kopia modelu z innym limitem: parametr max_tokens przekazany do complete() jest w LlamaIndex
            # nadpisywany limitem ustawionym w modelu (sprawdzone: odpowiedź i tak kończyła się na 512 tokenach)
            llm = llm.model_copy(update={"max_tokens": max_tokens})
        response = llm.complete(prompt).text
        return parse_triplet_lines(response)
    except Exception:
        logger.error("Błąd ekstrakcji trójek z tekstu przez LLM", exc_info=False)  # exc_info=False: pełny ślad błędu mógłby ujawnić treść promptu
        return []


def save_edited_triplets_to_neo4j(dataframe) -> tuple[int, int] | None:
    # Zapisuje trójki zatwierdzone przez użytkownika w tabeli do bazy Neo4j.
    # Używa MERGE, więc jeśli węzeł lub relacja już istnieją, nie powstają duplikaty.
    # Zwraca parę (liczba zapisanych trójek, liczba pominiętych z powodu pustego pola),
    # a None, gdy wystąpił błąd (np. brak połączenia z bazą).
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    saved = skipped = 0
    try:
        db_name = os.getenv("NEO4J_DATABASE", "praca")
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=db_name) as session:
                for _, row in dataframe.iterrows():
                    # puste komórki z tabeli edycji to None/NaN, nie wolno zamienić ich na tekst "None"
                    n1, rel, n2 = (
                        "" if pd.isna(row[col]) else str(row[col]).strip()
                        for col in TRIPLET_COLUMNS
                    )
                    # Usuń znaki specjalne z nazwy relacji (zostają litery, cyfry i _). Backticki w zapytaniu
                    # pozwalają użyć nazwy zaczynającej się od cyfry, np. "2019_ROK"
                    rel_clean = re.sub(r'\W+', '', rel)
                    if not (n1 or rel or n2):  # całkiem pusty wiersz tabeli – nie liczymy go jako pominiętego
                        continue
                    if not n1 or not rel_clean or not n2:
                        skipped += 1
                        continue
                    query = f"""
                    MERGE (a:Entity {{id: $n1}})
                    MERGE (b:Entity {{id: $n2}})
                    MERGE (a)-[:`{rel_clean}`]->(b)
                    """
                    session.run(query, n1=n1, n2=n2)
                    saved += 1
        return saved, skipped
    except Exception:
        logger.error("Błąd zapisu trójek do Neo4j", exc_info=False)  # exc_info=False: pełny ślad błędu mógłby ujawnić dane połączenia z bazą
        return None
