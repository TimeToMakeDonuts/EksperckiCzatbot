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


def extract_triplets_with_llm(text: str) -> list:
    # Prosi model LLM o wyciągnięcie z tekstu relacji w formacie
    # "Węzeł 1 | Relacja | Węzeł 2" i zamienia odpowiedź na listę słowników.
    # Każdy słownik to jeden wiersz tabeli edycji w zakładce "Baza Wiedzy" (st.data_editor).
    # Przy błędzie modelu zwraca pustą listę.
    prompt = f"""
    Jesteś analitykiem danych grafowych. Z poniższego tekstu wyciągnij najważniejsze
    relacje między encjami.
    Musisz zwrócić wynik DOKŁADNIE w formacie: Węzeł 1 | Relacja | Węzeł 2

    Przykład:
    Jan Kowalski | JEST_SZEFEM | Firma X

    Tekst do analizy:
    {text}

    Wypisz tylko relacje linijka po linijce, bez żadnego innego tekstu wstępnego:
    """
    try:
        response = llm_engine.llm.complete(prompt).text
        # Parsuj odpowiedź LLM: każda linia z separatorem | to jedna trójka
        triplets = []
        for line in response.split('\n'):
            parts = line.split('|')
            if len(parts) == 3:
                triplets.append({
                    "Obiekt 1 (Start)": parts[0].strip(),
                    "Relacja": parts[1].strip().upper().replace(" ", "_"),
                    "Obiekt 2 (Koniec)": parts[2].strip()
                })
        return triplets
    except Exception:
        logger.error("Błąd ekstrakcji trójek z tekstu przez LLM", exc_info=False)  # exc_info=False: pełny ślad błędu mógłby ujawnić treść promptu
        return []


def save_edited_triplets_to_neo4j(dataframe):
    # Zapisuje trójki zatwierdzone przez użytkownika w tabeli do bazy Neo4j.
    # Używa MERGE, więc jeśli węzeł lub relacja już istnieją, nie powstają duplikaty.
    # Zwraca True, gdy zapis się udał, a False, gdy wystąpił błąd (np. brak połączenia).
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    try:
        db_name = os.getenv("NEO4J_DATABASE", "praca")
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=db_name) as session:
                for _, row in dataframe.iterrows():
                    # puste komórki z tabeli edycji to None/NaN, nie wolno zamienić ich na tekst "None"
                    n1, rel, n2 = (
                        "" if pd.isna(row[col]) else str(row[col]).strip()
                        for col in ("Obiekt 1 (Start)", "Relacja", "Obiekt 2 (Koniec)")
                    )
                    if not n1 or not rel or not n2:
                        continue
                    # Usuń znaki specjalne z nazwy relacji (zostają litery, cyfry i _). Backticki w zapytaniu
                    # pozwalają użyć nazwy zaczynającej się od cyfry, np. "2019_ROK"
                    rel_clean = re.sub(r'\W+', '', rel)
                    if not rel_clean:
                        continue
                    query = f"""
                    MERGE (a:Entity {{id: $n1}})
                    MERGE (b:Entity {{id: $n2}})
                    MERGE (a)-[:`{rel_clean}`]->(b)
                    """
                    session.run(query, n1=n1, n2=n2)
        return True
    except Exception:
        logger.error("Błąd zapisu trójek do Neo4j", exc_info=False)  # exc_info=False: pełny ślad błędu mógłby ujawnić dane połączenia z bazą
        return False
