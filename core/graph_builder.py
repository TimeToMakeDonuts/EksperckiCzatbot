import os
import re
from PyPDF2 import PdfReader
from neo4j import GraphDatabase
from core.llm_engine import llm

#Wyciągnięcie tekstu z pliku PDF.
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_triplets_with_llm(text: str) -> list:

    # Prompt wymuszujący na LLM określony format.
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
        response = llm.complete(prompt).text

        # Parsowanie odpowiedzi od LLM
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
    except Exception as e:
        print(f"Błąd ekstrakcji: {e}")
        return []

# Zapis tabeli Pandas DataFrame do bazy Neo4j.
def save_edited_triplets_to_neo4j(dataframe):
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    try:
        db_name = os.getenv("NEO4J_DATABASE", "praca")
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=db_name) as session:
                for index, row in dataframe.iterrows():
                    n1 = str(row["Obiekt 1 (Start)"]).strip()
                    rel = str(row["Relacja"]).strip()
                    n2 = str(row["Obiekt 2 (Koniec)"]).strip()

                    if not n1 or not rel or not n2:
                        continue

                    # Wyczyszczenie nazw relacji ze znaków specjalnych
                    rel_clean = re.sub(r'\W+', '', rel)
                    if not rel_clean:
                        continue

                    # Zapytanie Cypher
                    query = f"""
                    MERGE (a:Entity {{id: $n1}})
                    MERGE (b:Entity {{id: $n2}})
                    MERGE (a)-[:{rel_clean}]->(b)
                    """
                    session.run(query, n1=n1, n2=n2)
        return True
    except Exception as e:
        print(f"Błąd zapisu do Neo4j: {e}")
        return False