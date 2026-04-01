import os
from dotenv import load_dotenv

# Importy z LlamaIndex
from llama_index.core import Document, StorageContext, KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# Importy z konfiguracji LLM
from core.llm_engine import setup_llm_environment

# 1. Wczytanie haseł i inicjalizacja środowiska
load_dotenv()
print("Inicjalizacja LLM...")
setup_llm_environment()


def get_graph_store():
    # Nawiązanie połączenia z bazą Neo4j.
    return Neo4jGraphStore(
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI"),
        database="praca"
    )


def build_knowledge_graph():
    # Główna funkcja budująca graf wiedzy z surowych tekstów.

    print("Łączenie z bazą Neo4j...")
    graph_store = get_graph_store()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # 2. Przygotowanie danych wejściowych

    tekst = """
    Przykładowy tekst.
    """

    # Tworzymy obiekt dokumentu zrozumiały dla LlamaIndex
    documents =[Document(text=tekst)]

    print("Generowanie...")

    try:
        # 3. Ekstrakcja i zapis do bazy
        index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            max_triplets_per_chunk=15,
            include_embeddings=True
        )
        print("Wiedza została pomyślnie wstrzyknięta do bazy Neo4j.")

    except Exception as e:
        print(f"Wystąpił błąd podczas budowania grafu: {e}")


if __name__ == "__main__":
    build_knowledge_graph()