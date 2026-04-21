import os
from dotenv import load_dotenv

from llama_index.core import StorageContext, KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore

load_dotenv()

    #inicjalizacja z bazą danych
def get_graph_store():
    try:
        graph_store = Neo4jGraphStore(
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            url=os.getenv("NEO4J_URI"),
            database="praca"
        )
        return graph_store
    except Exception as e:
        return None

    #sprawdzenie czy baza jest pusta
def is_graph_empty(graph_store) -> bool:
    try:
        result = graph_store.query("MATCH (n) RETURN count(n) as count")
        return result[0]['count'] == 0
    except Exception:
        return True

    # funkcja realizująca architekturę Graph RAG.
def get_graph_rag_response(prompt: str) -> str | dict[str, str]:
    graph_store = get_graph_store()

    if not graph_store:
        return "**[Błąd krytyczny]** Nie udało się połączyć z bazą Neo4j."

    # Zabezpieczenie przed błędem Pydantic (NoneType template)
    if is_graph_empty(graph_store):
        return (f"**Brak danych w grafie wiedzy.**\n\n"
                f"Twoja baza Neo4j jest obecnie pusta. Model językowy nie może wygenerować "
                f"zapytania (Cypher) dla pustego schematu. Aby RAG działał, musimy najpierw "
                f"wstrzyknąć wiedzę do bazy.")

    try:
        storage_context = StorageContext.from_defaults(graph_store=graph_store)

        # Inicjalizacja LlamaIndex (przez KnowledgeGraphIndex)
        # Tworzy to obiekt indeksu, który poprawnie binduje wewnętrzne prompty
        index = KnowledgeGraphIndex.from_documents(
            documents=[],
            storage_context=storage_context
        )

        # Wygenerowanie silnika odpytującego
        query_engine = index.as_query_engine(
            include_text=False,
            response_mode="tree_summarize",
            verbose=True
        )

        response = query_engine.query(prompt)

        if not str(response) or "Empty Response" in str(response):
            return "**Brak powiązań w grafie dla tego zapytania.**"

        raw_context = "\n".join([node.node.text for node in response.source_nodes])

        return {
            "answer": f"**Odpowiedź z Grafu Neo4j**\n\n{str(response)}",
            "context": raw_context
        }

    except Exception as e:
        return f"**[Błąd silnika RAG]** Szczegóły: {e}"