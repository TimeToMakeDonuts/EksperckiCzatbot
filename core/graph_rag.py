import os
from dotenv import load_dotenv

from llama_index.core import StorageContext, KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore

load_dotenv()


def get_graph_store():
    try:
        graph_store = Neo4jGraphStore(
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            url=os.getenv("NEO4J_URI"),
            database=os.getenv("NEO4J_DATABASE", "praca")
        )
        return graph_store
    except Exception:
        return None


def is_graph_empty(graph_store) -> bool:
    try:
        result = graph_store.query("MATCH (n) RETURN count(n) as count")
        return result[0]['count'] == 0
    except Exception:
        return True


def get_graph_rag_response(prompt: str) -> dict[str, str]:
    graph_store = get_graph_store()

    if not graph_store:
        return {
            "answer": "**[Błąd krytyczny]** Nie udało się połączyć z bazą Neo4j.",
            "context": ""
        }

    if is_graph_empty(graph_store):
        return {
            "answer": (
                "**Brak danych w grafie wiedzy.**\n\n"
                "Twoja baza Neo4j jest obecnie pusta. Model językowy nie może wygenerować "
                "zapytania (Cypher) dla pustego schematu. Aby RAG działał, musimy najpierw "
                "wstrzyknąć wiedzę do bazy."
            ),
            "context": ""
        }

    try:
        storage_context = StorageContext.from_defaults(graph_store=graph_store)

        index = KnowledgeGraphIndex.from_documents(
            documents=[],
            storage_context=storage_context
        )

        query_engine = index.as_query_engine(
            include_text=False,
            response_mode="tree_summarize",
            verbose=True
        )

        response = query_engine.query(prompt)

        if not str(response) or "Empty Response" in str(response):
            return {
                "answer": "**Brak powiązań w grafie dla tego zapytania.**",
                "context": ""
            }

        raw_context = "\n".join([node.node.text for node in response.source_nodes])

        return {
            "answer": f"**Odpowiedź z Grafu Neo4j**\n\n{str(response)}",
            "context": raw_context
        }

    except Exception as e:
        return {
            "answer": f"**[Błąd silnika RAG]** Szczegóły: {e}",
            "context": ""
        }
