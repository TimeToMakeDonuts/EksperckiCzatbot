import os
from neo4j import GraphDatabase
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever

from core import llm_engine
from core.logger import get_logger
from core.graph_rag import _deduplicate_sentences

logger = get_logger(__name__)

# Globalny indeks wektorowy — budowany raz na sesję, przechowywany w RAM
_vector_index: VectorStoreIndex | None = None


def _load_triplets_from_neo4j(limit: int = 500) -> list[str]:
    # Pobiera trójki z Neo4j (maks. `limit`) jako teksty "podmiot | RELACJA | obiekt".
    # Ten sam format co w Graph RAG, dzięki temu obie metody pracują na identycznych
    # danych i można je uczciwie porównać.
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    db_name = os.getenv("NEO4J_DATABASE", "praca")

    texts: list[str] = []
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=db_name) as session:
                records = session.run(
                    "MATCH (n:Entity)-[r]->(m:Entity) "
                    "RETURN n.id AS src, type(r) AS rel, m.id AS tgt "
                    "LIMIT $lim",
                    lim=limit,
                )
                for rec in records:
                    texts.append(f"{rec['src']} | {rec['rel']} | {rec['tgt']}")
    except Exception:
        pass
    return texts


def build_vector_index() -> tuple[bool, int]:
    # Buduje indeks wektorowy: każda trójka z Neo4j staje się osobnym dokumentem
    # z własnym embeddingiem (wektorem liczbowym opisującym znaczenie tekstu).
    # Zwraca parę (czy się udało, ile trójek zaindeksowano).
    global _vector_index
    # Upewnij się, że model embeddingów jest zainicjowany przed indeksowaniem
    if not llm_engine._embed_initialized:
        llm_engine.init_or_update_llm()
    texts = _load_triplets_from_neo4j()
    if not texts:
        return False, 0
    documents = [Document(text=t) for t in texts]
    try:
        _vector_index = VectorStoreIndex.from_documents(
            documents,
            embed_model=Settings.embed_model,
            show_progress=False,
        )
        return True, len(texts)
    except Exception:
        _vector_index = None
        return False, 0


def is_index_built() -> bool:
    # Zwraca True, jeśli indeks wektorowy został już zbudowany.
    return _vector_index is not None


def get_vector_rag_response(prompt: str) -> dict[str, str]:
    # Główna funkcja potoku Vector RAG: pytanie użytkownika -> odpowiedź.
    # Kroki:
    #   1. sprawdź, czy indeks jest zbudowany,
    #   2. znajdź 10 trójek najbardziej podobnych znaczeniowo do pytania
    #      (podobieństwo kosinusowe embeddingów),
    #   3. wygeneruj odpowiedź modelem na podstawie tych trójek.
    # Zawsze zwraca słownik z kluczami "answer" i "context".
    if _vector_index is None:
        return {
            "answer": (
                "**[Brak indeksu wektorowego]**\n\n"
                "Kliknij **Zbuduj indeks wektorowy** w zakładce Ewaluacja "
                "przed uruchomieniem ewaluacji."
            ),
            "context": "",
        }

    # Wyszukiwanie semantyczne — top 10 trójek najbardziej podobnych do pytania
    try:
        retriever = VectorIndexRetriever(index=_vector_index, similarity_top_k=10)
        nodes = retriever.retrieve(prompt)
        context = "\n".join(node.get_content() for node in nodes)
    except Exception:
        logger.error("Błąd wyszukiwania wektorowego", exc_info=False)
        return {
            "answer": "**[Błąd wyszukiwania wektorowego]** Zbuduj indeks ponownie.",
            "context": "",
        }

    if not context.strip():
        return {
            "answer": "**Brak powiązań semantycznych dla tego zapytania.**",
            "context": "",
        }

    # Prompt identyczny jak w Graph RAG: różni się tylko sposób wyszukiwania trójek,
    # dzięki temu porównanie obu metod jest uczciwe
    answer_prompt = (
        "You are a knowledge base assistant. Answer questions using ONLY the facts below.\n\n"
        "KNOWLEDGE BASE FACTS (subject | relation | object):\n"
        f"{context}\n\n"
        f"USER QUESTION: {prompt}\n\n"
        "RULES:\n"
        "1. Write the answer in Polish.\n"
        "2. Use ONLY the facts listed above — no external knowledge, no assumptions, "
        "no domain explanations beyond what is explicitly stated.\n"
        "3. Use ALL facts that are relevant to the question — do not skip any relevant fact.\n"
        "4. Combine related facts into coherent sentences and paragraphs.\n"
        "5. Do NOT simply list the triplets — write natural flowing prose.\n"
        "6. If no facts are relevant, write: "
        "'Brak wystarczających danych w bazie wiedzy na ten temat.'\n"
        "7. When comparing two numeric values (e.g. year-over-year revenue), "
        "derive the direction from the numbers: a larger new value = increase/growth, smaller = decrease.\n"
        "8. Relation names encode units — use them to interpret the value correctly: "
        "_MLN_PLN = millions of PLN, _LAT = years, _W = watts, _KM = kilometres. "
        "Example: 'CZAS_PRACY_BATERII_LAT | 7' means '7 years of battery life', NOT 7 hours.\n\n"
        "Comprehensive answer in Polish:"
    )

    try:
        raw = llm_engine.llm.complete(answer_prompt).text.strip()
        answer = _deduplicate_sentences(raw)
        return {
            "answer": f"**Odpowiedź z RAG Wektorowego**\n\n{answer}",
            "context": context,
        }
    except Exception:
        logger.error("Błąd generowania odpowiedzi Vector RAG", exc_info=False)
        return {
            "answer": (
                "**[Błąd generowania odpowiedzi]** Sprawdź adres API i czy "
                "serwer LLM jest uruchomiony."
            ),
            "context": context,
        }
