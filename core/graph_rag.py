import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
# Import całego modułu (a nie `from ... import llm`), żeby zawsze używać najnowszego modelu
# utworzonego przez init_or_update_llm() po zmianie ustawień
from core import llm_engine
from core.logger import get_logger

logger = get_logger(__name__)
load_dotenv()


def _make_driver():
    # Łączy się z bazą Neo4j na podstawie zmiennych środowiskowych
    # (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) i sprawdza, czy baza odpowiada.
    # Zwraca obiekt połączenia albo None, gdy połączenie się nie udało.
    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )
        driver.verify_connectivity()
        return driver
    except Exception:
        return None

def is_graph_empty() -> bool:
    # Sprawdza, czy w bazie Neo4j są jakiekolwiek węzły.
    # Brak połączenia z bazą też traktujemy jak "pusta baza" (zwraca True).
    driver = _make_driver()
    if not driver:
        return True
    try:
        db = os.getenv("NEO4J_DATABASE", "praca")
        with driver.session(database=db) as session:
            count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            return count == 0
    except Exception:
        return True
    finally:
        driver.close()


# Słowa funkcyjne odfiltrowane z wyników ekstrakcji słów kluczowych
# (polskie i angielskie, bo małe modele czasem odpowiadają po angielsku)
_STOPWORDS = {
    "jakie", "jaką", "jaki", "jakim", "jakich", "czym", "jest", "są", "jak",
    "się", "tego", "tej", "które", "który", "którą", "oraz", "dla", "przy",
    "przez", "jako", "czy", "ile", "gdzie", "kiedy", "dlaczego", "między",
    "porównaniu", "względem", "wobec", "czego", "sobie", "temu", "będzie",
    "mogą", "można", "mają", "jego", "tego", "bardzo",
    "what", "which", "that", "this", "with", "from", "have", "been",
    "about", "does", "more", "than", "also", "their", "when", "there",
}


def _parse_llm_output(raw: str) -> list[str]:
    # Zamienia odpowiedź modelu ze słowami kluczowymi na listę fraz.
    # Małe modele odpowiadają w różnych formatach, więc funkcja radzi sobie z:
    # numeracją ("1. termin"), wypunktowaniem ("- termin"), listą po przecinku
    # ("a, b, c") i doklejonym prefiksem ("Search terms: ...").
    # Odrzuca frazy zbyt krótkie, zbyt długie oraz słowa z listy _STOPWORDS.
    # Zwraca maksymalnie 6 fraz.
    # Usuń prefiksy które małe modele często doklejają przed właściwą odpowiedzią
    raw = re.sub(
        r'(?i)^(search\s+terms?|terms?|keywords?|key\s+(terms?|phrases?)|terminy?)\s*[:：]?\s*',
        '', raw.strip()
    )
    # Usuń numerację i symbole listy
    raw = re.sub(r'^\s*[\d]+[.)]\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^\s*[-•]\s*', '', raw, flags=re.MULTILINE)

    # Podziel po przecinkach, średnikach i nowych liniach
    parts = re.split(r'[,;\n]+', raw)
    result = []
    for p in parts:
        clean = p.strip().strip('"\'()[]').lower()
        if 2 < len(clean) < 60 and clean not in _STOPWORDS:
            result.append(clean)
    return result[:6]


_BRAK_RE = re.compile(
    r'\bbrak\s+(informacji|wystarczaj[aą]cych|danych|wiadomo[śs]ci)',
    re.IGNORECASE,
)


def _strip_brak_contradiction(text: str) -> str:
    # Usuwa z odpowiedzi zdania typu "Brak informacji o ...", ale tylko wtedy,
    # gdy odpowiedź zawiera też właściwą treść. Model 3B często dopisuje takie zdanie
    # po udzieleniu pełnej odpowiedzi, co jest sprzeczne z resztą tekstu i obniża
    # faithfulness. Jeśli WSZYSTKIE zdania mówią "brak danych", tekst zostaje
    # bez zmian, bo wtedy model faktycznie nie miał danych.
    parts = re.split(r'(?<=[.!?])\s+|\n', text)
    real = [p for p in parts if p.strip() and not _BRAK_RE.search(p)]
    if not real:
        return text
    return ' '.join(p for p in parts if p.strip() and not _BRAK_RE.search(p))


def _deduplicate_sentences(text: str) -> str:
    # Usuwa powtórzone zdania z odpowiedzi modelu. Hermes-3 3B potrafi wypisać
    # ten sam akapit kilka razy z drobnymi zmianami. Dwa zdania uznajemy za to samo,
    # gdy mają takie same pierwsze 60 znaków (po usunięciu spacji, bez rozróżniania
    # wielkości liter).
    # Podziel tekst na zdania po znakach kończących lub nowej linii
    parts = re.split(r'(?<=[.!?])\s+|\n', text)
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        # Klucz: pierwsze 60 znaków bez spacji, małe litery — wystarczy do wykrycia duplikatu
        key = re.sub(r'\s+', '', stripped[:60]).lower()
        if key and key not in seen:
            seen.add(key)
            result.append(stripped)
    return ' '.join(result)


def _stem5(word: str) -> str:
    # Prosty "stemming": obcina słowo do pierwszych 5 znaków, żeby różne formy
    # fleksyjne trafiały na ten sam rdzeń. Przykłady: "modele" -> "model",
    # "grafowych" -> "grafo", "relacyjnych" -> "relac".
    return word[:5] if len(word) > 5 else word


def _extract_keywords(prompt: str) -> tuple[list[str], list[str]]:
    # Wyciąga z pytania słowa kluczowe do przeszukania grafu dwoma sposobami:
    #   1. LLM - model podaje frazy w formie podstawowej (mianownik),
    #   2. stemming - każde słowo z pytania jest obcinane do 5 znaków.
    # Obie listy są łączone (frazy z LLM mają pierwszeństwo).
    # Zwraca parę:
    #   - wszystkie słowa kluczowe do wyszukiwania (maks. 12),
    #   - same frazy z LLM, używane później w _rerank_triplets. Trzymamy je osobno,
    #     bo pasowanie samym obciętym słowem bywa błędne (np. "danyc" z "centrum danych"
    #     pasuje do węzła "grafowa baza danych").
    llm_phrases: list[str] = []
    llm_keywords: list[str] = []

    # Prompt po angielsku — lepsze wyniki dla małych modeli niż po polsku
    extract_prompt = (
        "Extract 3-5 key noun phrases from the question as database search terms.\n"
        "Use base/dictionary form. Return ONLY terms separated by commas, nothing else.\n\n"
        f"Question: {prompt}\n\n"
        "Search terms:"
    )
    try:
        raw = llm_engine.llm.complete(extract_prompt).text.strip()
        parsed = _parse_llm_output(raw)
        llm_phrases = list(parsed)  # zachowane osobno do post-filtrowania
        llm_keywords = list(parsed)
        # Rozbij frazy wielosłowne tylko na krótkie tokeny (skróty: LLM, RAG, NIP)
        # Dłuższe tokeny fraz są niepotrzebne — ich stemy i tak trafiają przez stem_keywords.
        # Rozszerzanie długich tokenów wypychałoby ważne stemy (np. "funkc", "udzie") za limit listy.
        for phrase in parsed:
            for token in phrase.split():
                token = token.strip().lower()
                if 3 <= len(token) <= 4 and token not in _STOPWORDS:
                    llm_keywords.append(token)
    except Exception:
        pass  # przy błędzie LLM fallback do stemowanych słów z pytania

    # Stemowane słowa bezpośrednio z pytania użytkownika (min 3 znaki — łapie skróty LLM, RAG)
    raw_words = re.findall(r'\b\w{3,}\b', prompt.lower())
    stem_keywords = [_stem5(w) for w in raw_words if w not in _STOPWORDS]

    # Usuń duplikaty zachowując kolejność; LLM keywords mają pierwszeństwo
    combined = list(dict.fromkeys(llm_keywords + stem_keywords))
    return combined[:12], llm_phrases


# Cache wyników wyszukiwania: klucz → lista trójek
_MAX_CACHE = 128  # limit wpisów, żeby cache nie rósł w nieskończoność podczas długiej sesji
_retrieval_cache: dict[str, list[str]] = {}


def clear_retrieval_cache() -> None:
    # Czyści pamięć podręczną wyników wyszukiwania.
    # Trzeba ją wywołać po załadowaniu nowych danych do grafu, bo stare wyniki
    # byłyby nieaktualne.
    _retrieval_cache.clear()
    logger.info("Cache wyszukiwania wyczyszczony")


def _rerank_triplets(triplets: list[str], llm_phrases: list[str], prompt: str) -> list[str]:
    # Odrzuca trójki niezwiązane z pytaniem, punktując każdą z nich:
    #   - +2 za każdą frazę z LLM zawartą w trójce. Frazy pasujące do ponad 70% trójek
    #     są pomijane jako "huby" (np. nazwa firmy występuje wszędzie i nic nie odróżnia),
    #   - +1 za każdy obcięty rdzeń słowa z pytania zawarty w trójce.
    # Zostają trójki z wynikiem co najmniej połowy najlepszego wyniku.
    # Zabezpieczenie: jeśli po filtrowaniu zostałyby mniej niż 3 trójki albo nikt
    # nie zdobył punktów, zwracamy oryginalną listę.
    if not triplets:
        return triplets

    stem_keywords = [_stem5(w) for w in re.findall(r'\b\w{3,}\b', prompt.lower()) if w not in _STOPWORDS]

    # Wyklucz frazy-huby: fraza pasująca do > 70% trójek (np. nazwa firmy) nie daje sygnału.
    # Próg 70% zamiast 50% chroni frazy tematyczne (np. "soilmind pro 3.0" przy ~65% coverage)
    # przed błędnym wykluczeniem — wyłącza tylko globalne huby jak "zorbex s.a." (> 80%).
    n = len(triplets)
    triplets_lower = [t.lower() for t in triplets]
    effective_phrases = [
        p for p in llm_phrases
        if len(p) >= 3 and sum(1 for tl in triplets_lower if p.lower() in tl) < n * 0.7
    ]

    def score(triplet: str) -> int:
        t_low = triplet.lower()
        phrase_s = sum(2 for p in effective_phrases if p.lower() in t_low)
        stem_s = sum(1 for s in stem_keywords if len(s) >= 4 and s in t_low)
        return phrase_s + stem_s

    scored = [(t, score(t)) for t in triplets]
    max_s = max(s for _, s in scored)

    if max_s == 0:
        return triplets

    threshold = max(1, max_s // 2)
    filtered = [t for t, s in scored if s >= threshold]

    return filtered if len(filtered) >= 3 else triplets


def _fetch_entity_neighbors(filtered: list[str], all_triplets: list[str]) -> list[str]:
    # Dociąga z Neo4j dodatkowe fakty o "rzadkich" podmiotach, czyli takich, które
    # występują w wyfiltrowanych trójkach dokładnie raz i nie są hubami (hub = węzeł
    # w ponad 30% wszystkich pobranych trójek).
    # Po co: pytanie może nie zawierać słowa łączącego z potrzebnym faktem. Np. gdy
    # znaleziono inwestora, a pytanie nie ma słowa "grant", ta funkcja pobierze
    # UDZIELILA_GRANTU_MLN_PLN wprost z grafu (to jest właśnie przeszukiwanie grafu).
    # Zwraca listę nowych trójek (bez tych, które już mamy), maksymalnie 10.
    if not filtered or not all_triplets:
        return []

    # Wyznacz huby: węzły w > 30% wszystkich pobranych trójek
    n = len(all_triplets)
    node_counts: dict[str, int] = {}
    for t in all_triplets:
        parts = t.split(" | ")
        for part in ([parts[0]] if parts else []) + ([parts[2]] if len(parts) >= 3 else []):
            node_counts[part] = node_counts.get(part, 0) + 1
    hubs = {node for node, c in node_counts.items() if c > n * 0.3}

    # Znajdź "rzadkie" podmioty: dokładnie 1 trafienie w filtered, nie-hub
    subj_counts: dict[str, int] = {}
    for t in filtered:
        parts = t.split(" | ")
        if parts:
            subj_counts[parts[0]] = subj_counts.get(parts[0], 0) + 1
    rare = {s for s, c in subj_counts.items() if c == 1 and s not in hubs}

    if not rare:
        return []

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    db_name = os.getenv("NEO4J_DATABASE", "praca")

    already = set(filtered)
    extra: list[str] = []
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=db_name) as session:
                for subj in rare:
                    records = session.run(
                        """
                        MATCH (n:Entity {id: $subj})-[r]->(m:Entity)
                        RETURN n.id AS src, type(r) AS rel, m.id AS tgt
                        LIMIT 10
                        """,
                        subj=subj,
                    )
                    for rec in records:
                        t = f"{rec['src']} | {rec['rel']} | {rec['tgt']}"
                        if t not in already:
                            extra.append(t)
                            already.add(t)
    except Exception:
        logger.error("Błąd graph traversal w Neo4j", exc_info=False)

    return extra[:10]  # limit — zapobiega eksplozji kontekstu gdy jest wiele rzadkich podmiotów


def _retrieve_triplets(keywords: list[str]) -> list[str]:
    # Wyszukuje w Neo4j trójki pasujące do słów kluczowych (Cypher CONTAINS
    # na nazwach węzłów i typach relacji). Wyniki trafiają do pamięci podręcznej,
    # więc to samo zapytanie nie odpytuje bazy drugi raz.
    # Zwraca listę tekstów w formacie "podmiot | RELACJA | obiekt".
    cache_key = "|".join(sorted(kw for kw in keywords if kw))  # sorted → order-independent key
    if cache_key in _retrieval_cache:
        logger.debug("Cache hit dla kluczy: %s", cache_key[:60])
        return _retrieval_cache[cache_key]

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    db_name = os.getenv("NEO4J_DATABASE", "praca")

    found: set[str] = set()
    query_failed = False
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=db_name) as session:
                for kw in keywords:
                    if not kw:
                        continue
                    limit = 30 if len(kw) <= 4 else 15  # krótkie tokeny (LLM, RAG) pasują do wielu węzłów, więc wyższy limit, by nic nie pominąć
                    records = session.run(
                        """
                        MATCH (n:Entity)-[r]->(m:Entity)
                        WHERE toLower(n.id)   CONTAINS toLower($kw)
                           OR toLower(m.id)   CONTAINS toLower($kw)
                           OR toLower(type(r)) CONTAINS toLower($kw)
                        RETURN n.id AS src, type(r) AS rel, m.id AS tgt
                        LIMIT $lim
                        """,
                        kw=kw,
                        lim=limit,
                    )
                    for rec in records:
                        found.add(f"{rec['src']} | {rec['rel']} | {rec['tgt']}")
    except Exception:
        logger.error("Błąd wyszukiwania trójek w Neo4j", exc_info=False)
        query_failed = True

    result = list(found)

    # Wyniku z błędnego zapytania nie zapamiętujemy, inaczej pusta lista zostałaby w cache
    # także po tym, jak Neo4j znów zacznie działać
    if query_failed:
        return result

    # Ogranicz rozmiar cache przez usunięcie najstarszego wpisu (FIFO)
    if len(_retrieval_cache) >= _MAX_CACHE:
        _retrieval_cache.pop(next(iter(_retrieval_cache)))
    _retrieval_cache[cache_key] = result

    return result


def get_graph_rag_response(prompt: str) -> dict[str, str]:
    # Główna funkcja potoku Graph RAG: pytanie użytkownika -> odpowiedź z grafu.
    # Kroki:
    #   1. sprawdź połączenie z Neo4j,
    #   2. sprawdź, czy baza ma jakiekolwiek dane,
    #   3. wyciągnij słowa kluczowe, wyszukaj trójki, przefiltruj je i dołóż sąsiadów,
    #   4. wygeneruj odpowiedź modelem wyłącznie na podstawie znalezionych trójek,
    #   5. oczyść odpowiedź (usuń duplikaty i sprzeczne "brak informacji").
    # Zawsze zwraca słownik z kluczami:
    #   "answer"  - tekst odpowiedzi lub komunikat o błędzie,
    #   "context" - użyte trójki (puste, gdy nic nie znaleziono).
    # Krok 1: weryfikacja połączenia
    driver = _make_driver()
    if not driver:
        return {
            "answer": (
                "**[Błąd krytyczny]** Nie udało się połączyć z bazą Neo4j.\n"
                "Sprawdź ustawienia URI i hasła w panelu bocznym."
            ),
            "context": "",
        }
    driver.close()

    # Krok 2: sprawdź czy baza ma jakiekolwiek dane
    if is_graph_empty():
        return {
            "answer": (
                "**Brak danych w grafie wiedzy.**\n\n"
                "Baza Neo4j jest pusta. Przejdź do zakładki **Baza Wiedzy** "
                "lub użyj przycisku **Załaduj dane testowe** w zakładce Ewaluacja."
            ),
            "context": "",
        }

    # Krok 3: ekstrakcja słów kluczowych, retrieval i post-filtrowanie
    keywords, llm_phrases = _extract_keywords(prompt)
    all_triplets = _retrieve_triplets(keywords)
    triplets = _rerank_triplets(all_triplets, llm_phrases, prompt)
    # Graph traversal: pobierz krawędzie rzadkich podmiotów poza zasięgiem CONTAINS
    # (np. UDZIELILA_GRANTU_MLN_PLN dla inwestora gdy pytanie nie zawiera słowa "grant")
    extra = _fetch_entity_neighbors(triplets, all_triplets)
    if extra:
        triplets = list(dict.fromkeys(triplets + extra))

    if not triplets:
        return {
            "answer": (
                "**Brak powiązań w grafie dla tego zapytania.**\n\n"
                f"Szukano węzłów pasujących do: *{', '.join(keywords)}*.\n"
                "Spróbuj wczytać do bazy wiedzę na ten temat."
            ),
            "context": "",
        }

    context = "\n".join(triplets)

    # Krok 4: generowanie odpowiedzi
    # Prompt po angielsku: małe modele (Hermes-3 3B) lepiej stosują się do ponumerowanych reguł
    # w tym języku niż po polsku
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
        raw_answer = llm_engine.llm.complete(answer_prompt).text.strip()
        answer = _deduplicate_sentences(raw_answer)
        answer = _strip_brak_contradiction(answer)
        return {
            "answer": f"**Odpowiedź z Grafu Neo4j**\n\n{answer}",
            "context": context,
        }
    except Exception:
        logger.error("Błąd generowania odpowiedzi Graph RAG", exc_info=False)
        return {
            "answer": (
                "**[Błąd generowania odpowiedzi]** Sprawdź adres API i czy "
                "serwer LLM jest uruchomiony."
            ),
            "context": context,
        }
