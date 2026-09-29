import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from core import llm_engine
from core.llm_engine import get_standard_llm_response
from core.graph_rag import get_graph_rag_response
from core.vector_rag import get_vector_rag_response

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_eval_dataset(path: str | None = None) -> list[dict]:
    # Wczytuje zestaw pytań ewaluacyjnych z pliku JSON (lista obiektów z polami
    # question i ground_truth). Bez argumentu bierze domyślny zorbex_eval_dataset.json.
    if path is None:
        path = Path(__file__).parent / "zorbex_eval_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ask_llm_for_score(prompt: str) -> int | None:
    # Wysyła prompt do modelu-sędziego i z jego odpowiedzi bierze pierwszą liczbę
    # z zakresu 0-100. Zwraca None, gdy sędzia nie podał liczby lub wystąpił błąd.
    try:
        raw = llm_engine.get_judge().complete(prompt).text.strip()
        match = re.search(r'\b(\d{1,3})\b', raw)
        if match:
            return min(max(int(match.group(1)), 0), 100)
        return None
    except Exception:
        return None


def score_faithfulness(answer: str, context: str) -> int | None:
    # Faithfulness (wierność źródłom), 0-100: czy odpowiedź opiera się wyłącznie
    # na faktach z kontekstu i niczego nie zmyśla.
    # Sędzia wypisuje twierdzenia z odpowiedzi, oznacza każde jako YES (poparte
    # kontekstem) lub NO, a wynik to odsetek YES.
    # Odpowiedź sędziego czytamy na trzy sposoby, od najbardziej do najmniej
    # wiarygodnego:
    #   1. linia "SCORE: X",
    #   2. policzenie oznaczeń "-> YES" i "-> NO",
    #   3. ostatnia liczba w tekście.
    # Zwraca None, gdy brakuje odpowiedzi lub kontekstu.
    if not context.strip() or not answer.strip():
        return None

    prompt = (
        "You are a strict AI evaluator measuring answer faithfulness to source facts.\n\n"
        "SOURCE FACTS (subject | relation | object):\n"
        f"{context}\n\n"
        "AI ANSWER:\n"
        f"{answer}\n\n"
        "Task: check each claim in the AI ANSWER against SOURCE FACTS.\n"
        "Use bullet points (no numbering). Mark each: -> YES (supported) or -> NO (not supported).\n"
        "Then write SCORE = round(YES / total * 100).\n\n"
        "Output format:\n"
        "- [claim] -> YES\n"
        "- [claim] -> NO\n"
        "SCORE: [integer 0-100]\n\n"
        "Evaluate:"
    )
    try:
        raw = llm_engine.get_judge().complete(prompt).text.strip()

        # Warstwa 1: szukaj linii "SCORE: X"
        score_match = re.search(r'(?i)score[:\s=]+(\d{1,3})', raw)
        if score_match:
            return min(max(int(score_match.group(1)), 0), 100)

        # Warstwa 2: policz markery -> YES / -> NO i oblicz proporcję
        yes_count = len(re.findall(r'->\s*YES', raw, re.IGNORECASE))
        no_count = len(re.findall(r'->\s*NO', raw, re.IGNORECASE))
        total = yes_count + no_count
        if total > 0:
            return round(yes_count / total * 100)

        # Warstwa 3: weź ostatnią liczbę z tekstu
        # Bierzemy OSTATNIĄ liczbę: pierwsza mogłaby być numerem z listy ("1."),
        # a wynik sędzia zawsze podaje na końcu swojego rozumowania
        all_nums = re.findall(r'\b(\d{1,3})\b', raw)
        if all_nums:
            return min(max(int(all_nums[-1]), 0), 100)

        return None
    except Exception:
        return None


def score_answer_correctness(generated: str, ground_truth: str) -> int | None:
    # Poprawność odpowiedzi względem wzorca (ground truth), 0-100.
    # Sędzia bierze pod uwagę zgodność faktów (50%), kompletność (30%) i brak
    # zmyśleń (20%). Liczymy ją dla wszystkich trzech metod (LLM, Vector RAG,
    # Graph RAG), więc można je bezpośrednio porównać.
    if not generated.strip() or not ground_truth.strip():
        return None

    prompt = f"""Jesteś obiektywnym sędzią oceniającym jakość odpowiedzi.

WZORZEC (ground truth — poprawna odpowiedź):
---
{ground_truth}
---

OCENIANA ODPOWIEDŹ:
---
{generated}
---

Oceń POPRAWNOŚĆ ocenianej odpowiedzi (0-100) według następujących kryteriów:
- Zgodność faktyczna ze wzorcem (waga 50%): czy kluczowe fakty są poprawne?
- Kompletność (waga 30%): czy odpowiedź obejmuje główne punkty wzorca?
- Brak halucynacji (waga 20%): czy odpowiedź nie zawiera fałszywych twierdzeń?

Zwróć TYLKO jedną liczbę całkowitą (np. 65). Żadnego innego tekstu."""
    return _ask_llm_for_score(prompt)


def score_citation_precision(answer: str, context: str) -> float | None:
    # Precyzja cytowania, 0.0-1.0: jaka część trójek z kontekstu jest widoczna
    # w odpowiedzi. Trójka liczy się jako "zacytowana", gdy w tekście odpowiedzi
    # występują co najmniej 2 jej części (podmiot, relacja lub obiekt).
    # Niski wynik oznacza, że do modelu trafiło dużo faktów zbędnych dla pytania.
    if not context.strip() or not answer.strip():
        return None

    lines = [ln.strip() for ln in context.strip().splitlines() if ln.strip()]
    if not lines:
        return None

    answer_lower = answer.lower()
    cited_count = 0

    for line in lines:
        # Rozbij trójkę na części niezależnie od separatora (|, >, -, nawiasy)
        parts = [p.strip().lower() for p in re.split(r'[\->,|()]+', line) if p.strip()]
        if not parts:
            continue
        hits = sum(1 for p in parts if len(p) > 2 and p in answer_lower)
        if hits >= min(2, len(parts)):  # min(2, ...) handles single-element triplets gracefully
            cited_count += 1

    return round(cited_count / len(lines), 3)


def classify_error(rag_correctness: int | None, llm_correctness: int | None,
                   faithfulness: int | None, rag_context: str) -> str:
    # Przypisuje pytaniu etykietę jakości odpowiedzi RAG (sprawdzane po kolei):
    #   - pusty kontekst              -> "Brak kontekstu w grafie",
    #   - faithfulness poniżej 40     -> "Halucynacja RAG",
    #   - brak oceny poprawności      -> "Błąd obliczenia metryki",
    #   - poprawność 75 i więcej      -> "OK",
    #   - poprawność 50-74            -> "Odpowiedź częściowa",
    #   - poprawność poniżej 50 i o ponad 15 pkt gorsza niż zwykły LLM
    #                                 -> "RAG gorszy od LLM",
    #   - pozostałe przypadki         -> "Zła odpowiedź".
    if not rag_context.strip():
        return "Brak kontekstu w grafie"
    if faithfulness is not None and faithfulness < 40:
        return "Halucynacja RAG"
    if rag_correctness is None:
        return "Błąd obliczenia metryki"
    if rag_correctness >= 75:
        return "OK"
    if rag_correctness >= 50:
        return "Odpowiedź częściowa"
    if llm_correctness is not None and llm_correctness > rag_correctness + 15:
        return "RAG gorszy od LLM"
    return "Zła odpowiedź"


def run_evaluation(dataset: list[dict], progress_callback=None) -> pd.DataFrame:
    # Uruchamia pełną ewaluację na liście pytań. Dla każdego pytania:
    #   1. pobiera odpowiedzi trzech metod (LLM, Graph RAG, Vector RAG) i mierzy czas,
    #   2. liczy metryki (faithfulness, poprawność, precyzja cytowania),
    #   3. przypisuje etykietę błędu.
    # Wyniki trafiają do tabeli i do pliku CSV w evaluation/results/ (nazwa z datą
    # i godziną, więc każde uruchomienie tworzy nowy plik).
    # progress_callback(i, total, pytanie) służy do aktualizacji paska postępu.
    results = []
    total = len(dataset)

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]
        category = item.get("category", "ogólne")

        if progress_callback:
            progress_callback(i, total, question)

        # Generuj odpowiedzi z pomiarem czasu
        _t = time.time()
        llm_answer = get_standard_llm_response(question)
        czas_llm = round(time.time() - _t, 2)

        _t = time.time()
        rag_output = get_graph_rag_response(question)
        czas_rag = round(time.time() - _t, 2)
        rag_answer = rag_output["answer"]
        rag_context = rag_output["context"]

        _t = time.time()
        vec_output = get_vector_rag_response(question)
        czas_vec = round(time.time() - _t, 2)
        vec_answer = vec_output["answer"]
        vec_context = vec_output["context"]

        # Oblicz metryki dla wszystkich trzech metod
        faith_rag = score_faithfulness(rag_answer, rag_context)
        llm_corr = score_answer_correctness(llm_answer, ground_truth)
        rag_corr = score_answer_correctness(rag_answer, ground_truth)
        cit_prec_rag = score_citation_precision(rag_answer, rag_context)
        error_label = classify_error(rag_corr, llm_corr, faith_rag, rag_context)

        faith_vec = score_faithfulness(vec_answer, vec_context)
        vec_corr = score_answer_correctness(vec_answer, ground_truth)
        cit_prec_vec = score_citation_precision(vec_answer, vec_context)

        results.append({
            "id": item.get("id", i + 1),
            "kategoria": category,
            "pytanie": question,
            "ground_truth": ground_truth,
            "odpowiedz_llm": llm_answer,
            "poprawnosc_llm": llm_corr,
            "czas_llm_s": czas_llm,
            "odpowiedz_rag": rag_answer,
            "kontekst_rag": rag_context,
            "faithfulness_rag": faith_rag,
            "poprawnosc_rag": rag_corr,
            "citation_precision_rag": cit_prec_rag,
            "czas_rag_s": czas_rag,
            "odpowiedz_vec": vec_answer,
            "kontekst_vec": vec_context,
            "faithfulness_vec": faith_vec,
            "poprawnosc_vec": vec_corr,
            "citation_precision_vec": cit_prec_vec,
            "czas_vec_s": czas_vec,
            "klasyfikacja_bledu": error_label,
        })

    if progress_callback:
        progress_callback(total, total, "Zakończono")

    df = pd.DataFrame(results)
    # Zapisz wyniki z timestampem — każde uruchomienie tworzy nowy plik
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{timestamp}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def compute_summary(df: pd.DataFrame) -> dict:
    # Liczy średnią, minimum i maksimum każdej metryki z tabeli wyników.
    # Puste wartości (None) są pomijane. Precyzję cytowania przeliczamy na procenty.
    metrics = {
        "poprawnosc_llm":         {"label": "Poprawność LLM",          "scale": 1},
        "poprawnosc_vec":         {"label": "Poprawność Vector RAG",    "scale": 1},
        "poprawnosc_rag":         {"label": "Poprawność Graph RAG",     "scale": 1},
        "faithfulness_vec":       {"label": "Faithfulness Vector RAG",  "scale": 1},
        "faithfulness_rag":       {"label": "Faithfulness Graph RAG",   "scale": 1},
        "citation_precision_rag": {"label": "Citation Precision RAG",   "scale": 100},
        "citation_precision_vec": {"label": "Citation Precision Vec",   "scale": 100},
        "czas_llm_s":             {"label": "Śr. czas LLM (s)",         "scale": 1},
        "czas_vec_s":             {"label": "Śr. czas Vector RAG (s)",  "scale": 1},
        "czas_rag_s":             {"label": "Śr. czas Graph RAG (s)",   "scale": 1},
    }

    summary = {}
    for col, meta in metrics.items():
        valid = df[col].dropna()
        scale = meta["scale"]
        summary[col] = {
            "label": meta["label"],
            "mean": round(valid.mean() * scale, 1) if len(valid) > 0 else None,
            "min":  round(valid.min()  * scale, 1) if len(valid) > 0 else None,
            "max":  round(valid.max()  * scale, 1) if len(valid) > 0 else None,
            "n_valid": len(valid),
            "n_total": len(df),
        }
    return summary
