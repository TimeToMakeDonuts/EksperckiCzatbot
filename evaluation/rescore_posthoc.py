import argparse
import json
import os
from pathlib import Path

import pandas as pd

from core import llm_engine
from core.graph_rag import postprocess_answer
from evaluation.evaluator import judge_faithfulness, score_answer_correctness

# Korekta zapisanych wyników ewaluacji po ujednoliceniu przetwarzania końcowego.
# W rundach R1-R9 Vector RAG nie usuwał zdań "Brak informacji...", które Graph RAG usuwał.
# Przetwarzanie końcowe jest deterministyczne, więc nałożenie go na zapisane odpowiedzi
# Vector RAG daje dokładnie to, co wygenerowałby poprawiony kod dla tej samej odpowiedzi
# modelu. Ponownie oceniane są tylko odpowiedzi, które się zmieniły.
#
# Użycie (z katalogu głównego projektu, przy uruchomionym LM Studio z modelem sędziego):
#   python -m evaluation.rescore_posthoc evaluation/results/eval_20260810_015918.csv
#   python -m evaluation.rescore_posthoc evaluation/results/eval_20260909_125413.csv --rejudge-rag 1

_SETTINGS_FILE = Path("settings.json")


def _init_judge_from_settings() -> None:
    # Konfiguruje sędziego tak samo jak aplikacja: adres, model i temperatura z settings.json,
    # klucz API ze zmiennej środowiskowej (klucze nie są zapisywane w settings.json).
    settings = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8")) if _SETTINGS_FILE.exists() else {}
    if settings.get("use_separate_judge"):
        llm_engine.init_or_update_judge_llm(
            settings.get("judge_api_base", "http://localhost:1234/v1"),
            os.getenv("JUDGE_API_KEY", "lm-studio"),
            settings.get("judge_model_name", "local-model"),
            settings.get("judge_temp", 0.1),
        )
    else:
        llm_engine.init_or_update_llm(
            settings.get("api_base", "http://localhost:1234/v1"),
            os.getenv("LLM_API_KEY", "lm-studio"),
            settings.get("model_name", "local-model"),
            settings.get("temp", 0.1),
            settings.get("max_tokens", 512),
        )


def rescore_vector(csv_path: Path, rejudge_rag_ids: list[int] | None = None) -> Path:
    # Nakłada wspólne przetwarzanie końcowe na odpowiedzi Vector RAG, ponownie ocenia
    # zmienione odpowiedzi i zapisuje wynik obok oryginału z przyrostkiem "_posthoc".
    # rejudge_rag_ids: pytania, których (niezmienione) odpowiedzi Graph RAG sędzia ma ocenić
    # jeszcze raz, np. podejrzanie niski wynik Q1 w R9. Nowa ocena trafia do osobnej kolumny,
    # a oryginalny wynik zostaje bez zmian.
    df = pd.read_csv(csv_path)
    if rejudge_rag_ids:
        df["faithfulness_rag_ponowna"] = None
        df["sedzia_faith_rag_raw"] = ""
        for i, row in df[df["id"].isin(rejudge_rag_ids)].iterrows():
            judge = judge_faithfulness(str(row["odpowiedz_rag"]), str(row["kontekst_rag"]))
            df.loc[i, "faithfulness_rag_ponowna"] = judge["score"]
            df.loc[i, "sedzia_faith_rag_raw"] = judge["raw"]
            print(f"Graph RAG Q{row['id']}: faithfulness {row['faithfulness_rag']} -> ponownie {judge['score']} "
                  f"(warstwa: {judge['layer']})")
    # Pliki z rund R1-R9 nie mają kolumn z surową odpowiedzią sędziego
    for col in ("sedzia_faith_vec_warstwa", "sedzia_faith_vec_raw"):
        if col not in df.columns:
            df[col] = ""
    df = df.astype({"faithfulness_vec": "object", "poprawnosc_vec": "object"})
    changed = []
    for i, row in df.iterrows():
        header, sep, body = str(row["odpowiedz_vec"]).partition("\n\n")
        if not sep:
            continue
        new_body = postprocess_answer(body)
        if new_body == body:
            continue
        new_answer = f"{header}\n\n{new_body}"
        judge = judge_faithfulness(new_answer, str(row["kontekst_vec"]))
        df.loc[i, "odpowiedz_vec"] = new_answer
        df.loc[i, "faithfulness_vec"] = judge["score"]
        df.loc[i, "poprawnosc_vec"] = score_answer_correctness(new_answer, str(row["ground_truth"]))
        df.loc[i, "sedzia_faith_vec_warstwa"] = judge["layer"]
        df.loc[i, "sedzia_faith_vec_raw"] = judge["raw"]
        changed.append((row["id"], row["faithfulness_vec"], judge["score"]))

    out_path = csv_path.with_name(csv_path.stem + "_posthoc.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    original = pd.read_csv(csv_path)
    print(f"Zmienione odpowiedzi Vector RAG (id, faithfulness przed, po): {changed}")
    for col in ("faithfulness_vec", "poprawnosc_vec"):
        print(f"{col}: {original[col].mean():.1f} -> {pd.to_numeric(df[col]).mean():.1f}")
    print(f"Zapisano: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Korekta post-hoc wyników Vector RAG")
    parser.add_argument("csv", nargs="+", type=Path, help="pliki CSV z evaluation/results/")
    parser.add_argument("--rejudge-rag", nargs="*", type=int, default=[], metavar="ID",
                        help="id pytań, których odpowiedzi Graph RAG sędzia ma ocenić ponownie")
    args = parser.parse_args()
    _init_judge_from_settings()
    for path in args.csv:
        rescore_vector(path, args.rejudge_rag)
