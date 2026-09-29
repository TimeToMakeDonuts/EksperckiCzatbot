import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import os
import re
import tempfile
import json
import time
from pathlib import Path
from pyvis.network import Network
from dotenv import load_dotenv

# Wczytaj zmienne środowiskowe z pliku .env (Neo4j URI, hasła itp.)
load_dotenv()

# Konfiguracja logowania — przed importami core, żeby wyciszyć biblioteki zewnętrzne
from core.logger import setup_logging
setup_logging()

# Import całego modułu (a nie `from ... import llm`), żeby zawsze używać najnowszego modelu
# utworzonego przez init_or_update_llm() po zmianie ustawień
from core import llm_engine
from core.llm_engine import (
    get_standard_llm_response,
    init_or_update_llm,
    init_or_update_judge_llm,
    reset_judge_llm,
)
from core.graph_rag import get_graph_rag_response, clear_retrieval_cache
from evaluation.evaluator import score_faithfulness
from core.graph_builder import extract_text_from_pdf, extract_triplets_with_llm, save_edited_triplets_to_neo4j

st.set_page_config(layout="wide", page_title="GraphRAG vs LLM Chatbot")
st.title("System Ekspercki: RAG Grafowy")

# --- TRWAŁE USTAWIENIA ---

_SETTINGS_FILE = Path("settings.json")

# Klucze ustawień zapisywanych lokalnie między sesjami.
# Hasło Neo4j i klucze API celowo NIE należą do tej listy — nie są zapisywane na dysku
# (wartości domyślne pochodzą ze zmiennych środowiskowych / pliku .env).
_SETTINGS_KEYS = [
    "api_base", "model_name", "temp", "max_tokens",
    "neo_uri", "neo_user",
    "use_separate_judge", "judge_api_base", "judge_model_name", "judge_temp",
]


def _load_settings() -> None:
    # Wczytuje ustawienia (adres API, model, URI Neo4j itd.) z pliku settings.json
    # do st.session_state, żeby pola w panelu bocznym miały wartości z poprzedniej sesji.
    # Działa tylko raz na sesję. Hasła i klucze API nigdy nie są w tym pliku.
    if st.session_state.get("_settings_loaded"):
        return
    if _SETTINGS_FILE.exists():
        try:
            saved = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
            for k, v in saved.items():
                if k in _SETTINGS_KEYS and k not in st.session_state:
                    st.session_state[k] = v
            if any(k not in _SETTINGS_KEYS for k in saved):
                # usuń z pliku dane wrażliwe zapisane przez starszą wersję aplikacji
                _SETTINGS_FILE.write_text(
                    json.dumps({k: v for k, v in saved.items() if k in _SETTINGS_KEYS},
                               ensure_ascii=False, indent=2),
                    encoding="utf-8")
        except Exception:
            pass  # uszkodzony plik — zignoruj, użytkownik ustawi ręcznie
    st.session_state["_settings_loaded"] = True


def _save_settings() -> None:
    # Zapisuje bieżące ustawienia z panelu bocznego do pliku settings.json,
    # aby przy następnym uruchomieniu aplikacji zostały wczytane automatycznie.
    # Zapisywane są tylko klucze z listy _SETTINGS_KEYS (bez haseł).
    data = {k: st.session_state[k] for k in _SETTINGS_KEYS if k in st.session_state}
    _SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


_load_settings()

# --- PANEL BOCZNY: USTAWIENIA ---
with st.sidebar:
    st.header("Ustawienia Systemu")

    st.subheader("Model Językowy (LLM)")
    # Używamy st.session_state żeby wartości nie resetowały się przy każdym przeładowaniu
    api_base = st.text_input("Adres API", value=st.session_state.get("api_base", "http://localhost:1234/v1"))
    api_key = st.text_input("Klucz API", type="password",
                            value=st.session_state.get("api_key", os.getenv("LLM_API_KEY", "lm-studio")))
    model_name = st.text_input("Nazwa Modelu", value=st.session_state.get("model_name", "google/gemma-3-27b"))
    temperature = st.slider("Kreatywność (Temperature)", 0.0, 1.0, st.session_state.get("temp", 0.1), 0.1)
    max_tokens = st.number_input("Limit tokenów", 100, 4000, st.session_state.get("max_tokens", 512), step=100)

    st.subheader("Baza Grafowa")
    neo4j_uri = st.text_input("Adres URI",
                              value=st.session_state.get("neo_uri", os.getenv("NEO4J_URI", "bolt://localhost:7687")))
    neo4j_user = st.text_input("Użytkownik",
                               value=st.session_state.get("neo_user", os.getenv("NEO4J_USERNAME", "neo4j")))
    neo4j_pass = st.text_input("Hasło do bazy", type="password",
                               value=st.session_state.get("neo_pass", os.getenv("NEO4J_PASSWORD", "")))

    st.subheader("Sędzia (LLM-as-Judge)")
    use_separate_judge = st.checkbox(
        "Użyj oddzielnego modelu jako sędziego",
        value=st.session_state.get("use_separate_judge", False),
        help="Gdy włączone, metryki Faithfulness i Poprawność są oceniane przez inny model/maszynę niż ten, który generuje odpowiedzi.",
    )
    if use_separate_judge:
        judge_api_base = st.text_input(
            "Adres API sędziego",
            value=st.session_state.get("judge_api_base", "http://192.168.1.100:1234/v1"),
            help="Adres IP/hostname drugiej maszyny z LM Studio lub innym serwerem OpenAI-compatible.",
        )
        judge_api_key = st.text_input(
            "Klucz API sędziego",
            type="password",
            value=st.session_state.get("judge_api_key", os.getenv("JUDGE_API_KEY", "lm-studio")),
        )
        judge_model_name = st.text_input(
            "Nazwa modelu sędziego",
            value=st.session_state.get("judge_model_name", "local-model"),
        )
        judge_temp = st.slider(
            "Temperature sędziego",
            0.0, 1.0,
            st.session_state.get("judge_temp", 0.0),
            0.05,
            help="Dla oceniania zalecane 0.0 (deterministyczne).",
        )
        st.caption(f"Aktywny sędzia: **{st.session_state.get('judge_model_name', '—')}** @ `{st.session_state.get('judge_api_base', '—')}`")
    else:
        st.caption("Sędzia: ten sam model co główny LLM")

    apply_clicked = st.button("Zapisz i zastosuj ustawienia", type="primary")
    # Przy pierwszym renderowaniu sesji ustawienia z settings.json są zastosowane automatycznie,
    # inaczej pola panelu pokazywałyby wartości, których silnik LLM (w tym osobny sędzia) nie używa
    auto_apply = not st.session_state.get("_settings_applied")
    if apply_clicked or auto_apply:
        st.session_state["_settings_applied"] = True
        # Zapisz do session_state żeby wartości przetrwały przeładowanie
        st.session_state.api_base = api_base
        st.session_state.api_key = api_key
        st.session_state.model_name = model_name
        st.session_state.temp = temperature
        st.session_state.max_tokens = max_tokens
        st.session_state.neo_uri = neo4j_uri
        st.session_state.neo_user = neo4j_user
        st.session_state.neo_pass = neo4j_pass
        st.session_state.use_separate_judge = use_separate_judge

        # Nadpisz zmienne środowiskowe procesu — moduly graph_rag i graph_builder czytają je przez os.getenv()
        os.environ["NEO4J_URI"] = neo4j_uri
        os.environ["NEO4J_USERNAME"] = neo4j_user
        os.environ["NEO4J_PASSWORD"] = neo4j_pass

        # Reinicjalizuj silnik LLM z nowymi parametrami
        init_or_update_llm(api_base, api_key, model_name, temperature, max_tokens)

        # Skonfiguruj oddzielnego sędziego lub wróć do głównego LLM
        if use_separate_judge:
            st.session_state.judge_api_base = judge_api_base
            st.session_state.judge_api_key = judge_api_key
            st.session_state.judge_model_name = judge_model_name
            st.session_state.judge_temp = judge_temp
            init_or_update_judge_llm(judge_api_base, judge_api_key, judge_model_name, judge_temp)
            if apply_clicked:
                st.success(f"Zaaktualizowano konfigurację. Sędzia: {judge_model_name} @ {judge_api_base}")
        else:
            reset_judge_llm()
            if apply_clicked:
                st.success("Zaaktualizowano konfigurację. Sędzia: główny LLM")

        # Zapisz ustawienia do pliku (tylko po kliknięciu) — wczytają się przy następnym uruchomieniu
        if apply_clicked:
            _save_settings()


# --- FUNKCJE POMOCNICZE ---

def get_visual_graph_html(context: str) -> str | None:
    # Buduje interaktywny graf (biblioteka PyVis) z trójek, które posłużyły do
    # odpowiedzi na ostatnie pytanie. Rysuje tylko te trójki, a nie cały graf z Neo4j.
    # Wejście: tekst, w którym każda linia to trójka "podmiot | relacja | obiekt".
    # Wynik: kod HTML grafu do wyświetlenia w Streamlit albo None, gdy nie ma czego rysować.
    if not context or not context.strip():
        return None
    lines = [ln.strip() for ln in context.strip().splitlines() if ln.strip()]
    if not lines:
        return None

    net = Network(height="350px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
    rendered = 0
    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            src, rel, tgt = parts
            net.add_node(src, label=src, title=src, color="#ff4b4b")
            net.add_node(tgt, label=tgt, title=tgt, color="#4b4bff")
            net.add_edge(src, tgt, title=rel, label=rel, color="#aaaaaa")
            rendered += 1

    if rendered == 0:
        return None

    net.toggle_physics(True)
    # Zapisz do pliku tymczasowego i odczytaj HTML — PyVis nie obsługuje zwracania stringa bezpośrednio
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".html")
    os.close(tmp_fd)
    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp_path)
    return html


# --- INICJALIZACJA STANU SESJI ---
# Przechowuje historię czatu i wyniki między przeładowaniami Streamlit
if "messages_llm" not in st.session_state:
    st.session_state.messages_llm = []
if "messages_rag" not in st.session_state:
    st.session_state.messages_rag = []
if "latest_graph_html" not in st.session_state:
    st.session_state.latest_graph_html = None
if "latest_metric" not in st.session_state:
    st.session_state.latest_metric = None
if "latest_times" not in st.session_state:
    st.session_state.latest_times = {}  # {"llm": float, "rag": float} w sekundach

tab1, tab2, tab3 = st.tabs(["Czat z Modelem", "Baza Wiedzy (Dodaj Dane)", "Ewaluacja Systemu"])


def render_chat_history(messages, column_context):
    # Wyświetla w podanej kolumnie Streamlit całą dotychczasową rozmowę
    # (listę wiadomości z polami "role" i "content").
    for msg in messages:
        with column_context:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


# --- ZAKŁADKA 1: CZAT Z MODELEM ---
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Standardowy LLM")
        st.caption("Odpowiedź generowana wyłącznie na podstawie wbudowanej wiedzy modelu.")
        render_chat_history(st.session_state.messages_llm, col1)

    with col2:
        st.subheader("LLM + Graph RAG")
        st.caption("Odpowiedź wzbogacona o kontekst pobrany z grafowej bazy danych.")
        render_chat_history(st.session_state.messages_rag, col2)

        # Wyświetl metrykę faithfulness i czasy generowania ostatniej odpowiedzi
        if st.session_state.latest_metric is not None or st.session_state.latest_times:
            st.divider()
            st.caption("Analiza ostatniej odpowiedzi RAG")
            if st.session_state.latest_metric is not None:
                score = st.session_state.latest_metric
                color = "green" if score > 80 else "orange" if score > 50 else "red"
                st.markdown(f"**Faithfulness (Wierność źródłom):** :{color}[**{score}%**]")
                st.progress(score / 100.0)
            times = st.session_state.latest_times
            if times:
                t_llm = times.get("llm")
                t_rag = times.get("rag")
                parts = []
                if t_llm is not None:
                    parts.append(f"LLM: **{t_llm:.1f}s**")
                if t_rag is not None:
                    parts.append(f"Graph RAG: **{t_rag:.1f}s**")
                if parts:
                    st.caption("⏱ Czas generowania — " + " | ".join(parts))

        # Wyświetl interaktywny graf trójek użytych w ostatniej odpowiedzi
        if st.session_state.latest_graph_html:
            with st.expander("Zobacz strukturę grafu", expanded=False):
                components.html(st.session_state.latest_graph_html, height=360)

    prompt = st.chat_input("Wpisz swoje zapytanie.")

    if prompt:
        # Dodaj pytanie użytkownika do historii obu kolumn
        st.session_state.messages_llm.append({"role": "user", "content": prompt})
        st.session_state.messages_rag.append({"role": "user", "content": prompt})

        with col1:
            with st.chat_message("user"):
                st.markdown(prompt)
        with col2:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Generuj odpowiedź standardowego LLM z pomiarem czasu
        with col1:
            with st.chat_message("assistant"):
                with st.spinner("Generowanie odpowiedzi w LM Studio..."):
                    _t0 = time.time()
                    response_llm = get_standard_llm_response(prompt)
                    st.session_state.latest_times["llm"] = round(time.time() - _t0, 2)
                    st.markdown(response_llm)
                    st.caption(f"⏱ {st.session_state.latest_times['llm']:.1f}s")
                    st.session_state.messages_llm.append({"role": "assistant", "content": response_llm})

        # Generuj odpowiedź Graph RAG, oblicz faithfulness i wyrenderuj graf
        with col2:
            with st.chat_message("assistant"):
                with st.spinner("Przeszukiwanie grafu i generowanie odpowiedzi..."):
                    _t0 = time.time()
                    rag_output = get_graph_rag_response(prompt)
                    st.session_state.latest_times["rag"] = round(time.time() - _t0, 2)

                    odpowiedz = rag_output["answer"]
                    kontekst = rag_output["context"]

                    st.markdown(odpowiedz)
                    st.caption(f"⏱ {st.session_state.latest_times['rag']:.1f}s")
                    st.session_state.messages_rag.append({"role": "assistant", "content": odpowiedz})

                with st.spinner("Wynik faithfulness..."):
                    if kontekst:
                        score = score_faithfulness(odpowiedz, kontekst)
                        st.session_state.latest_metric = score
                    else:
                        st.session_state.latest_metric = None

                with st.spinner("Renderowanie interaktywnego grafu..."):
                    html_graph = get_visual_graph_html(kontekst)
                    if html_graph:
                        st.session_state.latest_graph_html = html_graph

        # Wymuś odświeżenie strony żeby scroll zjechał na dół i pokazały się nowe wiadomości
        st.rerun()


# --- ZAKŁADKA 2: BAZA WIEDZY ---
with tab2:
    st.header("Zarządzanie Grafem Wiedzy")
    st.markdown("Wprowadź tekst lub plik PDF")

    input_method = st.radio("Sposób wprowadzania danych:", ["Wklej Tekst", "Wgraj PDF"])
    text_to_process = ""

    if input_method == "Wklej Tekst":
        text_to_process = st.text_area("Wklej artykuł/wiedzę tutaj:", height=200)
    else:
        uploaded_file = st.file_uploader("Wybierz plik PDF", type="pdf")
        if uploaded_file is not None:
            text_to_process = extract_text_from_pdf(uploaded_file)
            st.success("Plik PDF wczytany poprawnie")
            with st.expander("Podgląd tekstu z PDF"):
                st.write(text_to_process[:1000] + "...")

    # Wyślij tekst do LLM który wyodrębni relacje w formacie trójek
    if st.button("Znajdź Encje i Relacje", type="primary"):
        if text_to_process.strip() == "":
            st.warning("Brak wprowadzonego tekstu")
        else:
            with st.spinner("LLM analizuje tekst..."):
                triplets = extract_triplets_with_llm(text_to_process)
                if triplets:
                    # Zapisz do session_state żeby tabela nie znikała przy klikaniu innych elementów
                    st.session_state.temp_df = pd.DataFrame(triplets)
                    st.success(f"Znaleziono {len(triplets)} potencjalnych relacji")
                else:
                    st.error("Model nie znalazł żadnych relacji w podanym tekście.")

    # Pokaż edytowalną tabelę trójek — użytkownik może poprawić błędy LLM przed zapisem
    if "temp_df" in st.session_state:
        st.subheader("Panel Edycji Relacji")
        st.info("Możesz edytować komórki, dodawać nowe wiersze na dole tabeli lub usunąć.")

        edited_df = st.data_editor(
            st.session_state.temp_df,
            num_rows="dynamic",
            width='stretch',
        )

        if st.button("Zapisz zatwierdzone dane"):
            with st.spinner("Aktualizowanie grafowej bazy danych..."):
                success = save_edited_triplets_to_neo4j(edited_df)
                if success:
                    clear_retrieval_cache()  # nowe dane w grafie, więc stare wyniki wyszukiwania są nieaktualne
                    st.success("Wiedza została poprawnie wprowadzona.")
                    del st.session_state.temp_df
                    st.rerun()
                else:
                    st.error("Wystąpił błąd podczas zapisu do bazy.")


# --- ZAKŁADKA 3: EWALUACJA SYSTEMU ---
with tab3:
    from evaluation.evaluator import (
        load_eval_dataset, run_evaluation, compute_summary
    )
    from evaluation.zorbex_seed_data import load_zorbex_triplets_to_neo4j as load_seed_triplets_to_neo4j, ZORBEX_TRIPLETS as SEED_TRIPLETS
    from core.vector_rag import build_vector_index, is_index_built

    st.header("Ewaluacja Systemu RAG")
    st.markdown(
        "Automatyczne porównanie jakości odpowiedzi: "
        "**Standardowy LLM** vs **Vector RAG** vs **Graph RAG** "
        "na podstawie zestawu pytań z ręcznie napisanymi wzorcami."
    )

    # Sekcja 0: załaduj dane testowe do Neo4j i zbuduj indeks wektorowy
    st.subheader("0. Dane testowe i indeks wektorowy")

    col_seed, col_vec, col_seed_info = st.columns([1, 1, 2])

    with col_seed:
        st.markdown(
            f"Ładuje **{len(SEED_TRIPLETS)} trójek** do Neo4j "
            f"pokrywających wszystkie 10 pytań."
        )
        if st.button("📥 Załaduj dane testowe do grafu", key="load_seed"):
            with st.spinner("Ładowanie trójek do Neo4j..."):
                try:
                    loaded, total_s = load_seed_triplets_to_neo4j()
                    st.success(f"Załadowano {loaded}/{total_s} trójek.")
                    st.session_state.latest_graph_html = None
                    clear_retrieval_cache()  # nowe dane w bazie → stary cache jest nieaktualny
                except Exception as e:
                    st.error(f"Błąd połączenia z Neo4j: {e}")

    with col_vec:
        idx_status = "✅ Gotowy" if is_index_built() else "⚠️ Niezbudowany"
        st.markdown(f"Indeks wektorowy: **{idx_status}**")
        st.markdown("Wymagany do Vector RAG.")
        if st.button("🔢 Zbuduj indeks wektorowy", key="build_vec_idx"):
            with st.spinner("Obliczanie embeddingów trójek z Neo4j..."):
                ok, n = build_vector_index()
                if ok:
                    st.success(f"Zindeksowano {n} trójek.")
                else:
                    st.error("Nie udało się zbudować indeksu. Sprawdź połączenie z Neo4j.")

    with col_seed_info:
        with st.expander("Podgląd trójek testowych"):
            seed_df = pd.DataFrame(SEED_TRIPLETS, columns=["Podmiot", "Relacja", "Obiekt"])
            st.dataframe(seed_df, width='stretch', hide_index=True, height=220)

    st.divider()

    # Sekcja 1: wczytaj zestaw pytań (domyślny lub własny)
    st.subheader("1. Zestaw pytań ewaluacyjnych")
    col_up, col_hint = st.columns([2, 1])
    with col_up:
        uploaded_eval = st.file_uploader(
            "Wgraj własny plik JSON (opcjonalnie)",
            type="json",
            key="eval_upload",
            help="Plik musi zawierać listę obiektów z polami: question, ground_truth, category (opcjonalne)."
        )
    with col_hint:
        st.info(
            "**Format JSON:**\n"
            "```json\n"
            "[\n"
            "  {\n"
            '    "id": 1,\n'
            '    "question": "...",\n'
            '    "ground_truth": "...",\n'
            '    "category": "..."\n'
            "  }\n"
            "]\n"
            "```"
        )

    if uploaded_eval is not None:
        try:
            eval_data = json.loads(uploaded_eval.read().decode("utf-8"))
            st.session_state.eval_dataset = eval_data
            st.success(f"Załadowano {len(eval_data)} pytań z pliku.")
        except Exception as e:
            st.error(f"Błąd parsowania pliku JSON: {e}")

    # Jeśli nie wgrano własnego pliku, użyj domyślnego eval_dataset.json
    if "eval_dataset" not in st.session_state:
        try:
            st.session_state.eval_dataset = load_eval_dataset()
        except Exception as e:
            st.session_state.eval_dataset = []
            st.error(f"Nie udało się wczytać domyślnego zestawu: {e}")

    eval_data = st.session_state.get("eval_dataset", [])

    if not eval_data:
        st.warning("Brak zestawu pytań. Wgraj plik JSON.")
        st.stop()

    preview_cols = [k for k in ["id", "category", "question", "ground_truth"] if k in eval_data[0]]
    with st.expander(f"Podgląd zestawu ({len(eval_data)} pyt.)", expanded=False):
        st.dataframe(pd.DataFrame(eval_data)[preview_cols], width='stretch', hide_index=True)

    # Sekcja 2: uruchom ewaluację
    st.subheader("2. Uruchomienie")
    n_calls = len(eval_data) * 8  # 3 odpowiedzi + 2×faithfulness + 3×poprawność
    st.warning(
        f"Ewaluacja wykona **~{n_calls} wywołań LLM** "
        f"({len(eval_data)} pyt. × 8: odpowiedź LLM/Vec/RAG, faithfulness ×2, poprawność ×3). "
        "Czas zależy od szybkości lokalnego modelu. "
        "**Upewnij się, że indeks wektorowy jest zbudowany (punkt 0).**"
    )

    if st.button("▶ Uruchom ewaluację", type="primary", key="run_eval"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_progress(current, total, question):
            pct = current / total if total > 0 else 0.0
            progress_bar.progress(pct)
            short_q = question[:70] + "..." if len(question) > 70 else question
            status_text.text(f"[{current}/{total}] {short_q}")

        with st.spinner("Ewaluacja w toku — nie zamykaj aplikacji..."):
            results_df = run_evaluation(eval_data, progress_callback=on_progress)
            st.session_state.eval_results = results_df

        progress_bar.progress(1.0)
        status_text.text("✅ Ewaluacja zakończona. Wyniki zapisano do evaluation/results/")
        st.rerun()

    # Sekcja 3: wyniki
    if "eval_results" in st.session_state:
        df = st.session_state.eval_results
        summary = compute_summary(df)

        st.divider()
        st.subheader("3. Wyniki")

        def _fmt_pct(val):
            return f"{val}%" if val is not None else "N/A"

        # Tabela porównawcza średnich wyników dla trzech metod
        st.markdown("**Porównanie metod (średnie)**")
        llm_mean  = summary["poprawnosc_llm"]["mean"]
        vec_mean  = summary["poprawnosc_vec"]["mean"]
        rag_mean  = summary["poprawnosc_rag"]["mean"]
        fvec_mean = summary["faithfulness_vec"]["mean"]
        frag_mean = summary["faithfulness_rag"]["mean"]
        cvec_mean = summary["citation_precision_vec"]["mean"]
        crag_mean = summary["citation_precision_rag"]["mean"]

        def _delta(val, base):
            # Zwraca różnicę w punktach procentowych między wynikiem metody a bazowym
            # standardowym LLM, np. "+12.5pp" (używane w kafelkach z wynikami ewaluacji).
            if val is None or base is None:
                return None
            return f"{round(val - base, 1):+}pp"

        mc1, mc2, mc3 = st.columns(3)
        t_llm_mean = summary.get("czas_llm_s", {}).get("mean")
        t_vec_mean = summary.get("czas_vec_s", {}).get("mean")
        t_rag_mean = summary.get("czas_rag_s", {}).get("mean")

        def _fmt_s(val):
            return f"{val:.1f}s" if val is not None else "N/A"

        with mc1:
            st.markdown("#### LLM")
            st.metric("Poprawność", _fmt_pct(llm_mean))
            st.metric("Faithfulness", "N/A")
            st.metric("Citation Prec.", "N/A")
            st.metric("Śr. czas", _fmt_s(t_llm_mean))
        with mc2:
            st.markdown("#### Vector RAG")
            st.metric("Poprawność", _fmt_pct(vec_mean), delta=_delta(vec_mean, llm_mean))
            st.metric("Faithfulness", _fmt_pct(fvec_mean))
            st.metric("Citation Prec.", _fmt_pct(cvec_mean))
            st.metric("Śr. czas", _fmt_s(t_vec_mean))
        with mc3:
            st.markdown("#### Graph RAG")
            st.metric("Poprawność", _fmt_pct(rag_mean), delta=_delta(rag_mean, llm_mean))
            st.metric("Faithfulness", _fmt_pct(frag_mean))
            st.metric("Citation Prec.", _fmt_pct(crag_mean))
            st.metric("Śr. czas", _fmt_s(t_rag_mean))

        st.divider()

        col_tbl, col_err = st.columns([3, 1])

        with col_tbl:
            st.markdown("**Wyniki szczegółowe**")
            # Pokaż tylko kolumny które faktycznie istnieją w DataFrame
            display_cols = [c for c in [
                "id", "kategoria", "pytanie",
                "poprawnosc_llm", "poprawnosc_vec", "poprawnosc_rag",
                "faithfulness_vec", "faithfulness_rag",
                "citation_precision_vec", "citation_precision_rag",
                "klasyfikacja_bledu",
            ] if c in df.columns]
            st.dataframe(
                df[display_cols].rename(columns={
                    "id": "ID",
                    "kategoria": "Kat.",
                    "pytanie": "Pytanie",
                    "poprawnosc_llm": "Popr. LLM",
                    "poprawnosc_vec": "Popr. Vec",
                    "poprawnosc_rag": "Popr. RAG",
                    "faithfulness_vec": "Faith. Vec",
                    "faithfulness_rag": "Faith. RAG",
                    "citation_precision_vec": "Cit. Vec",
                    "citation_precision_rag": "Cit. RAG",
                    "klasyfikacja_bledu": "Klasyfikacja",
                }),
                width='stretch',
                hide_index=True,
            )

        with col_err:
            st.markdown("**Analiza błędów**")
            error_counts = (
                df["klasyfikacja_bledu"]
                .value_counts()
                .reset_index()
                .rename(columns={"klasyfikacja_bledu": "Typ błędu", "count": "Liczba"})
            )
            st.dataframe(error_counts, width='stretch', hide_index=True)

        # Podgląd pełnych odpowiedzi dla wybranego pytania
        with st.expander("Podgląd pełnych odpowiedzi (dla wybranego pytania)"):
            ids = df["id"].tolist()
            selected_id = st.selectbox("Wybierz ID pytania:", ids)
            row = df[df["id"] == selected_id].iloc[0]

            st.markdown("**Pytanie:**")
            st.write(row["pytanie"])
            st.markdown("**Ground Truth:**")
            st.info(row["ground_truth"])

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**🤖 LLM**")
                st.write(row["odpowiedz_llm"])
                st.caption(f"Poprawność: {row['poprawnosc_llm']}%")
            with col_b:
                st.markdown("**🔢 Vector RAG**")
                st.write(row.get("odpowiedz_vec", "–"))
                st.caption(
                    f"Popr: {row.get('poprawnosc_vec', '–')}% | "
                    f"Faith: {row.get('faithfulness_vec', '–')}% | "
                    f"Cit: {row.get('citation_precision_vec', '–')}"
                )
                with st.expander("Kontekst Vector RAG"):
                    st.code(row.get("kontekst_vec") or "(brak)", language=None)
            with col_c:
                st.markdown("**🕸️ Graph RAG**")
                st.write(row["odpowiedz_rag"])
                st.caption(
                    f"Popr: {row['poprawnosc_rag']}% | "
                    f"Faith: {row.get('faithfulness_rag', '–')}% | "
                    f"Cit: {row.get('citation_precision_rag', '–')}"
                )
                with st.expander("Kontekst Graph RAG"):
                    st.code(row["kontekst_rag"] if row["kontekst_rag"] else "(brak)", language=None)

        # Przycisk do pobrania wyników jako CSV
        st.divider()
        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="⬇️ Pobierz wyniki jako CSV",
            data=csv_bytes,
            file_name="eval_results.csv",
            mime="text/csv",
        )
