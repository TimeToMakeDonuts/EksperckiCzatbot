import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import os
import re
import tempfile
from pyvis.network import Network
from neo4j import GraphDatabase

# logika backendowa
from core.llm_engine import get_standard_llm_response, llm
from core.graph_rag import get_graph_rag_response
from core.graph_builder import extract_text_from_pdf, extract_triplets_with_llm, save_edited_triplets_to_neo4j

st.set_page_config(layout="wide", page_title="GraphRAG vs LLM Chatbot")
st.title("System Ekspercki: RAG Grafowy")

# FUNKCJE WIZUALIZACJI I METRYK

# Odpytanie Neo4j i utworzenie grafu
def get_visual_graph_html(prompt_text):

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))

        # Pobieramy 30 relacji z bazy (możliwe do zmiany sposób - testowanie)
        query = """
        MATCH (n)-[r]->(m)
        RETURN n.id AS source, type(r) AS rel, m.id AS target
        LIMIT 30
        """
        with driver.session(database="praca") as session:
            records = list(session.run(query))

        if not records:
            return None

        # Rysowanie interaktywnego grafu w PyVis
        net = Network(height="350px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
        for record in records:
            src, rel, tgt = str(record["source"]), str(record["rel"]), str(record["target"])
            net.add_node(src, label=src, title=src, color="#ff4b4b")
            net.add_node(tgt, label=tgt, title=tgt, color="#4b4bff")
            net.add_edge(src, tgt, title=rel, label=rel, color="#aaaaaa")

        net.toggle_physics(True)

        # Zapisz i odczytaj HTML
        path = tempfile.mktemp(suffix=".html")
        net.save_graph(path)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        return html
    except Exception as e:
        return f"<p style='color:red;'>Błąd wizualizacji grafu: {e}</p>"

    #Ocena wierność odpowiedzi względem faktów z grafu.
def calculate_faithfulness(generated_answer, context):

    if not context.strip():
        return 0

    prompt = f"""
    Jako obiektywny sędzia AI, oceń WIERNOŚĆ (faithfulness) poniższej odpowiedzi względem faktów.
    Fakty wprost z bazy danych:
    {context}

    Odpowiedź do oceny:
    {generated_answer}

    W jakim stopniu (od 0 do 100) wygenerowana odpowiedź bazuje WYŁĄCZNIE na podanych faktach? 
    Zwróć TYLKO LICZBĘ (np. 100). Nie pisz żadnego innego tekstu.
    """
    try:
        eval_result = llm.complete(prompt).text
        match = re.search(r'\d+', eval_result)
        if match:
            score = int(match.group())
            return min(max(score, 0), 100)
        return 100
    except:
        return 100


# Inicjalizacja stanu sesji (przechowywanie historii czatu) - definiujemy to globalnie na starcie
if "messages_llm" not in st.session_state:
    st.session_state.messages_llm = []
if "messages_rag" not in st.session_state:
    st.session_state.messages_rag = []
if "latest_graph_html" not in st.session_state:
    st.session_state.latest_graph_html = None
if "latest_metric" not in st.session_state:
    st.session_state.latest_metric = None

# Zakładki
tab1, tab2 = st.tabs(["Czat z Modelem", "Baza Wiedzy (Dodaj Dane)"])

# Funkcja pomocnicza do rysowania historii czatu
def render_chat_history(messages, column_context):
    for msg in messages:
        with column_context:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

# ZAKŁADKA 1: CZAT Z MODELEM
with tab1:
    col1, col2 = st.columns(2)

    # --- WIDOK KOLUMNY 1: STANDARDOWY LLM ---
    with col1:
        st.subheader("Standardowy LLM")
        st.caption("Odpowiedź generowana wyłącznie na podstawie wbudowanej wiedzy modelu.")
        render_chat_history(st.session_state.messages_llm, col1)

    # --- WIDOK KOLUMNY 2: GRAPH RAG ---
    with col2:
        st.subheader("LLM + Graph RAG")
        st.caption("Odpowiedź wzbogacona o kontekst pobrany z grafowej bazy danych.")
        render_chat_history(st.session_state.messages_rag, col2)

        # Wyświetlanie metryki
        if st.session_state.latest_metric is not None:
            st.divider()
            st.caption("Analiza ostatniej odpowiedzi RAG")
            score = st.session_state.latest_metric
            color = "green" if score > 80 else "orange" if score > 50 else "red"
            st.markdown(f"**Metryka Truthfulness (Wierność źródłom):** :{color}[**{score}%**]")
            st.progress(score / 100.0)

        # Wyświetlanie wizualizacji grafu
        if st.session_state.latest_graph_html:
            with st.expander("Zobacz strukturę grafu", expanded=False):
                components.html(st.session_state.latest_graph_html, height=360)

    # --- POLE WPROWADZANIA ZAPYTANIA ---
    prompt = st.chat_input("Wpisz swoje zapytanie.")

    if prompt:
        # Dodanie pytania użytkownika do historii w obu kolumnach
        st.session_state.messages_llm.append({"role": "user", "content": prompt})
        st.session_state.messages_rag.append({"role": "user", "content": prompt})

        # Wyświetlenie pytania użytkownika na bieżąco
        with col1:
            with st.chat_message("user"):
                st.markdown(prompt)
        with col2:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Generowanie odpowiedzi dla Standardowego LLM
        with col1:
            with st.chat_message("assistant"):
                with st.spinner("Generowanie odpowiedzi w LM Studio..."):
                    response_llm = get_standard_llm_response(prompt)
                    st.markdown(response_llm)
                    st.session_state.messages_llm.append({"role": "assistant", "content": response_llm})

        # Generowanie odpowiedzi dla Graph RAG
        with col2:
            with st.chat_message("assistant"):
                with st.spinner("Przeszukiwanie grafu i generowanie odpowiedzi..."):
                    rag_output = get_graph_rag_response(prompt)

                    odpowiedz = rag_output["answer"]
                    kontekst = rag_output["context"]

                    st.markdown(odpowiedz)
                    st.session_state.messages_rag.append({"role": "assistant", "content": odpowiedz})

                # Obliczanie metryki Faithfulness
                with st.spinner("Wynik faithfulness..."):
                    if kontekst:
                        score = calculate_faithfulness(odpowiedz, kontekst)
                        st.session_state.latest_metric = score
                    else:
                        st.session_state.latest_metric = None

                # Generowanie Wizualizacji PyVis
                with st.spinner("Renderowanie interaktywnego grafu..."):
                    html_graph = get_visual_graph_html(prompt)
                    if html_graph:
                        st.session_state.latest_graph_html = html_graph

        # Wymuszenie odświeżenia, aby scroll zjechał na dół
        st.rerun()

# ZAKŁADKA 2: EKSTRAKCJA I EDYCJA
with tab2:
    st.header("Zarządzanie Grafem Wiedzy")
    st.markdown("Wprowadź tekst lub plik PDF")

    # Opcje wprowadzania danych
    input_method = st.radio("Sposób wprowadzania danych:",["Wklej Tekst", "Wgraj PDF"])

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

    # Przycisk uruchamiający model lokalny
    if st.button("Znajdź Encje i Relacje", type="primary"):
        if text_to_process.strip() == "":
            st.warning("Brak wprowadzonego tekstu")
        else:
            with st.spinner("LLM analizuje tekst..."):
                triplets = extract_triplets_with_llm(text_to_process)
                if triplets:
                    # Zapis do sesji, żeby tabela nie znikała przy klikaniu
                    st.session_state.temp_df = pd.DataFrame(triplets)
                    st.success(f"Znaleziono {len(triplets)} potencjalnych relacji")
                else:
                    st.error("Model nie znalazł żadnych relacji w podanym tekście.")

    # Pokazujemy tabelę do edycji, jeśli mamy wyciągnięte dane
    if "temp_df" in st.session_state:
        st.subheader("Panel Edycji Relacji")
        st.info("Możesz edytować komórki, dodawać nowe wiersze na dole tabeli lub usunąć.")

        # Tabela
        edited_df = st.data_editor(
            st.session_state.temp_df,
            num_rows="dynamic",  # Pozwala dodawać i usuwać wiersze
            width='stretch'
        )

        # Przycisk zatwierdzenia
        if st.button("Zapisz zatwierdzone dane"):
            with st.spinner("Aktualizowanie grafowej bazy danych..."):
                success = save_edited_triplets_to_neo4j(edited_df)
                if success:
                    st.success("Wiedza została poprawnie wprowadzona.")
                    del st.session_state.temp_df
                    st.rerun()
                else:
                    st.error("Wystąpił błąd podczas zapisu do bazy.")