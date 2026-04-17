import streamlit as st
import pandas as pd

# logika backendowa
from core.llm_engine import get_standard_llm_response
from core.graph_rag import get_graph_rag_response
from core.graph_builder import extract_text_from_pdf, extract_triplets_with_llm, save_edited_triplets_to_neo4j

st.set_page_config(layout="wide", page_title="GraphRAG vs LLM Chatbot")
st.title("System Ekspercki: RAG Grafowy")

# Inicjalizacja stanu sesji (przechowywanie historii czatu) - definiujemy to globalnie na starcie
if "messages_llm" not in st.session_state:
    st.session_state.messages_llm =[]
if "messages_rag" not in st.session_state:
    st.session_state.messages_rag =[]

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
    # Podział interfejsu na dwie równe kolumny MUSI być wewnątrz zakładki
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
                    response_rag = get_graph_rag_response(prompt)
                    st.markdown(response_rag)
                    st.session_state.messages_rag.append({"role": "assistant", "content": response_rag})

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
        st.info(
            "Możesz edytować komórki, dodawać nowe wiersze na dole tabeli lub usunąć.")

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