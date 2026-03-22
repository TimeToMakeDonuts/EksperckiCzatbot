import streamlit as st

# Konfiguracja strony na szeroką, aby zmieścić dwie kolumny
st.set_page_config(layout="wide", page_title="GraphRAG vs LLM Chatbot")

st.title("Ekspercki Chatbot: Porównanie RAG Grafowy vs Standardowy LLM")
st.markdown("Aplikacja demonstrująca wpływ strukturalnych powiązań danych w grafie na jakość generowanych odpowiedzi.")

# Inicjalizacja stanu sesji (przechowywanie historii czatu)
if "messages_llm" not in st.session_state:
    st.session_state.messages_llm =[]
if "messages_rag" not in st.session_state:
    st.session_state.messages_rag =[]

# Podział interfejsu na dwie równe kolumny
col1, col2 = st.columns(2)

# Funkcja pomocnicza do rysowania historii czatu
def render_chat_history(messages, column_context):
    for msg in messages:
        with column_context:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

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

    # Symulacja generowania odpowiedzi dla Standardowego LLM
    with col1:
        with st.chat_message("assistant"):
            placeholder_llm = st.empty()
            # Zaślepka
            with st.spinner("Generowanie odpowiedzi LLM..."):
                response_llm = f"**Brak dostępu do kontekstu grafu.**\n\nOdpowiedź na zapytanie: '{prompt}'."
                placeholder_llm.markdown(response_llm)
                st.session_state.messages_llm.append({"role": "assistant", "content": response_llm})

    # Symulacja generowania odpowiedzi dla Graph RAG
    with col2:
        with st.chat_message("assistant"):
            placeholder_rag = st.empty()
            # Zaślepka
            with st.spinner("Przeszukiwanie grafu i generowanie odpowiedzi..."):
                response_rag = f"**Wykorzystano kontekst z Grafu Wiedzy.**\n\nZnaleziono encje powiązane z zapytaniem: '{prompt}'."
                placeholder_rag.markdown(response_rag)
                st.session_state.messages_rag.append({"role": "assistant", "content": response_rag})

    # Wymuszenie odświeżenia, aby scroll zjechał na dół
    st.rerun()