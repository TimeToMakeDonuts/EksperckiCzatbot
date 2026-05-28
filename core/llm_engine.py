from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Zmienne globalne
llm = None
_embed_initialized = False

# Funkcja dynamicznie (re)inicjalizująca model LLM na podstawie parametrów ze Streamlit
def init_or_update_llm(api_base="http://localhost:1234/v1", api_key="lm-studio", model_name="local-model", temp=0.1,
                       max_tok=512):
    global llm, _embed_initialized

    llm = OpenAILike(
        api_base=api_base,
        api_key=api_key,
        model=model_name,
        is_chat_model=True,
        temperature=temp,
        max_tokens=max_tok,
        timeout=300.0
    )

    Settings.llm = llm

    # Inicjujemy model wektoryzujący (Embeddings) tylko raz, jeśli go jeszcze nie ma
    if not _embed_initialized:
         Settings.embed_model = HuggingFaceEmbedding(model_name="sdadas/mmlw-retrieval-roberta-large")
         _embed_initialized = True

    return llm

# Pierwsze uruchomienie przy starcie aplikacji (domyślne wartości)
try:
    init_or_update_llm()
except Exception:
    pass

def get_standard_llm_response(prompt: str) -> str:
    # Funkcja odpytująca standardowy model LLM (bez kontekstu RAG).
    try:
        response = llm.complete(prompt)
        return response.text
    except Exception as e:
        return f"**Błąd połączenia z Modelem:** Upewnij się, że serwer i adres API są poprawne. Szczegóły: {e}"