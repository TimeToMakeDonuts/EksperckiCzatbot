from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def setup_llm_environment():
    # Konfiguracja LM Studio za pomocą klasy OpenAILike
    local_llm = OpenAILike(
        api_base="http://localhost:1234/v1",
        api_key="lm-studio",
        model="google/gemma-3-27b",
        is_chat_model=True,
        temperature=0.1,
        max_tokens=2000,
        timeout = 300.0,
    )

    # Konfiguracja darmowego modelu osadzeń
    local_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    # Zastosowanie ustawień
    Settings.llm = local_llm
    Settings.embed_model = local_embed_model

    return local_llm

llm = setup_llm_environment()

def get_standard_llm_response(prompt: str) -> str:
    # Funkcja odpytująca standardowy model LLM (bez kontekstu RAG).
    try:
        response = llm.complete(prompt)
        return response.text
    except Exception as e:
        return f"**Błąd połączenia z LM Studio:** Upewnij się, że serwer działa. Szczegóły: {e}"