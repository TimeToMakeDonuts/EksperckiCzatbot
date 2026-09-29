from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Globalne instancje modeli — współdzielone przez wszystkie moduły
llm = None
judge_llm = None        # oddzielny model oceniający (LLM-as-Judge); None = używa llm
_embed_initialized = False


def get_judge() -> "OpenAILike":
    # Zwraca model-sędziego, jeśli został skonfigurowany osobno.
    # W przeciwnym razie sędzią jest główny model llm.
    return judge_llm if judge_llm is not None else llm


def init_or_update_llm(api_base="http://localhost:1234/v1", api_key="lm-studio",
                       model_name="local-model", temp=0.1, max_tok=512):
    # Tworzy (lub odświeża) połączenie z serwerem LLM, np. LM Studio.
    # Wywoływana po kliknięciu "Zapisz i zastosuj ustawienia" w panelu bocznym.
    # Model embeddingów (potrzebny do Vector RAG) jest ładowany tylko raz na sesję.
    global llm, _embed_initialized
    llm = OpenAILike(
        api_base=api_base,
        api_key=api_key,
        model=model_name,
        is_chat_model=True,
        temperature=temp,
        max_tokens=max_tok,
        timeout=300.0,
    )
    Settings.llm = llm
    # Model embeddingów ładuje się tylko raz - zajmuje kilka sekund i ~1 GB RAM
    if not _embed_initialized:
        Settings.embed_model = HuggingFaceEmbedding(model_name="sdadas/mmlw-retrieval-roberta-large")
        _embed_initialized = True
    return llm


def init_or_update_judge_llm(api_base: str, api_key: str, model_name: str,
                              temp: float = 0.1, max_tok: int = 512) -> "OpenAILike":
    # Konfiguruje osobny model do oceniania odpowiedzi (LLM-as-Judge).
    global judge_llm
    judge_llm = OpenAILike(
        api_base=api_base,
        api_key=api_key,
        model=model_name,
        is_chat_model=True,
        temperature=temp,
        max_tokens=max_tok,
        timeout=300.0,
    )
    return judge_llm

def reset_judge_llm() -> None:
    # Wyłącza osobnego sędziego: ocenianie wraca do głównego modelu llm.
    global judge_llm
    judge_llm = None

# Inicjalizacja przy starcie aplikacji z domyślnymi wartościami
try:
    init_or_update_llm()
except Exception:
    pass  # serwer LLM może nie być uruchomiony przy starcie — użytkownik skonfiguruje go w sidebar

def get_standard_llm_response(prompt: str) -> str:
    # Odpytuje główny model bez żadnego kontekstu z bazy wiedzy (bez RAG).
    # To punkt odniesienia w porównaniu: model odpowiada tylko z własnej pamięci.
    # Przy błędzie połączenia zwraca tekst z komunikatem zamiast rzucać wyjątek.
    system_prompt = (
        "Jesteś pomocnym i rzeczowym asystentem. "
        "Odpowiadaj zawsze w języku polskim, chyba że użytkownik wyraźnie prosi o inny język. "
        "Staraj się być precyzyjny i zwięzły.\n\n"
    )
    full_prompt = f"{system_prompt}Pytanie użytkownika: {prompt}\n\nOdpowiedź:"
    try:
        return llm.complete(full_prompt).text
    except Exception as e:
        return f"**Błąd połączenia z Modelem:** Upewnij się, że serwer i adres API są poprawne. Szczegóły: {e}"
