import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    # Ustawia format i poziom logów dla całej aplikacji (na konsolę).
    # Wywołaj raz przy starcie. Biblioteki zewnętrzne są wyciszone do poziomu
    # WARNING, żeby nie zaśmiecały logów.
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # overrides any basicConfig already called by imported libraries
    )
    # Wycisz biblioteki zewnętrzne, które domyślnie logują zbyt dużo
    for noisy in ("neo4j", "llama_index", "httpx", "httpcore", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    # Zwraca logger dla danego modułu.
    # Użycie na górze pliku: logger = get_logger(__name__)
    return logging.getLogger(name)
