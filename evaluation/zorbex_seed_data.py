import os
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Trójki wiedzy fikcyjnej firmy Zorbex S.A. (AgriTech, Gniezno, zał. 2019).
# Dane w 100% wymyślone — żaden LLM nie ma ich w zbiorze treningowym.
# Używaj do testowania czy Graph RAG faktycznie czerpie z grafu, a nie z parametrycznej pamięci modelu.
ZORBEX_TRIPLETS: list[tuple[str, str, str]] = [

    # Informacje ogólne o firmie
    ("Zorbex S.A.", "ZALOZONO_W_ROKU", "2019"),
    ("Zorbex S.A.", "SIEDZIBA", "ul. Mieszka I 17, 62-200 Gniezno"),
    ("Zorbex S.A.", "NIP", "7842390156"),
    ("Zorbex S.A.", "FORMA_PRAWNA", "spolka akcyjna"),
    ("Zorbex S.A.", "BRANZA", "rolnicze technologie precyzyjne AgriTech"),
    ("Zorbex S.A.", "LICZBA_PRACOWNIKOW", "234"),
    ("Zorbex S.A.", "NOTOWANA_NA", "NewConnect GPW od 2022"),

    # Zarząd
    ("Marta Klejnot-Wasiak", "JEST_PREZESEM_ZARZADU", "Zorbex S.A."),
    ("Marta Klejnot-Wasiak", "PELNI_FUNKCJE_OD", "2021"),
    ("Henryk Dlubak", "BYL_PREZESEM", "Zorbex S.A."),
    ("Henryk Dlubak", "KIEROWAL_W_LATACH", "2019-2021"),
    ("Tomasz Bledowski", "JEST_CTO", "Zorbex S.A."),
    ("Zuzanna Prochownik", "JEST_CFO", "Zorbex S.A."),
    ("Radoslaw Trabka", "KIERUJE_DZIALEM", "Badania i Rozwoj R&D Zorbex S.A."),

    # Oddziały
    ("Zorbex S.A.", "ODDZIAL_CENTRALA_RD", "Gniezno"),
    ("Zorbex S.A.", "ODDZIAL_PRODUKCJA_ELEKTRONIKI", "Wroclaw"),
    ("Zorbex S.A.", "ODDZIAL_SERWIS_LOGISTYKA", "Gdansk"),

    # Produkt: SoilMind Pro 3.0
    ("SoilMind Pro 3.0", "JEST_PRODUKTEM", "Zorbex S.A."),
    ("SoilMind Pro 3.0", "MIERZY", "wilgotnosc gleby"),
    ("SoilMind Pro 3.0", "MIERZY", "temperature gleby"),
    ("SoilMind Pro 3.0", "MIERZY", "pH gleby"),
    ("SoilMind Pro 3.0", "MIERZY", "przewodnosc elektryczna gleby"),
    ("SoilMind Pro 3.0", "PROTOKOL_KOMUNIKACJI", "LoRaWAN"),
    ("SoilMind Pro 3.0", "ZASIEG_KOMUNIKACJI_KM", "2.4"),
    ("SoilMind Pro 3.0", "CENA_NETTO_PLN", "1890"),
    ("SoilMind Pro 3.0", "POBOR_MOCY_W", "0.3"),
    ("SoilMind Pro 3.0", "CZAS_PRACY_BATERII_LAT", "7"),
    ("SoilMind Pro 3.0", "STOPIEN_OCHRONY", "IP68"),
    ("SoilMind Pro 3.0", "CERTYFIKAT", "CE 2024/53/UE"),
    ("SoilMind Pro 3.0", "CHRONIONY_PATENTEM", "PL-2021-00847"),

    # Patent
    ("PL-2021-00847", "PATENT_NA", "algorytm kalibracji SoilMind Core"),
    ("PL-2021-00847", "ZAREJESTROWANY_W", "Urzad Patentowy Rzeczypospolitej Polskiej"),

    # Infrastruktura IT: Kwas Grid
    ("Kwas Grid", "JEST_PLATFORMA_SERWEROWA", "Zorbex S.A."),
    ("Kwas Grid", "CENTRUM_DANYCH_W", "ul. Przemyslowa 44, Rybnik"),
    ("Kwas Grid", "POJEMNOSC_RDZENI", "1200"),
    ("Kwas Grid", "SZYFROWANIE", "AES-256-GCM"),
    ("Kwas Grid", "OBSLUGUJE", "telemetria czujnikow SoilMind w czasie rzeczywistym"),

    # Język programowania: ZorbScript
    ("ZorbScript", "JEST_JEZYKIEM_PROGRAMOWANIA", "Zorbex S.A."),
    ("ZorbScript", "STWORZONY_PRZEZ", "Tomasz Bledowski"),
    ("ZorbScript", "BAZUJE_NA", "Python 3.11"),
    ("ZorbScript", "PRZEZNACZONY_DO", "konfiguracji czujnikow SoilMind"),
    ("ZorbScript", "ROZSZERZENIE_PLIKOW", ".zbs"),

    # Klienci
    ("Kooperatywa Zbozowa Podlasie", "JEST_KLIENTEM", "Zorbex S.A."),
    ("Kooperatywa Zbozowa Podlasie", "SIEDZIBA", "Bialystok"),
    ("Kooperatywa Zbozowa Podlasie", "UZYWA", "340 czujnikow SoilMind Pro 3.0"),
    ("Agrogospodarstwo Sadlocha i Mroz", "JEST_KLIENTEM", "Zorbex S.A."),
    ("Agrogospodarstwo Sadlocha i Mroz", "SIEDZIBA", "Zamosc"),
    ("Agrogospodarstwo Sadlocha i Mroz", "POWIERZCHNIA_GRUNTOW_HA", "8400"),

    # Inwestor
    ("Fundacja Przyszlosci Rolnictwa", "JEST_INWESTOREM", "Zorbex S.A."),
    ("Fundacja Przyszlosci Rolnictwa", "UDZIELILA_GRANTU_MLN_PLN", "12.4"),
    ("Fundacja Przyszlosci Rolnictwa", "GRANT_W_ROKU", "2020"),

    # Finanse
    ("Zorbex S.A.", "PRZYCHOD_2022_MLN_PLN", "31.7"),
    ("Zorbex S.A.", "PRZYCHOD_2023_MLN_PLN", "47.3"),
    ("Zorbex S.A.", "EBITDA_2023_MLN_PLN", "8.9"),
    ("Zorbex S.A.", "KOD_RAPORTU_ROCZNEGO_2023", "ZOR-2023-A"),

    # Nagrody
    ("Zorbex S.A.", "NAGRODA", "AgriTech Innovation Award 2022"),
    ("Zorbex S.A.", "NAGRODA", "Gazela Biznesu Dolina Krzemowa Agro 2023"),
]


def load_zorbex_triplets_to_neo4j() -> tuple[int, int]:
    # Ładuje trójki testowe ZORBEX_TRIPLETS do Neo4j przez MERGE
    # (istniejące węzły nie są duplikowane).
    # Zwraca parę (ile trójek załadowano, ile było wszystkich).
    # Przy błędzie połączenia rzuca wyjątek, który obsługuje wywołujący (app.py).
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    db_name = os.getenv("NEO4J_DATABASE", "praca")

    loaded = 0
    total = len(ZORBEX_TRIPLETS)

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=db_name) as session:
            for subj, rel, obj in ZORBEX_TRIPLETS:
                rel_clean = re.sub(r'[^a-zA-Z0-9_]', '_', rel).strip('_')  # Cypher wymaga nazw relacji tylko z liter ASCII, cyfr i podkreślenia
                if not rel_clean:
                    continue
                query = f"""
                MERGE (a:Entity {{id: $subj}})
                MERGE (b:Entity {{id: $obj}})
                MERGE (a)-[:{rel_clean}]->(b)
                """
                session.run(query, subj=subj, obj=obj)
                loaded += 1

    return loaded, total


if __name__ == "__main__":
    loaded, total = load_zorbex_triplets_to_neo4j()
    print(f"Zaladowano {loaded}/{total} trojek Zorbex S.A. do Neo4j.")
