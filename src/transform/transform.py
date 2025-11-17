from transform.process import zip_to_msa
from database.bridge import EngineProtocol

def extract_all_years_cms(database: str):
    # Historical data is only consistent back to 2021
    extract_cms_data(database, "./data/source/", 2025)
    extract_cms_data(database, "./data/historical/2024", 2024)
    extract_cms_data(database, "./data/historical/2023", 2023)
    extract_cms_data(database, "./data/historical/2022", 2022)
    extract_cms_data(database, "./data/historical/2021", 2021)


