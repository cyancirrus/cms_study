from initialize_environment import ENGINE
from etl.extract import (
    extract_all_years_cms,
    extract_augmented_tables,
)

if __name__ == "__main__":
    extract_all_years_cms(ENGINE)
    extract_augmented_tables(ENGINE)
