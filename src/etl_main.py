from initialize_environment import DATABASE
from etl.extract import (
    extract_all_years_cms,
    extract_augmented_tables,
)

if __name__ == "__main__":
    extract_all_years_cms(DATABASE)
    extract_augmented_tables(DATABASE)
