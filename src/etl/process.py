import pandas as pd
import sqlite3
from typing import Callable
from etl.loaders import (
    load_and_clean_csv,
    load_and_enrich_region_csv,
    load_and_clean_and_append_year_csv,
)


def insert_into_existing_table(df: pd.DataFrame, db_path: str, table_name: str):
    # Insert the expected schema ie the intersection of CSV cols and table cols
    with sqlite3.connect(db_path) as conn:
        existing_cols = pd.read_sql(f"PRAGMA table_info({table_name});", conn)[
            "name"
        ].tolist()
        available_cols = [c for c in df.columns if c in existing_cols]
        df[available_cols].to_sql(
            table_name,
            conn,
            if_exists="append",
            index=False,
        )


def process_table_method(
    func: Callable,
    database: str,
    data_path: str,
    table_name: str,
    year: int,
):
    print(f"loading table: {table_name}")
    df = func(data_path, year)
    insert_into_existing_table(df, database, table_name)


def process_table(database: str, data_path: str, table_name: str):
    print(f"loading table: {table_name}")
    df = load_and_clean_csv(data_path)
    insert_into_existing_table(df, database, table_name)


def process_table_region(database: str, data_path: str, table_name: str):
    print(f"loading table: {table_name}")
    df = load_and_enrich_region_csv(data_path)
    insert_into_existing_table(df, database, table_name)


def process_table_append_year(
    database: str,
    data_path: str,
    table_name: str,
    year: int,
):
    print(f"loading table: {table_name}")
    df = load_and_clean_and_append_year_csv(data_path, year)
    insert_into_existing_table(df, database, table_name)
