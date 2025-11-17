import pandas as pd
from enum import Enum
from typing import Callable, List
from database.bridge import EngineProtocol, WriteMode
from etl.loaders import (
    load_and_clean_csv,
    load_and_enrich_region_csv,
    load_and_clean_and_append_year_csv,
)


def insert_into_existing_table(
    engine: EngineProtocol, df: pd.DataFrame, table_name: Enum
):
    # Insert the expected schema ie the intersection of CSV cols and table cols
    available_cols: List[str] = [
        c for c in df.columns if c in engine.table_columns(table_name)
    ]
    df = df.loc[:, available_cols]
    assert isinstance(df, pd.DataFrame)
    engine.write(df, table_name, WriteMode.append)


def process_table_method(
    func: Callable,
    engine: EngineProtocol,
    data_path: str,
    table_name: Enum,
    year: int,
):
    print(f"loading table: {table_name}")
    df = func(data_path, year)
    insert_into_existing_table(engine, df, table_name)


def process_table(
    engine: EngineProtocol, data_path: str, table_name: Enum
):
    print(f"loading table: {table_name}")
    df = load_and_clean_csv(data_path)
    insert_into_existing_table(engine, df, table_name)


def process_table_region(
    engine: EngineProtocol, data_path: str, table_name: Enum, year: int
):
    print(f"loading table: {table_name}")
    df = load_and_enrich_region_csv(data_path, year)
    insert_into_existing_table(engine, df, table_name)


def process_table_append_year(
    engine: EngineProtocol, data_path: str, table_name: Enum, year: int
):
    print(f"loading table: {table_name}")
    df = load_and_clean_and_append_year_csv(data_path, year)
    insert_into_existing_table(engine, df, table_name)
