import os
import pandas as pd
import pyspark

BACKEND = os.getenv("DB_BACKEND", "sqlite")  # "sqlite" or "spark"

# --- SQLite backend ---
import sqlite3

SQLITE_PATH = os.getenv("SQLITE_PATH", "source.db")


def _sqlite_query(sql: str, params=None) -> pd.DataFrame:
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        return pd.read_sql_query(sql, conn, params=params or {})
    finally:
        conn.close()


# --- Spark backend (stubbed, fill in later) ---
_SPARK = None


def _get_spark():
    global _SPARK
    if _SPARK is None:
        from pyspark.sql import SparkSession

        _SPARK = SparkSession.builder.appName("healthy_sphinx").getOrCreate()
    return _SPARK


def _spark_query(sql: str, params=None) -> pd.DataFrame:
    spark = _get_spark()
    # naive param interpolation; you can improve this later
    if params:
        for k, v in params.items():
            placeholder = f":{k}"
            if isinstance(v, str):
                v = f"'{v}'"
            sql = sql.replace(placeholder, str(v))
    df_spark = spark.sql(sql)
    return df_spark.toPandas()


# --- public API ---
def query(sql: str, params=None) -> pd.DataFrame:
    if BACKEND == "sqlite":
        return _sqlite_query(sql, params=params)
    elif BACKEND == "spark":
        return _spark_query(sql, params=params)
    else:
        raise ValueError(f"Unknown DB_BACKEND={BACKEND!r}")
