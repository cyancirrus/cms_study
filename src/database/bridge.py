from initialize_environment import DATABASE
from pyspark.sql import SparkSession
from typing import Protocol, Union
import os
import pandas as pd
import pandas as pd
import sqlite3


BACKEND = os.getenv("BACKEND", "sqlite")
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

        _SPARK = SparkSession.builder.appName(
            "healthy_sphinx"
        ).getOrCreate()
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


class EngineProtocol(Protocol):
    def get_connection(
        self,
    ) -> Union[sqlite3.Connection, SparkSession]: ...

    def run_query(self, query: str) -> pd.DataFrame: ...


class SQLiteEngine:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

    def get_connection(self):
        return self.conn

    def run_query(self, query: str) -> pd.DataFrame:
        cursor = self.conn.execute(query)
        rows = cursor.fetchall()
        columns = [str(desc[0]) for desc in cursor.description]
        df = pd.DataFrame(rows)
        df.columns = columns
        return df


def query_bridge(engine: EngineProtocol, query: str) -> pd.DataFrame:
    conn = engine.get_connection()  # could be reused / pooled
    return engine.run_query(query)
