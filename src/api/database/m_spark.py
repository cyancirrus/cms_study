import pandas as pd
import pandas as pd
from enum import Enum
from typing import List
from pyspark.sql import SparkSession
from src.api.database.bridge import EngineProtocol, WriteMode


class SparkEngine(EngineProtocol):
    def __init__(self, database: str):
        self.conn = SparkSession.builder.appName(
            "healthy_sphinx"
        ).getOrCreate()

    def get_connection(self) -> SparkSession:
        return self.conn

    def read(self, table_name: Enum) -> pd.DataFrame:
        query = f"""
            SELECT
                *
            FROM {table_name.value}
        """
        df_spark = self.conn.sql(query)
        return df_spark.toPandas()

    def exec(self, query: str) -> pd.DataFrame:
        df_spark = self.conn.sql(query)
        return df_spark.toPandas()

    def write(
        self,
        df: pd.DataFrame,
        table_name: Enum,
        mode: WriteMode = WriteMode.overwrite,
    ) -> None:
        """
        Write pandas DataFrame to Spark table.
        mode: 'overwrite', 'append', 'ignore', 'error' (default Spark modes)
        """
        spark_df = self.conn.createDataFrame(df)
        spark_df.write.mode(mode.value).saveAsTable(table_name.value)

    def table_columns(self, table_name: Enum) -> List[str]:
        df = self.conn.table(table_name.value)
        return df.columns
