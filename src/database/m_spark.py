from pyspark.sql import SparkSession
import pandas as pd
import pandas as pd
from pyspark.sql import SparkSession
from database.bridge import EngineProtocol, WriteMode


class SparkEngine(EngineProtocol):
    def __init__(self, database: str):
        self.conn = SparkSession.builder.appName("healthy_sphinx").getOrCreate()

    def get_connection(self) -> SparkSession:
        return self.conn

    def read(self, query: str) -> pd.DataFrame:
        df_spark = self.conn.sql(query)
        return df_spark.toPandas()

    def write(
        self, df: pd.DataFrame, table_name: str, mode: WriteMode = WriteMode.overwrite
    ) -> None:
        """
        Write pandas DataFrame to Spark table.
        mode: 'overwrite', 'append', 'ignore', 'error' (default Spark modes)
        """
        spark_df = self.conn.createDataFrame(df)
        spark_df.write.mode(mode.value).saveAsTable(table_name)
