from enum import Enum
from pyspark.sql import SparkSession
from typing import Protocol, Union
import pandas as pd
import sqlite3


class WriteMode(Enum):
    overwrite = "overwrite"
    append = "append"
    ignore = "ignore"
    error = "error"


class EngineProtocol(Protocol):
    def get_connection(
        self,
    ) -> Union[sqlite3.Connection, SparkSession]: ...

    def read(self, table_name:Enum) -> pd.DataFrame: ...
    def exec(self, query: str) -> pd.DataFrame: ...
    def write(
        self,
        df: pd.DataFrame,
        table_name: Enum,
        mode: WriteMode = WriteMode.overwrite,
    ) -> None: ...
