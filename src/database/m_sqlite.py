import pandas as pd
import pandas as pd
import sqlite3
from typing import Literal
from database.bridge import EngineProtocol, WriteMode


class SQLiteEngine(EngineProtocol):
    def __init__(self, db_path: str):
        self.conn: sqlite3.Connection = sqlite3.connect(
            db_path, check_same_thread=False
        )

    def get_connection(self):
        return self.conn

    def read(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.conn)

    def write(
        self, df: pd.DataFrame, table_name: str, mode: WriteMode = WriteMode.overwrite
    ) -> None:
        """
        Write DataFrame to SQLite table.
        """
        sqlite_mode = self._map_mode_(mode)
        df.to_sql(table_name, self.conn, if_exists=sqlite_mode, index=False)

    @staticmethod
    def _map_mode_(mode: WriteMode) -> Literal["replace", "append", "fail"]:
        """
        Maps the write mode to sqlites internal specification
        """
        match mode:
            case WriteMode.overwrite:
                return "replace"
            case WriteMode.append:
                return "append"
            case WriteMode.ignore:
                return "replace"
            case WriteMode.error:
                return "fail"
