from src.database.bridge import EngineProtocol
from src.database.m_spark import SparkEngine
from src.database.m_sqlite import SQLiteEngine


def retrieve_engine(backend: str, database: str) -> EngineProtocol:
    match backend:
        case "sqlite":
            return SQLiteEngine(database)
        case "spark":
            return SparkEngine(database)
        case _:
            raise ValueError(f"Unssupported engine type {backend}")
