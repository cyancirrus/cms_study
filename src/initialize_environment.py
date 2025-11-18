from dotenv import load_dotenv
from typing import Final
from database.bridge import EngineProtocol
from database.unify import retrieve_engine
import os

load_dotenv()

BACKEND: Final[str] = os.getenv("BACKEND", "sqlite")
DATABASE: Final[str] = os.getenv("DATABASE", "source.db")
RANDOM_STATE: Final[int] = int(os.getenv("RANDOM_STATE", 42))
ENGINE: Final[EngineProtocol] = retrieve_engine(BACKEND, DATABASE)
