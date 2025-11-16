from dotenv import load_dotenv
from typing import Final
import os

load_dotenv()

DATABASE: Final[str] = str(os.getenv("DATABASE"))

# if DATABASE is None:
#     raise RuntimeError(".env missing DATABASE variable")
