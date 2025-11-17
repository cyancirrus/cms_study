# # import pytest
# from enum import Enum
# from dotenv import load_dotenv
# from src.dataengine.bridge import EngineProtocol
# from src.dataengine.unify import retrieve_engine
# from typing import Final
# import os

# load_dotenv()

# BACKEND: Final[str] = os.getenv("BACKEND", "sqlite")
# DATABASE: Final[str] = os.getenv("DATABASE", "source.db")
# ENGINE: Final[EngineProtocol] = retrieve_engine(BACKEND, DATABASE)


# class ZipTables(Enum):
#     ZIP_LAT_LONG = "zip_lat_long"
#     ZIP_DEMOGRAPHICS = "zip_demographics"
#     HVBP_TPS = "hvbp_tps"  # optional sanity check


# def get_table_zip_codes(table: ZipTables) -> set[int]:
#     """Fetch unique non-null ZIP codes via the ENGINE."""
#     df = ENGINE.read(table)
#     zips = df["zip_code"].dropna().unique()
#     return set(zips)


# def test_zip_lat_long_not_null():
#     zips = get_table_zip_codes(ZipTables.ZIP_LAT_LONG)
#     assert zips, "zip_lat_long has no ZIP codes!"


# def test_zip_demographics_not_null():
#     zips = get_table_zip_codes(ZipTables.ZIP_DEMOGRAPHICS)
#     assert zips, "zip_demographics has no ZIP codes!"


# def test_zip_counts_match():
#     zips_lat_long = get_table_zip_codes(ZipTables.ZIP_LAT_LONG)
#     zips_demo = get_table_zip_codes(ZipTables.ZIP_DEMOGRAPHICS)
#     assert (
#         zips_lat_long == zips_demo
#     ), "ZIP codes in zip_lat_long and zip_demographics do not match!"


# def test_hvbp_tps_subset_of_master():
#     master_zips = get_table_zip_codes(ZipTables.ZIP_LAT_LONG)
#     hvbp_zips = get_table_zip_codes(ZipTables.HVBP_TPS)
#     assert hvbp_zips.issubset(
#         master_zips
#     ), "hvbp_tps contains ZIP codes not in master ZIP list!"
