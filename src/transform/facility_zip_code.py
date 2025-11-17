import pandas as pd


def create_facility_zip_code(
    hospital_general_information: pd.DataFrame,
) -> pd.DataFrame:
    x = hospital_general_information[
        ["submission_year", "facility_id", "zip_code"]
    ].drop_duplicates()
    assert isinstance(x, pd.DataFrame)
    return x
