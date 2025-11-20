import pandas as pd
import numpy as np
from tables import CmsSchema
from database.bridge import EngineProtocol, WriteMode
from initialize_environment import CURRENT_YEAR


def normalize_series(s: pd.Series, invert=False) -> pd.Series:
    """Normalize to 0-1, optionally invert (for rates where lower is better)."""
    min_val, max_val = s.min(), s.max()
    if max_val == min_val:
        return pd.Series(0.5, index=s.index)
    norm = (s - min_val) / (max_val - min_val)
    return 1 - norm if invert else norm


def compute_combined_scores(df_ipfqr, df_hvbp, df_tps) -> pd.DataFrame:
    # Normalize as before
    ipfqr_cols = [
        c for c in df_ipfqr.columns if "rate" in c or "percent" in c
    ]
    hvbp_cols = [
        c for c in df_hvbp.columns if "mort" in c or "comp" in c
    ]
    tps_cols = [c for c in df_tps.columns if "pred" in c]

    for df, cols, invert in [
        (df_ipfqr, ipfqr_cols, True),
        (df_hvbp, hvbp_cols, True),
        (df_tps, tps_cols, False),
    ]:
        for col in cols:
            df[col] = normalize_series(df[col], invert=invert)

    # Outer merge on facility_id + submission_year
    df_merge = df_ipfqr.merge(
        df_hvbp, on=["facility_id", "submission_year"], how="outer"
    ).merge(df_tps, on=["facility_id", "submission_year"], how="outer")

    # Linear average per row, ignoring missing values
    all_score_cols = ipfqr_cols + hvbp_cols + tps_cols
    df_merge["score"] = df_merge[all_score_cols].mean(
        axis=1, skipna=True
    )

    # Add recommendation_year = submission_year + 1
    df_merge["recommendation_year"] = df_merge["submission_year"] + 1

    return df_merge[
        [
            "facility_id",
            "submission_year",
            "recommendation_year",
            "score",
        ]
    ]


# def attach_metadata(df_score, df_hosp, df_zip) -> pd.DataFrame:
#     df = df_score.merge(df_hosp[['submission_year', 'facility_id', 'facility_name', 'hospital_type', 'address', 'zip_code', 'state']],
#                         on='facility_id', how='inner')
#     df = df.merge(df_zip, on='zip_code', how='left')

#     # Expand to multiple service types per facility
#     service_types = ['Psychiatry', 'Medicare', 'General']
#     df_long = pd.concat([
#         df.assign(service_type=st) for st in service_types
#     ], ignore_index=True)
#     return df_long[['recommendation_year', 'facility_id', 'facility_name', 'hospital_type', 'address', 'zip_code', 'state', 'service_type', 'score']]


def compute_combined_scores_union(
    df_ipfqr, df_hvbp, df_tps
) -> pd.DataFrame:
    def score_table(df, cols, invert=False, service_type=None):
        df = df.copy()
        for col in cols:
            df[col] = normalize_series(df[col], invert=invert)
        df["score"] = df[cols].mean(axis=1)
        df["recommendation_year"] = df["submission_year"]
        df["service_type"] = service_type
        return df[
            [
                "facility_id",
                "submission_year",
                "recommendation_year",
                "score",
                "service_type",
            ]
        ]

    ipfqr_cols = [
        c for c in df_ipfqr.columns if "rate" in c or "percent" in c
    ]
    hvbp_cols = [
        c for c in df_hvbp.columns if "mort" in c or "comp" in c
    ]
    tps_cols = [c for c in df_tps.columns if "pred" in c]

    dfs = [
        score_table(
            df_ipfqr, ipfqr_cols, invert=True, service_type="Medicare"
        ),
        score_table(
            df_hvbp, hvbp_cols, invert=True, service_type="General"
        ),
        score_table(
            df_tps, tps_cols, invert=False, service_type="Psychiatry"
        ),
    ]

    df_union = pd.concat(dfs, ignore_index=True)
    print(f"DF UNION SHAPE : {df_union.shape}")
    return df_union


def attach_metadata(df_score, df_hosp, df_zip) -> pd.DataFrame:
    df_hosp_unique = df_hosp.drop_duplicates(subset=["facility_id"])
    df = df_score.merge(
        df_hosp_unique[
            [
                "facility_id",
                "facility_name",
                "hospital_type",
                "address",
                "zip_code",
                "state",
            ]
        ],
        on="facility_id",
        how="inner",
    )
    df = df.merge(df_zip, on="zip_code", how="left")
    print(f"DF LONG : {df.shape}")
    return df[
        [
            "recommendation_year",
            "facility_id",
            "facility_name",
            "hospital_type",
            "address",
            "zip_code",
            "state",
            "service_type",
            "latitude",
            "longitude",
            "score",
        ]
    ]


# def attach_metadata(df_score, df_hosp, df_zip) -> pd.DataFrame:
#     # Keep only unique facility_id rows from hospital table
#     df_hosp_unique = df_hosp.drop_duplicates(subset=['facility_id'])

#     # Merge
#     df = df_score.merge(
#         df_hosp_unique[['facility_id', 'facility_name', 'hospital_type', 'address', 'zip_code', 'state']],
#         on='facility_id', how='inner'
#     )

#     print(f"DF SHAPE : {df.shape}")
#     # Merge lat/lon
#     df = df.merge(df_zip, on='zip_code', how='left')

#     # Expand to multiple service types per facility
#     service_types = ['Psychiatry', 'Medicare', 'General']
#     df_long = pd.concat([
#         df.assign(service_type=st) for st in service_types
#     ], ignore_index=True)
#     print(f"DF LONG : {df_long.shape}")
#     # Merge lat/lon

#     return df_long[['recommendation_year', 'facility_id', 'facility_name', 'hospital_type', 'address', 'zip_code', 'state', 'service_type', 'score']]


def build_recommendation_table(engine, current_year: int):
    # Read prediction tables
    df_ipfqr = engine.read(
        CmsSchema.prediction_ipfqr_quality_measures_facility
    )
    df_hvbp = engine.read(CmsSchema.prediction_hvbp_clinical_outcomes)
    df_tps = engine.read(CmsSchema.prediction_hvbp_tps)

    # Hospital metadata
    df_hosp = engine.exec(
        f"select * from hospital_general_information where submission_year = {current_year}"
    )
    df_zip = engine.read(CmsSchema.zip_lat_long)

    # Compute scores
    df_score = compute_combined_scores_union(df_ipfqr, df_hvbp, df_tps)

    # Attach metadata & expand service types
    df_final = attach_metadata(df_score, df_hosp, df_zip)
    print(df_final.columns)
    # Write table
    engine.write(
        df_final, CmsSchema.recommendation_hospital, WriteMode.overwrite
    )  # or another table name
