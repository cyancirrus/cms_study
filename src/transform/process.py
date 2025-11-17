import pandas as pd
import numpy as np
from typing import Final, Dict
from sklearn.neighbors import BallTree

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

LINECODE_FEATURE_MAP: Final[Dict[str, str]] = {
    "1": "personal_income_k",
    "2": "population_density",
    "3": "per_capita_income",
}


# TODO: Ai generated script, did some things to test like basic data but need to review and add asserts
def zip_to_msa(
    zip_lat_long: pd.DataFrame,
    msa_dim: pd.DataFrame,
    msa_centroids: pd.DataFrame,
    msa_stats: pd.DataFrame,
):
    """
    Map ZIPs to nearest MSA, attach MSA dim and stats, output wide format ready for modeling.
    """
    # --- 1a. Filter centroids to MSAs that have stats ---
    msa_with_stats = msa_stats["cbsafp"].dropna().unique()
    msa_centroids_filtered = msa_centroids[
        msa_centroids["cbsafp"].isin(msa_with_stats)
    ]

    # --- 1b. Map ZIP to nearest MSA using filtered centroids ---
    zip_coords = np.radians(
        zip_lat_long[["latitude", "longitude"]].values
    )
    msa_coords = np.radians(
        msa_centroids_filtered[["latitude", "longitude"]].values
    )

    tree = BallTree(msa_coords, metric="haversine")
    dist, idx = tree.query(zip_coords, k=1)

    zip_lat_long["cbsafp"] = msa_centroids_filtered.iloc[idx.flatten()][
        "cbsafp"
    ].values
    zip_lat_long["msa_title"] = msa_centroids_filtered.iloc[
        idx.flatten()
    ]["msa_title"].values
    zip_lat_long["state_abbreviation"] = msa_centroids_filtered.iloc[
        idx.flatten()
    ]["state_abbreviation"].values
    zip_lat_long["distance_km"] = dist.flatten() * 6371

    # --- 2. Merge MSA dim ---
    zip_lat_long = zip_lat_long.merge(
        msa_dim[["cbsafp", "cbsa_title", "state_abbreviation"]],
        on=["cbsafp", "state_abbreviation"],
        how="left",
    )

    # --- 3. Map linecodes to descriptive feature names ---
    msa_stats["feature_name"] = msa_stats["linecode"].replace(
        LINECODE_FEATURE_MAP
    )

    # --- 4. Pivot stats to wide ---
    stats_wide = msa_stats.pivot_table(
        index="cbsafp", columns="feature_name", values="metric"
    ).reset_index()

    # --- 5. Merge wide stats into ZIPs ---
    zip_wide = zip_lat_long.merge(stats_wide, on="cbsafp", how="left")
    print("ZIP COLUMNS")
    print(zip_wide.columns)
    zip_wide = zip_wide.rename(columns=LINECODE_FEATURE_MAP)
    print("ZIP COLUMNS NEW")
    print(zip_wide.columns)

    return zip_wide
