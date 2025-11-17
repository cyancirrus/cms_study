import pandas as pd
import numpy as np
from typing import Final, Dict
from sklearn.neighbors import BallTree

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

LINECODE_FEATURE_MAP: Final[Dict[int, str]] = {
    1: "personal_income_k",
    2: "population",
    3: "per_capita_income",
}


def zip_to_msa(zip_df, msa_centroids_df, msa_dim_df, msa_stats_df):
    """
    Map ZIPs to nearest MSA, attach MSA dim and stats, output wide format ready for modeling.
    """
    # --- 1. Map ZIP to nearest MSA ---
    zip_coords = np.radians(zip_df[["latitude", "longitude"]].values)
    msa_coords = np.radians(msa_centroids_df[["latitude", "longitude"]].values)

    tree = BallTree(msa_coords, metric="haversine")
    dist, idx = tree.query(zip_coords, k=1)

    zip_df["cbsafp"] = msa_centroids_df.iloc[idx.flatten()]["cbsafp"].values
    zip_df["msa_title"] = msa_centroids_df.iloc[idx.flatten()]["msa_title"].values
    zip_df["state_abbreviation"] = msa_centroids_df.iloc[idx.flatten()][
        "state_abbreviation"
    ].values
    zip_df["distance_km"] = dist.flatten() * 6371  # rad → km

    # --- 2. Merge MSA dim ---
    zip_df = zip_df.merge(
        msa_dim_df[["cbsafp", "cbsa_title", "state_abbreviation"]],
        on=["cbsafp", "state_abbreviation"],
        how="left",
    )

    # --- 3. Map linecodes to descriptive feature names ---
    msa_stats_df["feature_name"] = msa_stats_df["linecode"].map(LINECODE_FEATURE_MAP)

    # --- 4. Pivot stats to wide ---
    stats_wide = msa_stats_df.pivot_table(
        index="cbsafp", columns="feature_name", values="metric"
    ).reset_index()

    # --- 5. Merge wide stats into ZIPs ---
    zip_wide = zip_df.merge(stats_wide, on="cbsafp", how="left")

    return zip_wide
