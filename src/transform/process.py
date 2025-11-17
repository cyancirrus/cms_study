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


def zip_to_msa(
    zip_lat_long: pd.DataFrame,
    msa_dim: pd.DataFrame,
    msa_centroids: pd.DataFrame,
    msa_stats: pd.DataFrame,
):
    """
    Map ZIPs to nearest MSA, attach MSA dim and stats, output wide format ready for modeling.
    """
    # --- 1. Map ZIP to nearest MSA ---
    zip_coords = np.radians(
        zip_lat_long[["latitude", "longitude"]].values
    )
    msa_coords = np.radians(
        msa_centroids[["latitude", "longitude"]].values
    )

    tree = BallTree(msa_coords, metric="haversine")
    dist, idx = tree.query(zip_coords, k=1)

    zip_lat_long["cbsafp"] = msa_centroids.iloc[idx.flatten()][
        "cbsafp"
    ].values
    zip_lat_long["msa_title"] = msa_centroids.iloc[idx.flatten()][
        "msa_title"
    ].values
    zip_lat_long["state_abbreviation"] = msa_centroids.iloc[
        idx.flatten()
    ]["state_abbreviation"].values
    # Radians -> Killometers
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

    return zip_wide
