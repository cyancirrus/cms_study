from sklearn.neighbors import BallTree
from typing import Final, Dict, List
import numpy as np
import pandas as pd

LINECODE_FEATURE_MAP: Final[Dict[str, str]] = {
    "1": "personal_income_k",
    "2": "population_density",
    "3": "per_capita_income",
}

TARGET_COLUMNS: Final[List[str]] = [
    "zip_code",
    "latitude",
    "longitude",
    "cbsafp_nearest",
    "msa_title_nearest",
    "distance_km_nearest",
    "msa_personal_income_k",
    "msa_population_density",
    "msa_per_capita_income",
]


TARGET_COLUMNS: Final[List[str]] = [
    "zip_code",
    "latitude",
    "longitude",
    "cbsafp_nearest",
    "msa_title_nearest",
    "distance_km_nearest",
    "msa_personal_income_k",
    "msa_population_density",
    "msa_per_capita_income",
]


def zip_to_msa_smoothed(
    zip_lat_long: pd.DataFrame,
    msa_centroids: pd.DataFrame,
    msa_stats: pd.DataFrame,
    k: int,
    sigma_km: float = 12.0,
) -> pd.DataFrame:
    # Ensure cbsafp is int
    for df_ in (msa_centroids, msa_stats):
        if "cbsafp" in df_.columns:
            df_["cbsafp"] = df_["cbsafp"].astype(int)

    # Keep only MSAs with stats
    msa_with_stats = msa_stats["cbsafp"].dropna().astype(int).unique()
    msa_centroids_filtered = msa_centroids[
        msa_centroids["cbsafp"].isin(msa_with_stats)
    ].copy()

    # Pivot stats wide
    if "linecode" in msa_stats.columns:
        msa_stats["feature_name"] = msa_stats["linecode"].replace(
            LINECODE_FEATURE_MAP
        )
    elif "feature_name" not in msa_stats.columns:
        raise ValueError(
            "msa_stats must have either 'linecode' or 'feature_name'."
        )

    stats_wide = msa_stats.pivot_table(
        index="cbsafp", columns="feature_name", values="metric"
    )

    # Filter centroids to MSAs with stats
    msa_ids_ordered = msa_centroids_filtered["cbsafp"].values
    feature_matrix = stats_wide.loc[msa_ids_ordered].values
    feature_names = stats_wide.columns

    # kNN
    zip_coords = np.radians(
        zip_lat_long[["latitude", "longitude"]].values
    )
    msa_coords = np.radians(
        msa_centroids_filtered[["latitude", "longitude"]].values
    )
    tree = BallTree(msa_coords, metric="haversine")
    dist_rad, idx = tree.query(zip_coords, k=k)
    dist_km = dist_rad * 6371.0

    # Gaussian weights
    weights = np.exp(-0.5 * (dist_km / sigma_km) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)

    # Smoothed features
    smoothed_features = np.einsum(
        "ij,ijk->ik", weights, feature_matrix[idx]
    )
    smoothed_df = pd.DataFrame(
        smoothed_features, columns=[f"msa_{c}" for c in feature_names]
    )

    # Nearest MSA info
    nearest_idx = idx[:, 0]
    nearest_msa = msa_centroids_filtered.iloc[nearest_idx].reset_index(
        drop=True
    )

    out = zip_lat_long.reset_index(drop=True).copy()
    out["cbsafp_nearest"] = nearest_msa["cbsafp"].values
    if "msa_title" in nearest_msa.columns:
        out["msa_title_nearest"] = nearest_msa["msa_title"].values
    out["distance_km_nearest"] = dist_km[:, 0]

    # Combine with smoothed features
    out = pd.concat([out, smoothed_df], axis=1)

    # Keep only target columns
    out = out[TARGET_COLUMNS]
    return out


# def zip_to_msa_smoothed(
#     zip_lat_long: pd.DataFrame,
#     msa_dim: pd.DataFrame,
#     msa_centroids: pd.DataFrame,
#     msa_stats: pd.DataFrame,
#     k: int,
#     sigma_km: float = 50.0,
# ) -> pd.DataFrame:
#     # Ensure consistent types
#     for df_ in (msa_centroids, msa_stats, msa_dim):
#         if "cbsafp" in df_.columns:
#             df_["cbsafp"] = df_["cbsafp"].astype(int)

#     # 1. Keep only MSAs that have stats
#     msa_with_stats = msa_stats["cbsafp"].dropna().astype(int).unique()
#     msa_centroids_filtered = msa_centroids[
#         msa_centroids["cbsafp"].isin(msa_with_stats)
#     ].copy()

#     # 2. Pivot stats to wide
#     msa_stats = msa_stats.copy()
#     if "linecode" in msa_stats.columns:
#         msa_stats["feature_name"] = msa_stats["linecode"].replace(
#             LINECODE_FEATURE_MAP
#         )
#     else:
#         if "feature_name" not in msa_stats.columns:
#             raise ValueError(
#                 "msa_stats must have either 'linecode' or 'feature_name'."
#             )

#     stats_wide = msa_stats.pivot_table(
#         index="cbsafp", columns="feature_name", values="metric"
#     )

#     # Now stats_wide has only MSAs with stats. Filter centroids to that same set:
#     stats_msa_ids = stats_wide.index.values
#     msa_centroids_filtered = msa_centroids_filtered[
#         msa_centroids_filtered["cbsafp"].isin(stats_msa_ids)
#     ].copy()

#     # Sanity: make sure we're not empty
#     if msa_centroids_filtered.empty:
#         raise ValueError(
#             "No MSAs left after intersecting centroids and stats."
#         )

#     # 3. Build feature matrix in centroid order
#     msa_ids_ordered = msa_centroids_filtered["cbsafp"].values
#     msa_features = stats_wide.loc[msa_ids_ordered]  # no NaN rows now
#     feature_names = msa_features.columns
#     feature_matrix = msa_features.values  # (n_msa, n_features)

#     # 4. kNN on filtered centroids
#     zip_coords = np.radians(
#         zip_lat_long[["latitude", "longitude"]].values
#     )
#     msa_coords = np.radians(
#         msa_centroids_filtered[["latitude", "longitude"]].values
#     )

#     tree = BallTree(msa_coords, metric="haversine")
#     dist_rad, idx = tree.query(zip_coords, k=k)
#     dist_km = dist_rad * 6371.0

#     # 5. Gaussian weights with guard
#     weights = np.exp(-0.5 * (dist_km / sigma_km) ** 2)
#     row_sums = weights.sum(axis=1, keepdims=True)
#     zero_mask = row_sums == 0
#     if np.any(zero_mask):
#         print(f"Warning: {zero_mask.sum()} ZIPs had zero weight sum.")
#         row_sums[zero_mask] = 1.0
#     weights = weights / row_sums

#     # 6. Smoothed features
#     smoothed_features = []
#     for i in range(idx.shape[0]):
#         neighbor_indices = idx[i]
#         w = weights[i]
#         neighbor_feats = feature_matrix[neighbor_indices]
#         smoothed = np.dot(w, neighbor_feats)
#         smoothed_features.append(smoothed)

#     smoothed_df = pd.DataFrame(smoothed_features, columns=feature_names)

#     # 7. Attach nearest MSA + dim for labels
#     nearest_idx = idx[:, 0]
#     nearest_msa = msa_centroids_filtered.iloc[nearest_idx].reset_index(
#         drop=True
#     )

#     out = zip_lat_long.reset_index(drop=True).copy()
#     out["cbsafp_nearest"] = nearest_msa["cbsafp"].values
#     if "msa_title" in nearest_msa.columns:
#         out["msa_title_nearest"] = nearest_msa["msa_title"].values
#     out["distance_km_nearest"] = dist_km[:, 0]

#     msa_dim["cbsafp"] = msa_dim["cbsafp"].astype(int)
#     out = out.merge(
#         msa_dim.add_suffix("_dim"),
#         left_on="cbsafp_nearest",
#         right_on="cbsafp_dim",
#         how="left",
#     )

#     # 8. Attach smoothed stats
#     out = pd.concat([out, smoothed_df.add_prefix("msa_")], axis=1)
#     out = out[TARGET_COLUMNS];
#     out = out.drop_duplicates()
#     assert isinstance(out, pd.DataFrame)
#     return out
