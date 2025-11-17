# from transform.process import zip_to_msa, zip_to_msa_smoothed
from transform.process import zip_to_msa_smoothed
from database.bridge import EngineProtocol
from tables import CmsSchema


def transform_msa_for_demographics(engine: EngineProtocol):
    df_zip_lat_long = engine.read(CmsSchema.zip_lat_long)
    df_msa_dim = engine.read(CmsSchema.msa_dim)
    df_msa_centroids = engine.read(CmsSchema.msa_centroid)
    df_msa_stats = engine.read(CmsSchema.msa_statistics)
    result = zip_to_msa_smoothed(
        df_zip_lat_long, df_msa_dim, df_msa_centroids, df_msa_stats, 3
    )
    engine.write(result, CmsSchema.zip_demographics)
