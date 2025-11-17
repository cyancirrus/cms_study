from transform.process import zip_to_msa
from database.bridge import EngineProtocol
from tables import CmsSchema


def transform_msa_for_demographics(engine:EngineProtocol):
    df_zip_lat_long= engine.read(CmsSchema.zip_lat_long);
    df_msa_dim = engine.read(CmsSchema.zip_lat_long);
    df_msa_centroids = engine.read(CmsSchema.msa_centoid);
    df_msa_stats = engine.read(CmsSchema.msa_statistics)
    result = zip_to_msa(
        df_zip_lat_long,
        df_msa_dim ,
        df_msa_centroids ,
        df_msa_stats ,
    )
    engine.write(result, CmsSchema.zip_demographics)
