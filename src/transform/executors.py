# from transform.process import zip_to_msa, zip_to_msa_smoothed
from transform.zip_msa import zip_to_msa_smoothed
from transform.facility_zip_code import create_facility_zip_code
from database.bridge import EngineProtocol
from tables import CmsSchema


def transform_msa_for_demographics(engine: EngineProtocol):
    df_zip_lat_long = engine.read(CmsSchema.zip_lat_long)
    df_msa_centroids = engine.read(CmsSchema.msa_centroid)
    df_msa_stats = engine.read(CmsSchema.msa_statistics)
    result = zip_to_msa_smoothed(
        df_zip_lat_long, df_msa_centroids, df_msa_stats, 4
    )
    engine.write(result, CmsSchema.zip_demographics)


def transform_facility_zip_code(engine: EngineProtocol):
    hospital_general_information = engine.read(
        CmsSchema.hospital_general_information
    )
    result = create_facility_zip_code(hospital_general_information)
    engine.write(result, CmsSchema.facility_zip_code)
