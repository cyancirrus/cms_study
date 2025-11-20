from typing import Optional
import pandas as pd
from src.database.bridge import EngineProtocol


def query_recommendation_hospital(
    engine: EngineProtocol,
    user_lat: float,
    user_long: float,
    distance_filter: float,
    hospital_type: Optional[str] = None,
    service_type: Optional[str] = None,
    number_results: int = 50,
) -> list[dict]:

    # Base SQL
    sql = f"""
    SELECT
        recommendation_year,
        facility_id,
        facility_name,
        hospital_type,
        address,
        zip_code,
        state,
        service_type,
        latitude,
        longitude,
        score,
        sqrt((latitude - {user_lat})*(latitude - {user_lat}) +
             (longitude - {user_long})*(longitude - {user_long})) AS distance
    FROM recommendation_hospital
    """

    # Optional filters
    where_clauses = []
    if hospital_type:
        where_clauses.append(f"hospital_type = '{hospital_type}'")
    if service_type:
        where_clauses.append(f"service_type = '{service_type}'")
    if distance_filter is not None:
        where_clauses.append(
            f"sqrt((latitude - {user_lat})*(latitude - {user_lat}) + "
            f"(longitude - {user_long})*(longitude - {user_long})) <= {distance_filter}"
        )

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    # Order + limit
    sql += f" ORDER BY score DESC LIMIT {number_results}"

    # Execute query
    df: pd.DataFrame = engine.exec(sql)

    # Convert to list of dicts for API
    return df.to_dict(orient="records")
