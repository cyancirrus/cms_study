from fastapi import FastAPI, Query, HTTPException
from datetime import datetime, timezone
from src.api.methods.query_recommendation import (
    query_recommendation_hospital,
)
from src.app_types.categories import (
    HospitalType,
    ServiceCategory,
)
from src.api.database.m_sqlite import SQLiteEngine

DATABASE = "source.db"
ENGINE = SQLiteEngine(DATABASE)
app = FastAPI(title="Hospital Recommendation API")


@app.get("/favicon.ico")
def favicon():
    return {}


@app.get("/recommend_hospital")
def retrieve_recommended_hospital(
    user_lat: float = Query(
        description="users latitude",
    ),
    user_long: float = Query(
        description="users longitude",
    ),
    hospital_type: HospitalType | None = Query(
        None, description="Type of hospital (optional)"
    ),
    service_category: ServiceCategory | None = Query(
        None, description="Type of service needed (optional)"
    ),
    distance_filter=Query(
        50, desciption="Hospital needs to be with this many kilometers"
    ),
    number_results: int = Query(
        10,
        ge=1,
        le=100,
        description="Max number of hospitals to return",
    ),
):
    """
    Recommend hospitals for a given ZIP code.
    Filters by hospital_type and service_category if provided.
    """

    hospital_type_str = (
        hospital_type.value if hospital_type is not None else None
    )
    service_type_str = (
        service_category.value if service_category is not None else None
    )

    results = query_recommendation_hospital(
        engine=ENGINE,
        user_lat=user_lat,
        user_long=user_long,
        distance_filter=distance_filter,
        number_results=number_results,
        hospital_type=hospital_type_str,
        service_type=service_type_str,
    )

    if not results:
        # Optional, but nice API ergonomics
        raise HTTPException(
            status_code=404,
            detail="No recommended hospitals found for the given filters.",
        )

    request_time = datetime.now(timezone.utc).isoformat()

    return {
        "request_time": request_time,
        "hospital_type": hospital_type_str,
        "service_category": service_type_str,
        "number_results": len(results),
        "recommendations": results,
    }


# --- Root endpoint --e
@app.get("/")
def root():
    return {"message": "Hospital Recommendation API is running!"}
