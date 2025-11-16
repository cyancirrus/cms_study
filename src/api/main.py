from fastapi import FastAPI, Query
from ..types.categories import HospitalType, ServiceCategory

app = FastAPI(title="Hospital Recommendation API")


@app.get("/recommend_hospital")
def retrieve_recommended_hospital(
    zip_code: int = Query(..., description="5-digit ZIP code of patient location"),
    hospital_type: HospitalType = Query(..., description="Type of hospital"),
    service_category: ServiceCategory = Query(
        ..., description="Type of service needed"
    ),
):
    """
    Returns a constant hospital recommendation for testing purposes.
    """
    return {
        "name": "Example Hospital",
        "zip_code": zip_code,
        "type": hospital_type.value,
        "services": [service_category.value],
        "message": "This is a constant response for scaffolding purposes",
    };


# --- Root endpoint ---
@app.get("/")
def root():
    return {"message": "Hospital Recommendation API is running!"}


if __name__ == "__main__":
    app.run()
