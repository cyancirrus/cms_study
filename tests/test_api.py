import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.app_types.categories import (
    HospitalType,
    ServiceCategory,
)

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert (
        response.json()["message"]
        == "Hospital Recommendation API is running!"
    )


def test_favicon_endpoint():
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert response.json() == {}


@pytest.mark.parametrize(
    "zip_code,hospital_type,service_category",
    [
        (
            12345,
            HospitalType.acute,
            ServiceCategory.psychiatric,
        ),
        (
            67890,
            HospitalType.critical,
            ServiceCategory.general,
        ),
    ],
)
def test_recommend_hospital_endpoint(
    zip_code, hospital_type, service_category
):
    response = client.get(
        "/recommend_hospital",
        params={
            "zip_code": zip_code,
            "hospital_type": hospital_type.value,
            "service_category": service_category.value,
        },
    )
    # assert response.status_code == 200
    # data = response.json()
    # assert data["zip_code"] == zip_code
    # assert data["type"] == hospital_type.value
    # assert service_category.value in data["services"]
    # assert "message" in data
