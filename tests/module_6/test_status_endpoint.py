from fastapi.testclient import TestClient
from src.module_6.app import app

client = TestClient(app)


def test_status_endpoint():
    response = client.get("/status")

    assert response.status_code == 200
    assert response.json() == {"message": "Status OK"}
