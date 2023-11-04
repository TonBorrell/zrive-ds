from fastapi.testclient import TestClient
from src.module_6.app import app

client = TestClient(app)


def test_predict_endpoint():
    """
    If we retrain the model this test will fail, because it will return a new prediction.
    We could come up with some ideas on how to treat this.
    For an MVP without a training pipeline, this works good
    """
    request_payload = {
        "user_id": "01284644755d542fa5a055ef50f8891da6c46e7cb96d604f1d611236ed2fd9f3f9ff8d079b09e29c5b6d7c8dbaafd2b3a54973949ad259e3d1dce5b8b469a1eb"
    }

    response = client.post("/predict", params=request_payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": 46.06071064475495}


def test_predict_wrong_user():
    request_payload = {"user_id": "non_existing_user_id"}

    response = client.post("/predict", params=request_payload)

    assert response.status_code == 404
    assert response.json() == {"detail": "User id not existing on DB"}
