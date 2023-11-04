import uvicorn
from src.module_6.exceptions import PredictionException, UserNotFoundException
from src.module_6.logger import Logger
from fastapi import FastAPI, HTTPException
from src.module_6.basket_model import basket_model, feature_store

# Create an instance of FastAPI
app = FastAPI()

logger = Logger()

# We instance the classes outside our endpoint so we have the data loaded in batch
# And no need to recalculate the data for each prediction
feature_store = feature_store.FeatureStore(logger)
basket_model = basket_model.BasketModel(logger)


@app.get("/")
def read_root():
    logger.info("GET /")
    logger.info("200 RESPONSE")
    return {"message": "Hello, World!"}


@app.get("/status")
def get_status():
    logger.info("Status request recieved")
    return {"message": "Status OK"}


@app.post("/predict")
def predict(user_id: str):
    logger.info(f"Predict request recieved for user {user_id}")
    try:
        user_features_df = feature_store.get_features(user_id)
        user_features_array = user_features_df.to_numpy()
        prediction = basket_model.predict(user_features_array)
        logger.info(f"Predict request for user {user_id} with prediction {prediction}")
        return {"prediction": prediction.mean()}
    except UserNotFoundException:
        raise HTTPException(404, "User id not existing on DB")
    except PredictionException:
        raise HTTPException(404, "Error in the Prediction")


# This block allows you to run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
