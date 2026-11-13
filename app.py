# ===== 1. Import the libraries =====
from typing import Literal  # to limit the possible values of a field
import mlflow  # load the model from the Registry
import mlflow.sklearn
import pandas as pd  # build the table expected by the model
from fastapi import FastAPI  # the framework of the API
from pydantic import BaseModel  # describe and validate the received data


# ===== 2. Create the application =====
app = FastAPI(title="Churn API")  # the title is displayed in the /docs page


# ===== 3. Load the champion ONLY ONCE, at startup =====
# (reloading it at every request would make the API very slow)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")


# ===== 4. Describe the expected data (validated automatically) =====
class Customer(BaseModel):
    age: int  # whole number
    salaire: float  # decimal number (salary)
    abonnement: Literal["Basique", "Standard", "Premium"]  # 3 possible values (subscription)
    anciennete: float  # seniority / tenure
    consommation: float  # usage


# ===== 5. Route /health: check that the API is alive =====
@app.get("/health")  # GET: we read, we send nothing
def health():
    return {"status": "ok"}


# ===== 6. Route /predict: receive a customer, return the prediction =====
@app.post("/predict")  # POST: we send a customer
def predict(customer: Customer):  # FastAPI checks the received JSON with Customer

    # a) turn the received customer into a one-row table
    row = pd.DataFrame([customer.model_dump()])

    # b) turn the text "abonnement" into 0/1 columns, as during training
    row = pd.get_dummies(row)

    # c) keep the model's columns, in the right order; missing ones are 0
    row = row.reindex(columns=model.feature_names_in_, fill_value=0)

    # d) predict: 1 = leaves, 0 = stays
    result = int(model.predict(row)[0])

    return {
        "churn": result,
        "message": "The customer will leave" if result == 1 else "The customer stays"
    }