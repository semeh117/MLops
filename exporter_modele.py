# ===== 1. Import the libraries =====
import mlflow  # the main MLflow module

import mlflow.sklearn  # to load a scikit-learn model from the Registry
#===== 2. Connect to the same database as Lab 4 =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

#==== 3. Load the champion model from the Registry =====
model = mlflow.sklearn.load_model("models:/churn-model@champion")

#==== 4. write it to an ordinary folder, so that it can be used by the API =====
# serialization_format="cloudpickle": same format as in Labs 3 and 4
# (the default format of MLflow 3 is refused at loading time).
# This folder will be copied into the Docker image: the image will no longer need the Registry.

mlflow.sklearn.save_model(
    model,"model", serialization_format="cloudpickle")
print("Champion model saved to the 'model' folder, ready for the API")