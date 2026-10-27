
# ===== 1. Import the libraries =====

import pandas as pd  # Read and manipulate the CSV
import mlflow  # Main MLflow module
import mlflow.sklearn  # MLflow integration for scikit-learn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Four metrics to evaluate the model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ===== 2. Load the data (only once) =====

df = pd.read_csv("churn_data.csv")  # Table of 500 customers

y = df["churn"]  # Target: 1 = leaves, 0 = stays


# ===== 3. Function: prepare the data for a list of columns =====

def prepare(columns):
    # Convert categorical text into dummy variables
    # (if 'abonnement' is included)
    X = pd.get_dummies(df[columns], drop_first=True)

    # Split the data: 80% training / 20% testing
    # random_state=42 ensures the same split every time
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


# ===== 4. Function: run ONE trial and record it in MLflow =====

def run_trial(name, n_estimators, max_depth, columns):

    # Prepare the training and testing sets
    X_train, X_test, y_train, y_test = prepare(columns)

    # Open an MLflow run named after the trial
    with mlflow.start_run(run_name=name):

        # ----- Create and train the model -----

        model = RandomForestClassifier(
            n_estimators=n_estimators,  # Number of trees
            max_depth=max_depth,        # Maximum depth of the trees
            random_state=42             # Ensure reproducibility
        )

        model.fit(X_train, y_train)  # Train the model

        # ----- Make predictions -----

        pred = model.predict(X_test)

        # ----- Log parameters -----

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(columns))

        # ----- Calculate and log metrics -----

        accuracy = accuracy_score(y_test, pred)

        precision = precision_score(
            y_test,
            pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            pred,
            zero_division=0
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # ----- Save the model as an artifact -----

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle"
        )

        # ----- Display the results -----

        print(
            name,
            "-> accuracy:", round(accuracy, 3),
            "| f1:", round(f1, 3)
        )


# ===== 5. Main program =====

# Set the MLflow experiment
mlflow.set_experiment("churn-prediction")

# List of all available features
ALL = [
    "age",
    "salaire",
    "abonnement",
    "anciennete",
    "consommation"
]


# ----- Trial 1: 100 trees, depth 5, all features -----

run_trial(
    "trial-1",
    100,
    5,
    ALL
)


# ----- Trial 2: 50 trees, depth 3, all features -----

run_trial(
    "trial-2",
    50,
    3,
    ALL
)


# ----- Trial 3: without the salary feature -----

run_trial(
    "trial-3",
    100,
    5,
    [
        "age",
        "abonnement",
        "anciennete",
        "consommation"
    ]
)
