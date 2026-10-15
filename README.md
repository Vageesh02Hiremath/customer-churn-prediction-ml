# Customer Churn Prediction

A FastAPI customer churn dashboard with an individual customer assessment form, risk drivers, suggested retention actions, session metrics, and CSV export.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8001
```

Open `http://127.0.0.1:8001` in your browser. The API is available at `/api/health` and `/api/predict`.

## Connect a trained model

The current workspace contains `churn_threshold.pkl` (the numeric decision threshold `0.38`), but no trained estimator. Without an estimator the app uses a clearly labeled demo heuristic; do not use those demo scores for customer decisions.

Set `CHURN_MODEL_PATH` to a trusted pickle or joblib file containing a fitted scikit-learn pipeline. The pipeline must accept a pandas DataFrame with these columns: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, and `TotalCharges`. It must implement `predict_proba` (preferred) or `decision_function`. For `predict_proba`, the positive churn class should be labeled `Yes`, `1`, or `True`.

PowerShell example:

```powershell
$env:CHURN_MODEL_PATH = "C:\path\to\your\trained_pipeline.joblib"
python -m uvicorn main:app --reload
```

Optionally set `CHURN_THRESHOLD_PATH` to a pickle containing a numeric threshold between `0` and `1`. Restart the server after changing either environment variable.
"# customer-churn-prediction" 
