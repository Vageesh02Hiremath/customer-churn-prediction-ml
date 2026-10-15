import math
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
THRESHOLD_PATH = Path(os.getenv("CHURN_THRESHOLD_PATH", BASE_DIR / "churn_threshold.pkl"))
MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", BASE_DIR / "churn_model.pkl"))


def load_threshold() -> float:
	try:
		with THRESHOLD_PATH.open("rb") as threshold_file:
			value = float(pickle.load(threshold_file))
		return min(max(value, 0.0), 1.0)
	except (OSError, pickle.PickleError, TypeError, ValueError, EOFError):
		return 0.5


def load_model() -> Any | None:
	if not MODEL_PATH.is_file():
		return None
	try:
		if MODEL_PATH.suffix.lower() == ".joblib":
			import joblib

			return joblib.load(MODEL_PATH)
		with MODEL_PATH.open("rb") as model_file:
			return pickle.load(model_file)
	except Exception as exc:
		raise RuntimeError(f"Could not load churn model at {MODEL_PATH}: {exc}") from exc


THRESHOLD = load_threshold()
MODEL = load_model()
app = FastAPI(title="Retention Intelligence", version="1.0.0")


class CustomerFeatures(BaseModel):
	gender: str
	SeniorCitizen: int
	Partner: str
	Dependents: str
	tenure: int
	PhoneService: str
	MultipleLines: str
	InternetService: str
	OnlineSecurity: str
	OnlineBackup: str
	DeviceProtection: str
	TechSupport: str
	StreamingTV: str
	StreamingMovies: str
	Contract: str
	PaperlessBilling: str
	PaymentMethod: str
	MonthlyCharges: float
	TotalCharges: float


def demo_probability(customer: CustomerFeatures) -> float:
	"""Return a transparent demo estimate until a trained model is configured."""
	score = -2.0
	score += 0.65 if customer.tenure <= 6 else 0.3 if customer.tenure <= 12 else 0.0
	score += {"Month-to-month": 0.95, "One year": 0.05, "Two year": -0.55}.get(customer.Contract, 0.0)
	score += 0.3 if customer.InternetService == "Fiber optic" else 0.0
	score += 0.35 if customer.OnlineSecurity == "No" else 0.0
	score += 0.25 if customer.TechSupport == "No" else 0.0
	score += 0.25 if customer.PaymentMethod == "Electronic check" else 0.0
	score += 0.25 if customer.MonthlyCharges >= 80 else 0.0
	score += 0.1 if customer.PaperlessBilling == "Yes" else 0.0
	score += 0.1 if customer.Partner == "No" else 0.0
	score += 0.1 if customer.SeniorCitizen == 1 else 0.0
	return 1 / (1 + math.exp(-score))


def model_probability(customer: CustomerFeatures) -> float:
	if MODEL is None:
		return demo_probability(customer)

	features = pd.DataFrame([customer.model_dump()])
	if hasattr(MODEL, "predict_proba"):
		probabilities = MODEL.predict_proba(features)[0]
		classes = list(getattr(MODEL, "classes_", range(len(probabilities))))
		positive_index = next(
			(index for index, label in enumerate(classes) if str(label).strip().lower() in {"yes", "1", "true"}),
			len(probabilities) - 1,
		)
		return float(probabilities[positive_index])
	if hasattr(MODEL, "decision_function"):
		decision = float(MODEL.decision_function(features)[0])
		return 1 / (1 + math.exp(-max(min(decision, 500), -500)))
	raise HTTPException(
		status_code=500,
		detail="The configured model must provide predict_proba or decision_function.",
	)


def customer_insights(customer: CustomerFeatures, probability: float) -> tuple[list[str], list[str]]:
	drivers: list[str] = []
	actions: list[str] = []
	if customer.Contract == "Month-to-month":
		drivers.append("Month-to-month contract")
		actions.append("Offer a renewal incentive for a 12-month plan")
	if customer.tenure <= 12:
		drivers.append("Early tenure")
		actions.append("Add a proactive check-in during the first year")
	if customer.OnlineSecurity == "No":
		drivers.append("No online security")
		actions.append("Introduce an internet security add-on")
	if customer.TechSupport == "No":
		drivers.append("No tech support")
		actions.append("Offer a support plan or guided setup")
	if customer.PaymentMethod == "Electronic check":
		drivers.append("Electronic check billing")
		actions.append("Make automatic bank or card payments easy to switch to")
	if not drivers:
		drivers.append("No major profile risk signals")
		actions.append("Maintain regular engagement and monitor the next billing cycle")
	if probability < THRESHOLD:
		actions.insert(0, "Keep the current plan; monitor at the next renewal")
	return drivers[:3], actions[:3]


@app.get("/")
def index() -> FileResponse:
	return FileResponse(BASE_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
	return {
		"status": "ok",
		"mode": "model" if MODEL is not None else "demo",
		"threshold": THRESHOLD,
	}


@app.post("/api/predict")
def predict(customer: CustomerFeatures) -> dict[str, Any]:
	try:
		probability = min(max(model_probability(customer), 0.0), 1.0)
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc
	drivers, actions = customer_insights(customer, probability)
	return {
		"probability": probability,
		"risk": "High" if probability >= THRESHOLD else "Low",
		"threshold": THRESHOLD,
		"mode": "model" if MODEL is not None else "demo",
		"drivers": drivers,
		"actions": actions,
	}
