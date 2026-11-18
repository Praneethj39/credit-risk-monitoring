"""Small REST API that scores one card account. Run: uvicorn api:app --reload (from src/)."""
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from features import FEATURES, ROOT, features_from_frame

MODELS_DIR = ROOT / "models"
model = joblib.load(MODELS_DIR / "champion.joblib")
card = json.loads((MODELS_DIR / "model_card.json").read_text())

app = FastAPI(title="Card default risk scorer", version=card["version"])


class Account(BaseModel):
    LIMIT_BAL: float = Field(gt=0)
    EDUCATION: int = 2
    PAY: list[int] = Field(min_length=6, max_length=6, description="repayment status, latest month first")
    BILL_AMT: list[float] = Field(min_length=6, max_length=6)
    PAY_AMT: list[float] = Field(min_length=6, max_length=6)


def risk_band(pd_value):
    return "low" if pd_value < 0.15 else "medium" if pd_value < 0.40 else "high"


@app.get("/health")
def health():
    return {"status": "ok", "model_version": card["version"]}


@app.post("/score")
def score(acc: Account):
    row = {"ID": 0, "LIMIT_BAL": acc.LIMIT_BAL, "SEX": 0, "EDUCATION": acc.EDUCATION,
           "MARRIAGE": 0, "AGE": 0, "dpnm": 0}
    row.update({f"PAY_{i+1}": v for i, v in enumerate(acc.PAY)})
    row.update({f"BILL_AMT{i+1}": v for i, v in enumerate(acc.BILL_AMT)})
    row.update({f"PAY_AMT{i+1}": v for i, v in enumerate(acc.PAY_AMT)})
    feats = features_from_frame(pd.DataFrame([row]))
    p = float(model.predict_proba(feats[FEATURES])[:, 1][0])
    return {"probability_of_default": round(p, 4), "risk_band": risk_band(p),
            "model_version": card["version"]}
