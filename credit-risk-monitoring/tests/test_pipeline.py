import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import FEATURES, RAW_CSV, features_from_frame  # noqa: E402
from metrics import ks_statistic, psi, psi_status  # noqa: E402


def test_psi_is_zero_for_same_distribution():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-6
    assert psi_status(psi(x, x)) == "stable"


def test_psi_flags_a_large_shift():
    rng = np.random.default_rng(1)
    assert psi_status(psi(rng.normal(size=5000), rng.normal(1.5, 1, 5000))) == "ALERT"


def test_ks_perfect_separation():
    assert ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_features_have_no_nulls_and_no_protected_attributes():
    raw = pd.read_csv(RAW_CSV, nrows=500)
    feats = features_from_frame(raw)
    assert feats[FEATURES].notna().all().all()
    assert not {"SEX", "MARRIAGE", "AGE"} & set(FEATURES)


def test_api_scores_an_account():
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    body = {"LIMIT_BAL": 50000, "EDUCATION": 2, "PAY": [2, 2, 0, 0, 0, 0],
            "BILL_AMT": [48000, 46000, 45000, 30000, 20000, 15000],
            "PAY_AMT": [0, 0, 1500, 1000, 1000, 800]}
    r = client.post("/score", json=body).json()
    assert 0 <= r["probability_of_default"] <= 1
    assert r["risk_band"] in {"low", "medium", "high"}
