"""Batch scoring + drift monitoring.

Simulates 6 monthly "production" batches drawn from customers the model never saw.
Months 1-3 look like normal business. Months 4-6 add a gradual stress scenario
(more customers paying late, higher balances, smaller repayments) - the kind of
shift a card portfolio sees in a downturn. Each month the job:
  1. builds features with the same SQL used in training
  2. scores every account with the saved champion model
  3. computes PSI for the score and every feature vs the training population
  4. checks AUC once outcomes (default labels) arrive
  5. writes a log row and raises alerts when thresholds are crossed
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import FEATURES, RAW_CSV, ROOT, features_from_frame
from metrics import psi, psi_status

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
BATCH_SIZE = 2000
AUC_FLOOR = 0.75  # alert if discrimination falls below this
STRESS = {4: 0.10, 5: 0.25, 6: 0.40}  # share of accounts hit by stress, by month


def make_batch(holdout_raw, month, rng):
    batch = holdout_raw.sample(BATCH_SIZE, replace=True, random_state=rng.integers(1e9)).copy()
    batch["ID"] = np.arange(len(batch)) + month * 100_000
    share = STRESS.get(month, 0.0)
    if share:
        hit = rng.random(len(batch)) < share
        pay_cols = ["PAY_1", "PAY_2", "PAY_3"]
        batch.loc[hit, pay_cols] = np.clip(batch.loc[hit, pay_cols].clip(lower=0) + 2, 0, 8)
        for c in ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]:
            batch.loc[hit, c] = (batch.loc[hit, c] * 1.3).round()
        for c in ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]:
            batch.loc[hit, c] = (batch.loc[hit, c] * 0.4).round()
        # stressed customers default more often in reality, but not all of them
        flip = hit & (rng.random(len(batch)) < 0.35)
        batch.loc[flip, "dpnm"] = 1
    return batch


def run():
    REPORTS_DIR.mkdir(exist_ok=True)
    model = joblib.load(MODELS_DIR / "champion.joblib")
    card = json.loads((MODELS_DIR / "model_card.json").read_text())
    ref = pd.read_csv(MODELS_DIR / "reference_population.csv")
    holdout_ids = pd.read_csv(MODELS_DIR / "holdout_ids.csv")["account_id"]
    raw = pd.read_csv(RAW_CSV)
    holdout_raw = raw[raw["ID"].isin(holdout_ids)]

    rng = np.random.default_rng(7)
    log, alerts = [], []
    for month in range(1, 7):
        batch = make_batch(holdout_raw, month, rng)
        feats = features_from_frame(batch)
        scores = model.predict_proba(feats[FEATURES])[:, 1]

        feature_psi = {f: psi(ref[f], feats[f]) for f in FEATURES}
        score_psi = psi(ref["score"], scores)
        auc = roc_auc_score(feats["defaulted"], scores)
        worst_feat = max(feature_psi, key=feature_psi.get)

        row = {
            "month": month, "model_version": card["version"], "accounts": len(feats),
            "mean_pd": round(float(scores.mean()), 4),
            "actual_default_rate": round(float(feats["defaulted"].mean()), 4),
            "score_psi": round(score_psi, 4), "score_status": psi_status(score_psi),
            "worst_feature": worst_feat, "worst_feature_psi": round(feature_psi[worst_feat], 4),
            "features_in_alert": sum(v >= 0.25 for v in feature_psi.values()),
            "auc": round(auc, 4),
        }
        log.append(row)
        if score_psi >= 0.25:
            alerts.append(f"Month {month}: score PSI {score_psi:.3f} - population has shifted; review cut-offs.")
        elif score_psi >= 0.10:
            alerts.append(f"Month {month}: score PSI {score_psi:.3f} - early warning, keep watching.")
        for f, v in feature_psi.items():
            if v >= 0.25:
                alerts.append(f"Month {month}: feature '{f}' PSI {v:.3f} - investigate data source / customer behaviour.")
        if auc < AUC_FLOOR:
            alerts.append(f"Month {month}: AUC {auc:.3f} below floor {AUC_FLOOR} - trigger retraining review.")

    log_df = pd.DataFrame(log)
    log_df.to_csv(REPORTS_DIR / "monitoring_log.csv", index=False)
    (REPORTS_DIR / "alerts.md").write_text(
        "# Monitoring alerts\n\n" + ("\n".join(f"- {a}" for a in alerts) or "No alerts.") + "\n")
    print(log_df.to_string(index=False))
    print(f"\n{len(alerts)} alerts written to reports/alerts.md")
    return log_df


if __name__ == "__main__":
    run()
