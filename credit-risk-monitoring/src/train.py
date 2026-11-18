"""Train and compare default-risk models, pick a champion, save it with a model card."""
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import FEATURES, ROOT, TARGET, build_features, load_raw_to_sqlite
from metrics import gini, ks_statistic

SEED = 42
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def candidate_models():
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=50, n_jobs=-1, random_state=SEED),
        "xgboost": XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, eval_metric="auc", random_state=SEED),
    }


def score_bands(y, score, n=10):
    """Split customers into 10 risk bands (1 = safest) and show default rate in each."""
    df = pd.DataFrame({"y": y, "score": score})
    df["band"] = pd.qcut(df["score"].rank(method="first"), n, labels=range(1, n + 1))
    out = df.groupby("band", observed=True).agg(
        accounts=("y", "size"), defaults=("y", "sum"),
        min_pd=("score", "min"), max_pd=("score", "max"))
    out.index = out.index.astype(int)
    out["default_rate"] = (out["defaults"] / out["accounts"]).round(4)
    out["share_of_all_defaults"] = (out["defaults"] / out["defaults"].sum()).round(4)
    return out


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    load_raw_to_sqlite()
    df = build_features()
    X, y = df[FEATURES], df[TARGET]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    results, fitted = [], {}
    for name, model in candidate_models().items():
        model.fit(X_tr, y_tr)
        p_val = model.predict_proba(X_val)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
        results.append({
            "model": name,
            "val_auc": round(roc_auc_score(y_val, p_val), 4),
            "test_auc": round(roc_auc_score(y_te, p_te), 4),
            "test_gini": round(gini(y_te, p_te), 4),
            "test_ks": round(ks_statistic(y_te, p_te), 4),
        })
        fitted[name] = model

    board = pd.DataFrame(results).sort_values("val_auc", ascending=False)
    board.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(board.to_string(index=False))

    champ_name = board.iloc[0]["model"]
    champ = fitted[champ_name]
    p_te = champ.predict_proba(X_te)[:, 1]

    bands = score_bands(y_te.values, p_te)
    bands.to_csv(REPORTS_DIR / "score_bands.csv")
    print("\nRisk bands on held-out test set (champion):\n", bands)

    top20 = bands.loc[[9, 10], "share_of_all_defaults"].sum()
    approve_80 = bands.loc[bands.index <= 8]
    bad_rate_approved = approve_80["defaults"].sum() / approve_80["accounts"].sum()
    policy = {
        "portfolio_default_rate": round(float(y_te.mean()), 4),
        "riskiest_20pct_capture_of_defaults": round(float(top20), 4),
        "bad_rate_if_riskiest_20pct_declined": round(float(bad_rate_approved), 4),
    }
    print("\nPolicy view:", policy)

    # feature importance for explainability
    if champ_name == "logistic_regression":
        imp = champ[-1].coef_[0]
    else:
        imp = champ.feature_importances_
    importance = pd.Series(imp, index=FEATURES).sort_values(key=np.abs, ascending=False)
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", header=["importance"])

    version = f"{champ_name}-{date.today():%Y%m%d}"
    joblib.dump(champ, MODELS_DIR / "champion.joblib")
    # reference data for drift monitoring = training population + its scores
    ref = X_tr.copy()
    ref["score"] = champ.predict_proba(X_tr)[:, 1]
    ref.to_csv(MODELS_DIR / "reference_population.csv", index=False)
    df.loc[X_te.index, ["account_id"]].to_csv(MODELS_DIR / "holdout_ids.csv", index=False)

    card = {
        "version": version, "champion": champ_name, "features": FEATURES,
        "excluded_attributes": ["SEX", "MARRIAGE", "AGE"],
        "train_rows": len(X_tr), "val_rows": len(X_val), "test_rows": len(X_te),
        "leaderboard": results, "policy_view": policy,
        "top_features": importance.head(5).round(4).to_dict(),
    }
    (MODELS_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"\nsaved champion {version}")


if __name__ == "__main__":
    main()
