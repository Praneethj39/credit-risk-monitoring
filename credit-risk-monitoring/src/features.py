"""Load raw data into SQLite and build model features with SQL."""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "credit_card_default.csv"
DB_PATH = ROOT / "data" / "credit.db"
SQL_PATH = Path(__file__).with_name("features.sql")

FEATURES = [
    "credit_limit", "education", "status_last_month", "months_late_6m",
    "worst_delay_6m", "utilisation", "avg_utilisation_6m", "balance_trend_6m",
    "repayment_ratio_6m", "last_payment_ratio", "zero_payment_months",
]
TARGET = "defaulted"


def load_raw_to_sqlite(csv_path=RAW_CSV, db_path=DB_PATH):
    raw = pd.read_csv(csv_path)
    with sqlite3.connect(db_path) as con:
        raw.to_sql("accounts", con, if_exists="replace", index=False)
    return len(raw)


def build_features(db_path=DB_PATH, table="accounts"):
    sql = SQL_PATH.read_text().replace("FROM accounts", f"FROM {table}")
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(sql, con)
    return df


def features_from_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Same SQL logic applied to an in-memory batch (used by monitoring + API)."""
    with sqlite3.connect(":memory:") as con:
        raw.to_sql("accounts", con, index=False)
        return pd.read_sql_query(SQL_PATH.read_text(), con)


if __name__ == "__main__":
    n = load_raw_to_sqlite()
    df = build_features()
    print(f"loaded {n} accounts, built {len(FEATURES)} features")
    print(df[FEATURES + [TARGET]].describe().T.round(3))
