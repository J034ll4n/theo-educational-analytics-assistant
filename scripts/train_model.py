"""Treina RandomForest e grava models/modelo.joblib."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from passos_magico.data_engine.loader import get_parquet_path, load_dados_df  # noqa: E402
from passos_magico.ml.features import build_xy  # noqa: E402

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "modelo.joblib"


def main() -> None:
    path = get_parquet_path()
    if not path.exists():
        raise SystemExit(f"Parquet ausente: {path}. Execute scripts/etl.py ou defina PASSOS_PARQUET.")
    df = load_dados_df(path)
    X, y = build_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=160,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred, digits=3))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "feature_order": list(X.columns)}, MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
