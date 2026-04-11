"""Converte relatorio.csv em data/dados.parquet. Gera CSV sintético se ausente."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "relatorio.csv"
PARQUET_PATH = DATA_DIR / "dados.parquet"

PEDRAS = ["Quartzo", "Ágata", "Ametista", "Topázio"]
TURMAS = ["A", "B", "C", "D"]


def generate_sample_csv(path: Path, n: int = 400, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        fase = int(rng.integers(1, 9))
        ano = int(rng.choice([2020, 2021, 2022, 2023]))
        turma = str(rng.choice(TURMAS))
        pedra = str(rng.choice(PEDRAS))
        base = 5.0 + fase * 0.35 + rng.normal(0, 0.8)
        rows.append(
            {
                "RA": f"RA{2020000 + i}",
                "Nome": f"Aluno {i + 1}",
                "Fase": fase,
                "Turma": turma,
                "Ano": ano,
                "INDE": float(np.clip(base + rng.normal(0, 0.5), 0, 10)),
                "IDA": float(np.clip(base - 0.3 + rng.normal(0, 0.6), 0, 10)),
                "IAN": float(np.clip(base - 0.1 + rng.normal(0, 0.5), 0, 10)),
                "IEG": float(np.clip(base - 0.2 + rng.normal(0, 0.7), 0, 10)),
                "IPV": float(np.clip(base + rng.normal(0, 0.6), 0, 10)),
                "Pedra": pedra,
            }
        )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    if not CSV_PATH.exists():
        print(f"CSV não encontrado. Gerando exemplo sintético em {CSV_PATH}")
        generate_sample_csv(CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    # normaliza tipos
    for col in ["Fase", "Ano"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ["INDE", "IDA", "IAN", "IEG", "IPV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Score de ML (mesmo modelo da aba de risco), quando models/modelo.joblib existir
    if "risco" in df.columns:
        df = df.drop(columns=["risco"])
    model_path = ROOT / "models" / "modelo.joblib"
    if model_path.exists():
        try:
            from passos_magico.ml.inference import load_model_bundle, predict_risk_probabilities

            bundle = load_model_bundle(model_path)
            df["risco"] = predict_risk_probabilities(bundle, df)
        except Exception as e:
            print(f"Aviso: não foi possível calcular coluna risco ({e}). Parquet sem ML.")
    else:
        print(
            "Modelo ausente; Parquet sem coluna risco. "
            "Execute scripts/train_model.py e rode este ETL novamente."
        )
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)
    cols = "com coluna risco (ML)" if "risco" in df.columns else "sem ML"
    print(f"Parquet gravado: {PARQUET_PATH} ({len(df)} linhas, {cols})")


if __name__ == "__main__":
    main()
    sys.exit(0)
