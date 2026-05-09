"""Cache e fingerprint do parecer dos dashboards."""

from __future__ import annotations

import json

import pandas as pd

from passos_magico.ui import dashboard_feedback as fb


def test_fingerprint_stable_for_same_frame(monkeypatch, tmp_path) -> None:
    df = pd.DataFrame(
        {
            "RA": ["a", "b"],
            "Ano": [2022, 2023],
            "INDE": [6.0, 7.0],
            "IAN": [7.0, 8.0],
            "risco": [0.2, 0.5],
        }
    )

    fake = tmp_path / "fake.parquet"
    fake.write_bytes(b"x")

    def _gp():
        return fake

    monkeypatch.setattr(fb, "get_parquet_path", _gp)
    fp1 = fb.compute_dashboard_fingerprint(df)
    fp2 = fb.compute_dashboard_fingerprint(df.copy())
    assert fp1 == fp2


def test_fingerprint_changes_when_rows_change(monkeypatch, tmp_path) -> None:
    fake = tmp_path / "fake.parquet"
    fake.write_bytes(b"x")

    def _gp():
        return fake

    monkeypatch.setattr(fb, "get_parquet_path", _gp)
    df1 = pd.DataFrame({"RA": ["a"], "INDE": [5.0]})
    df2 = pd.DataFrame({"RA": ["a", "b"], "INDE": [5.0, 6.0]})
    assert fb.compute_dashboard_fingerprint(df1) != fb.compute_dashboard_fingerprint(df2)


def test_roundtrip_meta(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fb, "FEEDBACK_PATH", tmp_path / "fb.txt")
    fp = "abc123"
    body = "## Olá\n\nTexto."
    fb.save_feedback(fp, body)
    assert fb.load_cached_feedback(fp) == body
    raw = fb.FEEDBACK_PATH.read_text(encoding="utf-8")
    assert raw.startswith(fb.META_PREFIX)
    meta = json.loads(raw[len(fb.META_PREFIX) : raw.index("\n\n")])
    assert meta["fingerprint"] == fp
