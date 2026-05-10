"""Testes da validação SQL, extração e execução DuckDB (perguntas complexas)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from passos_magico.data_engine.query import run_sql, validate_select_only
from passos_magico.llm.charts import figure_from_dataframe
from passos_magico.llm.insight_mode import infer_insight_response_mode
from passos_magico.llm.kpi_narration import kpi_narration_block
from passos_magico.llm.prompts import build_insight_user
from passos_magico.semantic.metadata import (
    GAMMA_CONTEXT_SECTION_PREFIX,
    context_without_gamma_narrative,
)
from passos_magico.llm.sql_parse import extract_sql_block, sql_passes_quick_validation


def _minimal_parquet_path() -> str:
    df = pd.DataFrame(
        {
            "RA": [f"RA{i}" for i in range(12)],
            "Nome": [f"Aluno {i}" for i in range(12)],
            "Fase": [6, 6, 6, 7, 7, 8, 8, 8, 6, 7, 8, 8],
            "Turma": list("AAABBBCCABCD"),
            "Ano": [2021] * 12,
            "INDE": [7.0] * 12,
            "IDA": [5.0, 5.5, 4.2, 6.0, 6.1, 7.0, 5.8, 6.2, 6.5, 5.9, 6.3, 6.4],
            "IAN": [7.0] * 12,
            "IEG": [7.0] * 12,
            "IPV": [7.0] * 12,
            "Pedra": ["Quartzo"] * 12,
            "risco": [0.3, 0.6, 0.2, 0.4, 0.5, 0.55, 0.25, 0.45, 0.35, 0.4, 0.5, 0.48],
        }
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.close()
    df.to_parquet(tmp.name, index=False)
    return tmp.name


class TestQuickValidation(unittest.TestCase):
    def test_from_dados_variants(self):
        self.assertTrue(sql_passes_quick_validation("SELECT 1 AS x FROM dados"))
        self.assertTrue(sql_passes_quick_validation('SELECT 1 FROM "dados"'))
        self.assertTrue(sql_passes_quick_validation("SELECT 1 FROM `dados`"))
        self.assertTrue(
            sql_passes_quick_validation(
                "SELECT t.x FROM (SELECT RA AS x FROM dados) AS t"
            )
        )

    def test_reject_garbage(self):
        self.assertFalse(sql_passes_quick_validation(None))
        self.assertFalse(sql_passes_quick_validation("SELECT 1"))
        self.assertFalse(sql_passes_quick_validation("SELECT (1"))


class TestValidateSelectWith(unittest.TestCase):
    def test_with_select_allowed(self):
        sql = """
        WITH agg AS (
          SELECT Fase, AVG(IDA) AS media_ida
          FROM dados WHERE Ano = 2021 AND Fase IN (6, 7, 8)
          GROUP BY Fase
        )
        SELECT * FROM agg
        """
        ok, err = validate_select_only(sql)
        self.assertTrue(ok, err)


class TestDuckdbComplexQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._parquet = _minimal_parquet_path()

    @classmethod
    def tearDownClass(cls):
        try:
            os.unlink(cls._parquet)
        except OSError:
            pass

    def test_turma_menor_ida_vs_media_fase_2021(self):
        """Pergunta complexa: menor média IDA por turma vs média da fase (mesmo ano)."""
        sql = """
        WITH por_turma AS (
          SELECT Turma, Fase, AVG(IDA) AS media_ida_turma
          FROM dados
          WHERE Ano = 2021 AND Fase IN (6, 7, 8)
          GROUP BY Turma, Fase
        ),
        por_fase AS (
          SELECT Fase, AVG(IDA) AS media_ida_fase
          FROM dados
          WHERE Ano = 2021 AND Fase IN (6, 7, 8)
          GROUP BY Fase
        )
        SELECT
          pt.Turma,
          pt.Fase,
          pt.media_ida_turma,
          pf.media_ida_fase,
          pt.media_ida_turma - pf.media_ida_fase AS diff_vs_fase
        FROM por_turma pt
        JOIN por_fase pf ON pt.Fase = pf.Fase
        ORDER BY pt.media_ida_turma ASC
        LIMIT 1
        """
        self.assertTrue(sql_passes_quick_validation(sql))
        ok, err = validate_select_only(sql)
        self.assertTrue(ok, err)
        df = run_sql(sql, parquet_path=Path(self._parquet))
        self.assertFalse(df.empty)
        self.assertIn("media_ida_turma", df.columns)

    def test_risco_fase8_turma(self):
        sql = """
        SELECT Turma, COUNT(*) AS n
        FROM dados
        WHERE Fase = 8 AND risco >= 0.5
        GROUP BY Turma
        ORDER BY n DESC
        """
        self.assertTrue(sql_passes_quick_validation(sql))
        df = run_sql(sql, parquet_path=Path(self._parquet))
        self.assertGreaterEqual(len(df), 0)


class TestExtractSql(unittest.TestCase):
    def test_block_with_quotes(self):
        raw = """Aqui está:
```sql
SELECT COUNT(*) FROM "dados" WHERE Fase = 8
```
"""
        s = extract_sql_block(raw)
        self.assertIsNotNone(s)
        self.assertIn("FROM", s.upper())

    def test_fenced_block_starts_with_with(self):
        raw = """Segue a consulta:
```sql
WITH t AS (
  SELECT Turma, AVG(IDA) AS m FROM dados WHERE Ano = 2021 GROUP BY Turma
)
SELECT * FROM t ORDER BY m ASC LIMIT 1
```
"""
        s = extract_sql_block(raw)
        self.assertIsNotNone(s)
        self.assertTrue(s.strip().upper().startswith("WITH"))
        self.assertIn("from dados", s.lower())


class TestChartKpiHeuristic(unittest.TestCase):
    def test_single_row_all_numeric_returns_kpi(self):
        df = pd.DataFrame({"total_defasados": [120]})
        _fig, kind = figure_from_dataframe(df, "auto")
        self.assertEqual(kind, "kpi")

    def test_single_row_cat_and_num_uses_bar(self):
        df = pd.DataFrame({"categoria": ["Defasados"], "n": [120]})
        _fig, kind = figure_from_dataframe(df, "auto")
        self.assertEqual(kind, "barras")

    def test_many_rows_ra_axis_skips_bar_chart(self):
        df = pd.DataFrame({"RA": [f"RA-{i}" for i in range(50)], "INDE": range(50)})
        _fig, kind = figure_from_dataframe(df, "auto")
        self.assertEqual(kind, "kpi")

    def test_few_rows_ra_axis_keeps_bar_chart(self):
        df = pd.DataFrame({"RA": [f"RA-{i}" for i in range(5)], "INDE": [7.0, 7.1, 7.2, 7.3, 7.4]})
        _fig, kind = figure_from_dataframe(df, "auto")
        self.assertEqual(kind, "barras")


class TestInsightResponseMode(unittest.TestCase):
    def test_kpi_from_chart_kind(self):
        df = pd.DataFrame({"x": range(20)})
        self.assertEqual(infer_insight_response_mode(df, "kpi"), "kpi")

    def test_kpi_single_row(self):
        df = pd.DataFrame({"t": [1]})
        self.assertEqual(infer_insight_response_mode(df, "barras"), "kpi")

    def test_analitico_multi_row(self):
        df = pd.DataFrame({"a": [1, 2]})
        self.assertEqual(infer_insight_response_mode(df, "barras"), "analitico")


class TestBuildInsightUserMode(unittest.TestCase):
    def test_includes_modo_resposta(self):
        u = build_insight_user(
            "Quantos?", "x\n1", "kpi", "", None, insight_mode="kpi"
        )
        self.assertIn("MODO_RESPOSTA: kpi", u)
        u2 = build_insight_user(
            "Compare fases", "a\n1", "barras", "", None, insight_mode="analitico"
        )
        self.assertIn("MODO_RESPOSTA: analitico", u2)

    def test_kpi_omits_long_institutional_context(self):
        """Evita que o relatório/Gamma entre no prompt de insight para totais de uma linha."""
        bloated = "### Resumo anual\n" + ("Lorem ipsum. " * 200)
        u = build_insight_user(
            "Quantos registos?",
            "total\n42",
            "kpi",
            bloated,
            None,
            insight_mode="kpi",
        )
        self.assertNotIn("Lorem ipsum", u)
        self.assertIn("ignore narrativas", u)

    def test_analitico_strips_gamma_section_from_context(self):
        toxic = "AUC 0.91 clusters modelo preditivo"
        merged = "### Dicionário\n- col: desc\n" + GAMMA_CONTEXT_SECTION_PREFIX + toxic
        u = build_insight_user(
            "Média por turma",
            "Turma  media\nA  5.1",
            "barras",
            merged,
            None,
            insight_mode="analitico",
        )
        self.assertNotIn("AUC 0.91", u)
        self.assertIn("Dicionário", u)
        self.assertIn("Gamma omitida", u)


class TestContextWithoutGamma(unittest.TestCase):
    def test_strip_leaves_dictionary(self):
        s = "DICT\n" + GAMMA_CONTEXT_SECTION_PREFIX + "LONG GAMMA"
        out = context_without_gamma_narrative(s)
        self.assertIn("DICT", out)
        self.assertNotIn("LONG GAMMA", out)


class TestKpiNarrationBlock(unittest.TestCase):
    def test_none_when_too_many_rows(self):
        df = pd.DataFrame({"x": range(20)})
        self.assertIsNone(kpi_narration_block(df))

    def test_single_row_numeric(self):
        df = pd.DataFrame({"Ano": [2022], "media_ida": [6.5]})
        s = kpi_narration_block(df)
        self.assertIsNotNone(s)
        self.assertIn("media_ida", s)
        self.assertRegex(s.replace(",", "."), r"6\.5")

    def test_single_row_count_integer_format(self):
        df = pd.DataFrame({"total": [234]})
        s = kpi_narration_block(df)
        self.assertIsNotNone(s)
        self.assertIn("234", s)
        self.assertNotIn("234.00", s)

    def test_grouped_means(self):
        df = pd.DataFrame({"Fase": [6, 7], "media_inde": [7.0, 7.5]})
        s = kpi_narration_block(df)
        self.assertIsNotNone(s)
        self.assertIn("média", s.lower())

    def test_degenerate_inde_all_zero_adds_alert(self):
        df = pd.DataFrame({"Turma": ["A", "B", "C"], "media_inde": [0.0, 0.0, 0.0]})
        s = kpi_narration_block(df)
        self.assertIsNotNone(s)
        self.assertIn("lacuna", (s or "").lower())

    def test_student_profile_skips_technical_means(self):
        """Um RA: não deve sugerir médias sobre cg/cf; booleanos como Sim/Não."""
        df = pd.DataFrame(
            {
                "RA": ["RA-910"] * 3,
                "Nome": ["Aluno-910"] * 3,
                "Ano": [2022, 2023, 2024],
                "INDE": [6.5, 7.0, 7.8],
                "cg": [280.0, 290.0, 295.0],
                "indicado_bolsa": [1, 1, 1],
            }
        )
        s = kpi_narration_block(df)
        self.assertIsNotNone(s)
        self.assertIn("RA-910", s or "")
        self.assertIn("Perfil (um aluno)", s or "")
        self.assertNotIn("280", s or "")
        self.assertNotIn("290", s or "")
        self.assertIn("indicado_bolsa", (s or "").lower())
        self.assertIn("Sim", s or "")

    def test_student_profile_shows_generic_name_hint(self):
        df = pd.DataFrame(
            {
                "RA": ["RA-910"],
                "Nome": ["Aluno-910"],
                "Ano": [2024],
                "INDE": [7.5],
            }
        )
        s = kpi_narration_block(df)
        self.assertIsNotNone(s)
        self.assertIn("genérico", (s or "").lower())


if __name__ == "__main__":
    unittest.main()
