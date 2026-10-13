"""Smoke tests run in CI after the pipeline."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "outputs/results.json").read_text())


def test_default_rate_is_reasonable():
    assert 0.05 < R["overall"]["default_rate"] < 0.5


def test_best_model_beats_baseline():
    assert R["models"][0]["roc_auc"] > 0.70


def test_all_sql_queries_ran():
    assert len(R["sql"]) == 11
    assert all(len(q["rows"]) > 0 for q in R["sql"].values())


def test_segments_present():
    assert {s["segment"] for s in R["segments"]} == {"Transactor", "Revolver", "At-Risk", "Dormant"}


def test_risk_ranking_is_monotonic_at_extremes():
    d = R["deciles"]
    assert d[0]["default_rate_pct"] > d[-1]["default_rate_pct"] * 3


def test_cleaned_data_has_no_nulls():
    assert pd.read_csv(ROOT / "data/processed/credit_clients_clean.csv").isna().sum().sum() == 0


def test_deliverables_exist():
    for f in ["dashboard/credit_risk_dashboard.html", "dashboard/dashboard_customer_level.csv",
              "reports/figures/08_risk_deciles.png", "README.md"]:
        assert (ROOT / f).stat().st_size > 0, f
