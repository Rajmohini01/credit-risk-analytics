"""Builds the self-contained interactive HTML dashboard from dashboard/ CSVs."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "dashboard/dashboard_customer_level.csv")
df["ub"] = pd.cut(df.utilization_avg, [-1, .2, .4, .6, .8, 99], labels=["0-20%", "20-40%", "40-60%", "60-80%", ">80%"]).astype(str)
dims = {"age": ["21-29", "30-39", "40-49", "50-59", "60+"], "lim": ["<=50k", "50-100k", "100-200k", "200-400k", ">400k"],
        "seg": ["Transactor", "Revolver", "At-Risk", "Dormant"], "del": [f"{i} late" for i in range(7)],
        "ub": ["0-20%", "20-40%", "40-60%", "60-80%", ">80%"]}
df["a"] = df.age_group.astype(str).map({v: i for i, v in enumerate(dims["age"])})
df["l"] = df.limit_band.astype(str).map({v: i for i, v in enumerate(dims["lim"])})
df["s"] = df.segment.map({v: i for i, v in enumerate(dims["seg"])})
df["u"] = df.ub.map({v: i for i, v in enumerate(dims["ub"])})
g = df.groupby(["a", "l", "s", "months_delayed", "u"]).agg(n=("ID", "size"), d=("default_flag", "sum"),
        su=("utilization_avg", "sum"), sp=("payment_ratio", "sum")).reset_index()
rows = [[int(r.a), int(r.l), int(r.s), int(r.months_delayed), int(r.u), int(r.n), int(r.d), round(r.su, 3), round(r.sp, 3)]
        for r in g.itertuples()]
dec = pd.read_csv(ROOT / "dashboard/dashboard_risk_deciles.csv")
D = {"dims": dims, "rows": rows,
     "dec": [[int(r.decile), float(r.default_rate_pct), float(r.cum_defaults_captured_pct)] for r in dec.itertuples()]}
html = (ROOT / "src/dashboard_template.html").read_text().replace("__DATA__", json.dumps(D, separators=(",", ":")))
(ROOT / "dashboard/credit_risk_dashboard.html").write_text(html)
print("cube rows:", len(rows), "size KB:", len(html) // 1024, "customers:", g.n.sum())
