"""Builds README.md from outputs/results.json so every number matches the latest run."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "outputs/results.json").read_text())
S = R["sql"]
pct = lambda x, d=1: f"{x*100:.{d}f}%"
md = lambda df: df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)


def table(rows, cols, names=None):
    names = names or cols
    out = ["| " + " | ".join(names) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


o, th, ew = R["overall"], R["threshold"], R["early_warning"]
best = R["models"][0]
q4, q5 = S["Q4"]["rows"], S["Q5"]["rows"]
dec = R["deciles"]
seg = {s["segment"]: s for s in R["segments"]}
d1, d3 = dec[0], dec[2]
cost_cut = 1 - th["cost_at_best"] / th["cost_at_0.5"]
models_tbl = table([{**m, **{k: f"{m[k]:.3f}" for k in ["roc_auc", "ks", "precision", "recall", "f1"]}} for m in R["models"]],
                   ["model", "roc_auc", "ks", "precision", "recall", "f1"],
                   ["Model", "ROC-AUC", "KS", "Precision", "Recall", "F1"])
seg_order = ["Transactor", "Revolver", "At-Risk", "Dormant"]
seg_tbl = table([{"segment": s, "share": f"{seg[s]['share_pct']}%", "customers": f"{seg[s]['customers']:,}",
                  "dr": f"{seg[s]['default_rate_pct']:.1f}%", "util": f"{seg[s]['avg_utilization']*100:.0f}%",
                  "pr": f"{seg[s]['avg_payment_ratio']:.2f}"} for s in seg_order],
                ["segment", "customers", "share", "dr", "util", "pr"],
                ["Segment", "Customers", "Share", "Default rate", "Avg utilization", "Payment ratio"])
dec_tbl = table([{"d": r["decile"], "dr": f"{r['default_rate_pct']:.1f}%", "cum": f"{r['cum_defaults_captured_pct']:.1f}%",
                  "lift": f"{r['lift']:.2f}x"} for r in dec],
                ["d", "dr", "cum", "lift"], ["Risk decile (1 = riskiest)", "Default rate", "Cumulative defaults captured", "Lift"])
feat = ", ".join(f"`{k}`" for k in list(R["top_features"])[:4])
clean = "\n".join(f"- {l}" for l in R["cleaning_log"])

README = f"""# Credit Risk & Spending Analytics: Loan Default and Credit Card Behavior

> **Data note:** this repo ships with a **synthetic** dataset that mirrors the schema of the UCI
> *Default of Credit Card Clients* data (30,000 customers, 6 months of history). The sandbox this was built in had no
> internet access. All numbers below come from the synthetic data and are for demonstration only.
> To get real findings, drop the real UCI CSV into `data/raw/credit_card_clients.csv` (see
> [Using the real dataset](#using-the-real-dataset)) and rerun the three commands in [How to run](#how-to-run).

## 1. Business problem
A retail bank sees rising credit card defaults and uneven card usage. As the data analyst I answer:
1. How big is the default problem and which segments drive it?
2. What are the strongest early signals of default?
3. Can a simple, explainable model rank customers by risk?
4. How do customers differ in spending and repayment behavior?
5. What should the bank do about it, and what is it worth?

## 2. Dataset
30,000 customers: credit limit, age, education, marital status, 6 months of repayment status (`PAY_1..6`),
bill amounts (`BILL_AMT1..6`), payment amounts (`PAY_AMT1..6`), and the target `default_flag`
(defaulted next month). Month 1 is the most recent month.

**Cleaning decisions** (logged by `src/pipeline.py`):
{clean}

## 3. Tools
Python (pandas, scikit-learn, matplotlib, seaborn), SQL (SQLite; portable to PostgreSQL/MySQL), Power BI or Tableau for the dashboard.

## 4. Key findings
- **Overall default rate is {pct(o['default_rate'])}** ({o['defaults']:,} of {o['customers']:,} customers).
- **Repayment history is the strongest signal.** Customers with 0 delayed months out of 6 default at {q5[0]['default_rate_pct']:.1f}%,
  versus {q5[-1]['default_rate_pct']:.1f}% for customers delayed in all 6 months (SQL Q5).
- **Utilization matters.** Default rate rises from {q4[0]['default_rate_pct']:.1f}% (0-20% utilization) to
  {q4[-1]['default_rate_pct']:.1f}% (above 80%) (SQL Q4).
- **Top model drivers** (permutation importance): {feat}.
- **Risk ranking works.** The riskiest 10% of customers default at {d1['default_rate_pct']:.1f}% ({d1['lift']:.1f}x average) and the
  top 3 deciles capture {d3['cum_defaults_captured_pct']:.1f}% of all defaults.
- **Segments differ sharply:** At-Risk customers default at {seg['At-Risk']['default_rate_pct']:.1f}% versus {seg['Transactor']['default_rate_pct']:.1f}% for Transactors.

![Default by repayment status](reports/figures/01_default_by_repayment_status.png)
![Default by utilization and age](reports/figures/02_default_by_utilization_and_age.png)

## 5. SQL analysis
`sql/analysis_queries.sql` answers 11 business questions using CTEs, window functions (`RANK`, `NTILE`, running `SUM() OVER`),
conditional aggregation, and a reusable view (`sql/schema.sql`). Result tables are saved in `outputs/sql_results/`.

| # | Question |
|---|---|
""" + "\n".join(f"| {k} | {v['title']} |" for k, v in S.items()) + f"""

## 6. Feature engineering
`utilization_avg`, `utilization_latest` (bill / limit), `payment_ratio` (paid / billed), `avg_delay`, `max_delay`,
`months_delayed`, `bill_volatility` (std of bills / limit), plus age, limit and utilization bands.

## 7. Predictive model
Stratified 80/20 split. Class weights handle the 22/78 imbalance. `SEX` and `MARRIAGE` are **excluded** from the model
(fair-lending practice) and only past-behavior features are used (no target leakage).

{models_tbl}

![ROC curves](reports/figures/05_roc_curves.png)

**Model choice:** {best['model']} performs on par with the tree models (ROC-AUC {best['roc_auc']:.3f}, KS {best['ks']:.3f}), so it is the
recommended model because it is simple and explainable to risk and compliance teams.

**Threshold chosen by business cost.** Assuming a missed default costs {th['cost_fn']}x a wrongly flagged good customer, the cost-minimizing
threshold is **{th['best_threshold']}**: it catches {pct(th['recall'])} of defaulters with {pct(th['precision'])} precision, flags {pct(th['flagged_pct'])}
of customers, and lowers total cost by {pct(cost_cut)} versus the default 0.5 cutoff. Accuracy is not used because 78% accuracy is achievable by predicting
"no default" for everyone.

{dec_tbl}

![Risk deciles](reports/figures/08_risk_deciles.png)
![Feature importance](reports/figures/07_feature_importance.png)

## 8. Customer segmentation
Rule-based segments (explainable to business teams), cross-checked with K-Means (`outputs/kmeans_profiles.csv`):

{seg_tbl}

- **Transactors:** pay in full, low risk. Offer rewards and limit increases.
- **Revolvers:** carry balances, profitable but watch closely. Offer reminders and installment plans.
- **At-Risk:** 2+ delayed months. Trigger early collections outreach and restructuring offers.
- **Dormant:** almost no usage. Run reactivation campaigns.

![Segments](reports/figures/09_segments.png)

## 9. Recommendations
1. **Deploy an early-warning rule:** flag customers with 2+ delayed months, or a delay last month with utilization above 80%.
   It flags {pct(ew['flagged_pct'])} of the book, those customers default at {pct(ew['precision'])} ({ew['lift']:.1f}x average), and it catches
   {pct(ew['recall'])} of all defaulters ({ew['defaulters_caught']:,} customers).
2. **Use the model score to prioritize collections:** work deciles 1-3 first (they hold {d3['cum_defaults_captured_pct']:.0f}% of defaults).
3. **Review limits** for customers with sustained utilization above 80% and a delay history.
4. **Grow healthy accounts:** target Transactors with rewards and selective limit increases.
5. **Estimated impact (assumption-based):** with an assumed loss given default of {pct(ew['assumed_lgd'],0)} and intervention success of
   {pct(ew['assumed_success'],0)}, acting on the flagged defaulters avoids roughly **{ew['estimated_loss_avoided']/1e6:,.1f}M** in currency units
   (out of {ew['total_defaulter_exposure']/1e6:,.0f}M exposure held by defaulters). Both assumptions are adjustable in `src/pipeline.py`.

## 10. Dashboard (Power BI / Tableau)
Import the files in `dashboard/`: `dashboard_customer_level.csv` (one row per customer with segment, risk score, decile, tier),
`dashboard_segments.csv`, `dashboard_risk_deciles.csv`.

| Page | Visuals |
|---|---|
| 1. Executive overview | KPI cards (customers, default rate, avg utilization, high-risk customers), default rate by age group / limit band, slicers |
| 2. Risk | Default rate by months delayed, utilization band vs default, risk tier distribution |
| 3. Spending behavior | Segment sizes, avg bill and payment ratio by segment, segment default rate |
| 4. Model | Decile table, cumulative capture (gains) chart, top-risk customer list |

Starter DAX measures:
```
Customers = COUNTROWS(dashboard_customer_level)
Default Rate = DIVIDE(SUM(dashboard_customer_level[default_flag]), [Customers])
High Risk Customers = CALCULATE([Customers], dashboard_customer_level[risk_tier] = "High")
Avg Utilization = AVERAGE(dashboard_customer_level[utilization_avg])
High Risk Default Rate = CALCULATE([Default Rate], dashboard_customer_level[risk_tier] = "High")
```
A `.pbix` file cannot be generated outside Power BI Desktop, so the dashboard itself must be built from these files.

## 11. How to run
```bash
pip install -r requirements.txt
python src/generate_data.py   # skip if you use the real dataset
python src/pipeline.py        # cleaning, SQL, EDA, model, segments, exports
python src/build_dashboard.py # builds dashboard/credit_risk_dashboard.html\npython src/build_readme.py    # regenerates this README with the latest numbers\npytest -q                     # smoke tests (or: make all)
```

## 12. Deploy with GitHub Actions
`.github/workflows/pipeline.yml` runs on every push to `main`, on pull requests, and on demand. It installs dependencies, runs the
pipeline, rebuilds the README and dashboard, runs `pytest`, uploads the results as a downloadable artifact, and (on `main`) publishes the
dashboard to **GitHub Pages**.

One-time setup: push the repo to GitHub, then go to **Settings > Pages > Build and deployment > Source** and choose **GitHub Actions**.
The dashboard will be live at `https://<your-username>.github.io/<repo-name>/`. Run history is under the **Actions** tab.

## 13. Using the real dataset
1. Download *Default of Credit Card Clients* (UCI or Kaggle).
2. Rename columns: `PAY_0` to `PAY_1`, `default payment next month` to `default_payment_next_month`.
3. Save as `data/raw/credit_card_clients.csv`, then rerun the commands above.

## 14. Repo structure
```
data/raw, data/processed   input and cleaned data
sql/                       schema.sql (view), analysis_queries.sql (11 queries)
src/                       generate_data.py, pipeline.py, build_dashboard.py, build_readme.py
tests/                     pytest smoke tests
.github/workflows/         CI/CD pipeline (run + test + deploy to GitHub Pages)
outputs/                   model comparison, deciles, segments, SQL result CSVs, results.json
reports/figures/           9 charts
dashboard/                 BI-ready CSV files
```

## 15. Limitations and future work
- Synthetic demo data; real findings need the real file. Historical snapshot, no macroeconomic data.
- Single random split; use time-based validation and cross-validation on real data.
- Add SHAP explanations, probability calibration, and a fairness audit across age and gender.
- Replace assumed LGD and intervention success with the bank's own figures.

## 16. Resume bullets (replace numbers after running on real data)
- Analyzed 30,000 credit card accounts with SQL and Python, identifying repayment delays and utilization as the top default drivers.
- Built an explainable credit risk model (ROC-AUC {best['roc_auc']:.2f}, KS {best['ks']:.2f}) whose top 3 risk deciles capture {d3['cum_defaults_captured_pct']:.0f}% of defaults.
- Designed an early-warning rule and customer segmentation that isolate a {pct(ew['flagged_pct'],0)} high-risk slice defaulting at {ew['lift']:.1f}x the average rate.
- Delivered a 4-page Power BI dashboard on default rate, risk segments, and spending behavior.
"""
(ROOT / "README.md").write_text(README)
print("README written", len(README))
