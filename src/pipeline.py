"""End-to-end credit risk analytics pipeline.
Sections are marked with '# %%' so this file opens as notebook cells in VS Code / Jupytext.
Run:  python src/pipeline.py
"""
# %% 0. Setup
import json, re, sqlite3, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"
for d in ["data/processed", "outputs/sql_results", "reports/figures", "dashboard"]:
    (ROOT / d).mkdir(parents=True, exist_ok=True)  # fresh git checkouts have no empty folders
SEED = 42
R = {}  # results collected for the README

PAY = [f"PAY_{i}" for i in range(1, 7)]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT = [f"PAY_AMT{i}" for i in range(1, 7)]


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=130)
    plt.close()


# %% 1. Data cleaning
raw = pd.read_csv(ROOT / "data/raw/credit_card_clients.csv")
df = raw.copy()
log = [f"Loaded {len(df):,} rows x {df.shape[1]} columns",
       f"Missing values: {int(df.isna().sum().sum())}; duplicate IDs: {int(df.ID.duplicated().sum())}"]
df = df.drop_duplicates("ID")
n = int(df.EDUCATION.isin([0, 5, 6]).sum())
df["EDUCATION"] = df.EDUCATION.replace({0: 4, 5: 4, 6: 4})
log.append(f"EDUCATION: {n} rows with undefined codes (0, 5, 6) merged into 'Other' (4)")
n = int((df.MARRIAGE == 0).sum())
df["MARRIAGE"] = df.MARRIAGE.replace({0: 3})
log.append(f"MARRIAGE: {n} rows with undefined code 0 merged into 'Other' (3)")
n = int((df[BILL] < 0).any(axis=1).sum())
log.append(f"BILL_AMT: {n} customers have negative bills (credit balances/refunds); kept, "
           "but clipped at 0 when computing payment ratio")
df = df.rename(columns={"default_payment_next_month": "default_flag"})
R["cleaning_log"] = log
base_cols = list(df.columns)
df.to_csv(ROOT / "data/processed/credit_clients_clean.csv", index=False)

# %% 2. SQL analysis (SQLite)
conn = sqlite3.connect(":memory:")
df.rename(columns=str.lower).to_sql("credit_clients", conn, index=False)
conn.executescript((ROOT / "sql/schema.sql").read_text())
text = (ROOT / "sql/analysis_queries.sql").read_text()
parts = re.split(r"^-- (Q\d+): (.*)$", text, flags=re.M)
sql_out = {}
for i in range(1, len(parts), 3):
    qid, title, query = parts[i], parts[i + 1], parts[i + 2].strip()
    res = pd.read_sql_query(query, conn)
    res.to_csv(ROOT / f"outputs/sql_results/{qid}.csv", index=False)
    sql_out[qid] = {"title": title, "rows": res.to_dict("records")}
R["sql"] = sql_out
print(f"Ran {len(sql_out)} SQL queries")

# %% 3. Feature engineering
tot_bill = df[BILL].clip(lower=0).sum(axis=1)
df["utilization_avg"] = (df[BILL].mean(axis=1) / df.LIMIT_BAL).clip(0, 1.5)
df["utilization_latest"] = (df.BILL_AMT1 / df.LIMIT_BAL).clip(0, 1.5)
df["payment_ratio"] = np.where(tot_bill > 0, df[PAYAMT].sum(axis=1) / tot_bill.replace(0, np.nan), 1.0)
df["payment_ratio"] = pd.Series(df["payment_ratio"], index=df.index).clip(0, 2).fillna(1.0)
df["avg_delay"] = df[PAY].clip(lower=0).mean(axis=1)
df["max_delay"] = df[PAY].clip(lower=0).max(axis=1)
df["months_delayed"] = (df[PAY] >= 1).sum(axis=1)
df["bill_volatility"] = df[BILL].std(axis=1) / df.LIMIT_BAL
df["avg_bill"] = df[BILL].mean(axis=1)
df["age_group"] = pd.cut(df.AGE, [20, 29, 39, 49, 59, 100], labels=["21-29", "30-39", "40-49", "50-59", "60+"])
df["limit_band"] = pd.cut(df.LIMIT_BAL, [0, 50e3, 100e3, 200e3, 400e3, 1e9],
                          labels=["<=50k", "50-100k", "100-200k", "200-400k", ">400k"])
df["util_band"] = pd.cut(df.utilization_avg, [-0.01, 0.2, 0.4, 0.6, 0.8, 10],
                         labels=["0-20%", "20-40%", "40-60%", "60-80%", ">80%"])
df["education_label"] = df.EDUCATION.map({1: "Graduate school", 2: "University", 3: "High school", 4: "Other"})

# %% 4. EDA charts
R["overall"] = {"customers": len(df), "defaults": int(df.default_flag.sum()),
                "default_rate": float(df.default_flag.mean())}
avg = df.default_flag.mean() * 100

fig, ax = plt.subplots(figsize=(8, 4.5))
g = df.assign(s=df.PAY_1.clip(-2, 4)).groupby("s").default_flag.agg(["mean", "size"])
g = g[g["size"] >= 50]
ax.bar(g.index.astype(str), g["mean"] * 100, color=sns.color_palette("deep")[0])
ax.axhline(avg, color="red", ls="--", label=f"Overall {avg:.1f}%")
ax.set(xlabel="Latest repayment status (PAY_1; -2 none, -1 paid, 0 revolving, 1+ months late, 4 = 4+)",
       ylabel="Default rate (%)", title="Default rate by latest repayment status")
ax.legend()
savefig("01_default_by_repayment_status")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, title in zip(axes, ["util_band", "age_group"], ["Credit utilization", "Age group"]):
    t = df.groupby(col, observed=True).default_flag.mean() * 100
    ax.bar(t.index.astype(str), t.values, color=sns.color_palette("deep")[1])
    ax.axhline(avg, color="red", ls="--")
    ax.set(title=f"Default rate by {title.lower()}", ylabel="Default rate (%)", xlabel=title)
savefig("02_default_by_utilization_and_age")

num_cols = ["LIMIT_BAL", "AGE", "utilization_avg", "payment_ratio", "avg_delay", "months_delayed",
            "bill_volatility", "default_flag"]
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Correlation heatmap")
savefig("03_correlation_heatmap")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, ["utilization_avg", "payment_ratio", "avg_bill"]):
    sns.boxplot(data=df, x="default_flag", y=col, ax=ax, showfliers=False)
    ax.set(xlabel="Default (0 = no, 1 = yes)", title=col)
savefig("04_boxplots_defaulters_vs_non")

# %% 5. Modeling
FEATURES = (["LIMIT_BAL", "AGE", "EDUCATION"] + PAY + BILL + PAYAMT +
            ["utilization_avg", "utilization_latest", "payment_ratio", "avg_delay", "max_delay",
             "months_delayed", "bill_volatility"])
# SEX and MARRIAGE intentionally excluded (fair-lending practice); only past-behaviour features used (no leakage).
X, y = df[FEATURES], df.default_flag
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                                        min_samples_leaf=40, l2_regularization=1.0,
                                                        class_weight="balanced", random_state=SEED),
}
rows, probs = [], {}
plt.figure(figsize=(6, 5))
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    probs[name] = p
    pred = (p >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_te, p)
    rows.append({"model": name, "roc_auc": roc_auc_score(y_te, p), "ks": float((tpr - fpr).max()),
                 "precision": precision_score(y_te, pred), "recall": recall_score(y_te, pred),
                 "f1": f1_score(y_te, pred)})
    plt.plot(fpr, tpr, label=f"{name} (AUC {rows[-1]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.8)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves (test set)")
plt.legend(loc="lower right")
savefig("05_roc_curves")
res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
res.to_csv(ROOT / "outputs/model_comparison.csv", index=False)
R["models"] = res.round(4).to_dict("records")
best = res.iloc[0].model
bm, bp = models[best], probs[best]
R["best_model"] = best

# Business-driven threshold: a missed default costs 5x a wrongly flagged good customer
COST_FN, COST_FP = 5, 1
ths = np.arange(0.05, 0.96, 0.01)
costs = []
for t in ths:
    tn, fp, fn, tp = confusion_matrix(y_te, (bp >= t).astype(int)).ravel()
    costs.append(COST_FN * fn + COST_FP * fp)
bt = float(ths[int(np.argmin(costs))])
tn, fp, fn, tp = confusion_matrix(y_te, (bp >= bt).astype(int)).ravel()
R["threshold"] = {"best_threshold": round(bt, 2), "cost_fn": COST_FN, "cost_fp": COST_FP,
                  "recall": tp / (tp + fn), "precision": tp / (tp + fp),
                  "flagged_pct": (tp + fp) / len(y_te),
                  "cost_at_best": int(min(costs)),
                  "cost_at_0.5": int(costs[int(round((0.5 - 0.05) / 0.01))])}
plt.figure(figsize=(7, 4))
plt.plot(ths, costs); plt.axvline(bt, color="red", ls="--", label=f"Best threshold {bt:.2f}")
plt.xlabel("Probability threshold"); plt.ylabel(f"Total cost (FN x{COST_FN} + FP x{COST_FP})")
plt.title("Choosing the decision threshold by business cost"); plt.legend()
savefig("06_threshold_cost")

# Feature importance (permutation, on test set)
pi = permutation_importance(bm, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
R["top_features"] = imp.head(8).round(4).to_dict()
plt.figure(figsize=(7, 5))
imp.head(12)[::-1].plot.barh(color=sns.color_palette("deep")[2])
plt.xlabel("Drop in ROC-AUC when shuffled"); plt.title(f"Top drivers of default ({best})")
savefig("07_feature_importance")

# Risk deciles on the test set (decile 1 = highest predicted risk)
t = pd.DataFrame({"y": y_te.values, "p": bp})
t["decile"] = pd.qcut(t.p.rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
dec = t.groupby("decile").agg(customers=("y", "size"), defaults=("y", "sum"), avg_score=("p", "mean"))
dec["default_rate_pct"] = dec.defaults / dec.customers * 100
dec["cum_defaults_captured_pct"] = dec.defaults.cumsum() / dec.defaults.sum() * 100
dec["lift"] = dec.default_rate_pct / (t.y.mean() * 100)
dec.round(2).to_csv(ROOT / "outputs/risk_deciles.csv")
R["deciles"] = dec.round(2).reset_index().to_dict("records")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(dec.index, dec.default_rate_pct, color=sns.color_palette("deep")[3])
ax.set(xlabel="Risk decile (1 = riskiest)", ylabel="Default rate (%)", title="Risk decile performance (test set)")
ax2 = ax.twinx(); ax2.plot(dec.index, dec.cum_defaults_captured_pct, "k-o"); ax2.set_ylabel("Cumulative % of defaults captured")
ax2.grid(False)
savefig("08_risk_deciles")

# Out-of-fold risk scores for every customer (used by the dashboard)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
df["risk_score"] = cross_val_predict(clone(bm), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
df["risk_decile"] = pd.qcut(df.risk_score.rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
df["risk_tier"] = pd.cut(df.risk_decile, [0, 2, 5, 10], labels=["High", "Medium", "Low"])

# %% 6. Segmentation
cond = [df.months_delayed >= 2,
        (df.utilization_avg < 0.05) & (df.months_delayed == 0),
        (df.months_delayed == 0) & (df.payment_ratio >= 0.8)]
df["segment"] = np.select(cond, ["At-Risk", "Dormant", "Transactor"], default="Revolver")
seg = df.groupby("segment").agg(customers=("ID", "size"), default_rate_pct=("default_flag", lambda s: s.mean() * 100),
                                avg_limit=("LIMIT_BAL", "mean"), avg_bill=("avg_bill", "mean"),
                                avg_utilization=("utilization_avg", "mean"), avg_payment_ratio=("payment_ratio", "mean"),
                                avg_risk_score=("risk_score", "mean")).round(3)
seg["share_pct"] = (seg.customers / len(df) * 100).round(1)
seg.to_csv(ROOT / "outputs/segments.csv")
R["segments"] = seg.reset_index().to_dict("records")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
order = ["Transactor", "Revolver", "At-Risk", "Dormant"]
sns.barplot(x=seg.loc[order].index, y=seg.loc[order].customers, ax=axes[0], color=sns.color_palette("deep")[0])
axes[0].set(title="Customers per segment", xlabel="", ylabel="Customers")
sns.barplot(x=seg.loc[order].index, y=seg.loc[order].default_rate_pct, ax=axes[1], color=sns.color_palette("deep")[3])
axes[1].set(title="Default rate by segment (%)", xlabel="", ylabel="Default rate (%)")
savefig("09_segments")

# Data-driven cross-check with K-Means
cl_feats = ["utilization_avg", "payment_ratio", "avg_delay", "bill_volatility"]
km = KMeans(4, n_init=10, random_state=SEED).fit(StandardScaler().fit_transform(df[cl_feats]))
df["kmeans_cluster"] = km.labels_
kp = df.groupby("kmeans_cluster").agg(customers=("ID", "size"), default_rate_pct=("default_flag", lambda s: s.mean() * 100),
                                      **{c: (c, "mean") for c in cl_feats}).round(3)
kp.to_csv(ROOT / "outputs/kmeans_profiles.csv")
R["kmeans"] = kp.reset_index().to_dict("records")

# %% 7. Early-warning rule and impact estimate
flag = (df.months_delayed >= 2) | ((df.PAY_1 >= 1) & (df.utilization_latest > 0.8))
caught = df[flag & (df.default_flag == 1)]
LGD, SUCCESS = 0.60, 0.25  # ASSUMPTIONS: loss given default, share of flagged defaulters saved by intervention
exposure = float(caught.BILL_AMT1.clip(lower=0).mean())
R["early_warning"] = {"flagged_pct": float(flag.mean()), "precision": float(df[flag].default_flag.mean()),
                      "recall": float(len(caught) / df.default_flag.sum()),
                      "lift": float(df[flag].default_flag.mean() / df.default_flag.mean()),
                      "defaulters_caught": int(len(caught)), "avg_exposure": exposure,
                      "assumed_lgd": LGD, "assumed_success": SUCCESS,
                      "estimated_loss_avoided": float(len(caught) * exposure * LGD * SUCCESS),
                      "total_defaulter_exposure": float(df[df.default_flag == 1].BILL_AMT1.clip(lower=0).sum())}

# %% 8. Export dashboard-ready data
export = ["ID", "LIMIT_BAL", "AGE", "age_group", "limit_band", "education_label", "PAY_1", "months_delayed",
          "utilization_avg", "utilization_latest", "payment_ratio", "avg_bill", "avg_delay", "segment",
          "risk_score", "risk_decile", "risk_tier", "default_flag"]
df[export].to_csv(ROOT / "dashboard/dashboard_customer_level.csv", index=False)
seg.reset_index().to_csv(ROOT / "dashboard/dashboard_segments.csv", index=False)
dec.reset_index().to_csv(ROOT / "dashboard/dashboard_risk_deciles.csv", index=False)

(ROOT / "outputs/results.json").write_text(json.dumps(R, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps({k: R[k] for k in ["overall", "models", "threshold", "early_warning", "top_features"]}, indent=1,
                 default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(seg)
print(dec.round(2))
