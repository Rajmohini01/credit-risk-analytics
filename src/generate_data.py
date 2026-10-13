"""Generate a synthetic credit-card-client dataset with the SAME schema as the
UCI 'Default of Credit Card Clients' data (30,000 rows).

Replace data/raw/credit_card_clients.csv with the real UCI file (rename columns:
PAY_0->PAY_1, 'default payment next month'->default_payment_next_month) and the
rest of the pipeline runs unchanged.
Month 1 = most recent month, month 6 = oldest.
PAY_x: -2 = no consumption, -1 = paid in full, 0 = revolving, 1..8 = months of delay.
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n=30000, seed=42, target_rate=0.221):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)  # hidden "financial stress" factor

    age = np.clip(np.round(rng.gamma(9, 3.3, n) + 8), 21, 79).astype(int)
    sex = rng.choice([1, 2], n, p=[0.40, 0.60])
    edu = rng.choice([1, 2, 3, 4, 0, 5, 6], n, p=[0.352, 0.468, 0.164, 0.004, 0.0005, 0.0095, 0.002])
    mar = rng.choice([1, 2, 3, 0], n, p=[0.455, 0.532, 0.011, 0.002])

    limit = np.exp(11.5 + 0.55 * rng.normal(size=n) + 0.15 * (edu == 1) - 0.2 * z)
    limit = np.clip(np.round(limit / 10000) * 10000, 10000, 1_000_000)

    tend = 0.8 * z + rng.normal(0, 0.6, n)
    pay = {}
    for t in range(6, 0, -1):
        x = 1.1 * tend + rng.normal(0, 0.7, n) - 0.75 + 0.08 * (6 - t) * np.maximum(tend, 0)
        pay[t] = np.clip(np.round(x), -2, 8).astype(int)

    util = np.clip(0.35 + 0.13 * z + 0.25 * rng.normal(size=n), 0.0, 1.25)
    bill = {}
    for t in range(1, 7):
        b = limit * np.clip(util + rng.normal(0, 0.07, n), 0, 1.4)
        refund = rng.random(n) < 0.003
        b = np.where(refund, -rng.uniform(0, 3000, n), b)
        bill[t] = np.round(b)

    payamt = {}
    for t in range(1, 7):
        state = pay[t]
        ratio = np.where(state <= -1, rng.uniform(0.9, 1.6, n),
                 np.where(state == 0, rng.uniform(0.04, 0.25, n), rng.uniform(0, 0.06, n)))
        base = np.maximum(bill[t + 1] if t < 6 else bill[6], 0)
        payamt[t] = np.round(base * ratio)

    avg_util = np.mean([bill[t] for t in range(1, 7)], axis=0) / limit
    recent = np.clip(pay[1], -1, 6)
    logit = 0.55 * z + 0.60 * recent + 0.9 * (avg_util - 0.4) + rng.normal(0, 0.35, n)
    lo, hi = -10.0, 10.0
    for _ in range(60):  # calibrate intercept to the target default rate
        mid = (lo + hi) / 2
        if sigmoid(logit + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    default = (rng.random(n) < sigmoid(logit + mid)).astype(int)

    df = pd.DataFrame({"ID": np.arange(1, n + 1), "LIMIT_BAL": limit.astype(int), "SEX": sex,
                       "EDUCATION": edu, "MARRIAGE": mar, "AGE": age})
    for t in range(1, 7):
        df[f"PAY_{t}"] = pay[t]
    for t in range(1, 7):
        df[f"BILL_AMT{t}"] = bill[t].astype(int)
    for t in range(1, 7):
        df[f"PAY_AMT{t}"] = payamt[t].astype(int)
    df["default_payment_next_month"] = default
    return df


if __name__ == "__main__":
    out = ROOT / "data/raw/credit_card_clients.csv"
    d = generate()
    d.to_csv(out, index=False)
    print(f"Saved {out} shape={d.shape} default_rate={d.default_payment_next_month.mean():.3f}")
