-- Q1: What is the overall default rate?
SELECT COUNT(*) AS total_customers, SUM(default_flag) AS defaults,
       ROUND(100.0*SUM(default_flag)/COUNT(*),2) AS default_rate_pct
FROM customer_summary;

-- Q2: Default rate by education and marital status
SELECT education_label, marriage_label, COUNT(*) AS customers,
       ROUND(100.0*AVG(default_flag),2) AS default_rate_pct
FROM customer_summary
GROUP BY education_label, marriage_label
HAVING COUNT(*) >= 100
ORDER BY default_rate_pct DESC;

-- Q3: Default rate by age group and credit-limit band (limit is the income proxy)
SELECT age_group, limit_band, COUNT(*) AS customers,
       ROUND(100.0*AVG(default_flag),2) AS default_rate_pct
FROM customer_summary
GROUP BY age_group, limit_band
HAVING COUNT(*) >= 100
ORDER BY default_rate_pct DESC
LIMIT 15;

-- Q4: Default rate by credit utilization band
WITH banded AS (
  SELECT default_flag,
    CASE WHEN utilization < 0.2 THEN '1: 0-20%' WHEN utilization < 0.4 THEN '2: 20-40%'
         WHEN utilization < 0.6 THEN '3: 40-60%' WHEN utilization < 0.8 THEN '4: 60-80%'
         ELSE '5: >80%' END AS util_band
  FROM customer_summary)
SELECT util_band, COUNT(*) AS customers, ROUND(100.0*AVG(default_flag),2) AS default_rate_pct
FROM banded GROUP BY util_band ORDER BY util_band;

-- Q5: Do late payments in earlier months predict default? (number of delayed months out of 6)
SELECT months_delayed, COUNT(*) AS customers, SUM(default_flag) AS defaults,
       ROUND(100.0*AVG(default_flag),2) AS default_rate_pct
FROM customer_summary GROUP BY months_delayed ORDER BY months_delayed;

-- Q6: Six-month trend of average bill and payment, defaulters vs non-defaulters (month 6 = oldest)
SELECT default_flag,
  ROUND(AVG(bill_amt6)) AS bill_m6, ROUND(AVG(bill_amt5)) AS bill_m5, ROUND(AVG(bill_amt4)) AS bill_m4,
  ROUND(AVG(bill_amt3)) AS bill_m3, ROUND(AVG(bill_amt2)) AS bill_m2, ROUND(AVG(bill_amt1)) AS bill_m1,
  ROUND(AVG(pay_amt6)) AS paid_m6, ROUND(AVG(pay_amt3)) AS paid_m3, ROUND(AVG(pay_amt1)) AS paid_m1
FROM credit_clients GROUP BY default_flag;

-- Q7: Average spend, utilization and repayment ratio: defaulters vs non-defaulters
SELECT default_flag, COUNT(*) AS customers, ROUND(AVG(avg_bill)) AS avg_monthly_bill,
       ROUND(AVG(utilization),3) AS avg_utilization, ROUND(AVG(payment_ratio),3) AS avg_payment_ratio
FROM customer_summary GROUP BY default_flag;

-- Q8: Top 3 highest-utilization customers within each age group (window function)
WITH ranked AS (
  SELECT id, age_group, ROUND(utilization,2) AS utilization, default_flag,
         RANK() OVER (PARTITION BY age_group ORDER BY utilization DESC) AS risk_rank
  FROM customer_summary)
SELECT * FROM ranked WHERE risk_rank <= 3 ORDER BY age_group, risk_rank;

-- Q9: Cumulative share of all defaults captured by risk decile (simple rule-based score)
WITH scored AS (
  SELECT id, default_flag,
         2.0*MAX(pay_1,0) + months_delayed + 3.0*MIN(MAX(utilization,0),1.5) AS risk_score
  FROM customer_summary),
bucket AS (
  SELECT *, NTILE(10) OVER (ORDER BY risk_score DESC, id) AS risk_decile FROM scored),
agg AS (
  SELECT risk_decile, COUNT(*) AS customers, SUM(default_flag) AS defaults
  FROM bucket GROUP BY risk_decile)
SELECT risk_decile, customers, defaults,
       ROUND(100.0*defaults/customers,2) AS default_rate_pct,
       ROUND(100.0*SUM(defaults) OVER (ORDER BY risk_decile)/SUM(defaults) OVER (),2) AS cum_defaults_captured_pct
FROM agg ORDER BY risk_decile;

-- Q10: Early-warning rule: 2+ delayed months OR (delayed last month AND utilization > 80%)
WITH flagged AS (
  SELECT default_flag,
         CASE WHEN months_delayed >= 2 OR (pay_1 >= 1 AND utilization_latest > 0.8) THEN 1 ELSE 0 END AS flag
  FROM customer_summary)
SELECT flag AS early_warning_flag, COUNT(*) AS customers, SUM(default_flag) AS defaults,
       ROUND(100.0*AVG(default_flag),2) AS default_rate_pct,
       ROUND(100.0*SUM(default_flag)/(SELECT SUM(default_flag) FROM flagged),2) AS share_of_all_defaults_pct
FROM flagged GROUP BY flag;

-- Q11: Is the repayment status getting worse? (latest month vs previous month)
WITH t AS (
  SELECT default_flag,
    CASE WHEN pay_1 > pay_2 THEN 'Worsening' WHEN pay_1 = pay_2 THEN 'Stable' ELSE 'Improving' END AS trend
  FROM customer_summary)
SELECT trend, COUNT(*) AS customers, ROUND(100.0*AVG(default_flag),2) AS default_rate_pct
FROM t GROUP BY trend ORDER BY default_rate_pct DESC;
