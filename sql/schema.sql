-- View with business-friendly labels and engineered fields.
-- Base table: credit_clients (loaded from the cleaned CSV). Works in SQLite;
-- for PostgreSQL/MySQL replace MAX(a,b) with GREATEST(a,b) and MIN(a,b) with LEAST(a,b).
DROP VIEW IF EXISTS customer_summary;
CREATE VIEW customer_summary AS
SELECT
  id, limit_bal, age, pay_1, pay_2, pay_3, pay_4, pay_5, pay_6, default_flag,
  bill_amt1,
  CASE education WHEN 1 THEN 'Graduate school' WHEN 2 THEN 'University'
                 WHEN 3 THEN 'High school' ELSE 'Other' END AS education_label,
  CASE marriage WHEN 1 THEN 'Married' WHEN 2 THEN 'Single' ELSE 'Other' END AS marriage_label,
  CASE WHEN age < 30 THEN '21-29' WHEN age < 40 THEN '30-39' WHEN age < 50 THEN '40-49'
       WHEN age < 60 THEN '50-59' ELSE '60+' END AS age_group,
  CASE WHEN limit_bal <= 50000 THEN '1: <=50k' WHEN limit_bal <= 100000 THEN '2: 50-100k'
       WHEN limit_bal <= 200000 THEN '3: 100-200k' WHEN limit_bal <= 400000 THEN '4: 200-400k'
       ELSE '5: >400k' END AS limit_band,
  (bill_amt1+bill_amt2+bill_amt3+bill_amt4+bill_amt5+bill_amt6)/6.0 AS avg_bill,
  (bill_amt1+bill_amt2+bill_amt3+bill_amt4+bill_amt5+bill_amt6)/6.0/limit_bal AS utilization,
  bill_amt1*1.0/limit_bal AS utilization_latest,
  (pay_amt1+pay_amt2+pay_amt3+pay_amt4+pay_amt5+pay_amt6)*1.0 /
    NULLIF(MAX(bill_amt1,0)+MAX(bill_amt2,0)+MAX(bill_amt3,0)+MAX(bill_amt4,0)+MAX(bill_amt5,0)+MAX(bill_amt6,0),0) AS payment_ratio,
  ((pay_1>=1)+(pay_2>=1)+(pay_3>=1)+(pay_4>=1)+(pay_5>=1)+(pay_6>=1)) AS months_delayed
FROM credit_clients;
