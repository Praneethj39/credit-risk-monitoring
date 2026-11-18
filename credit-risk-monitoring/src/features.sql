-- Feature engineering for credit card default risk.
-- Runs against the raw `accounts` table loaded into SQLite.
-- PAY_1..PAY_6 = repayment status for the last 6 months (-2/-1/0 = paid or no use, 1..8 = months late)
-- SEX, MARRIAGE and AGE are deliberately left out (fair-lending: protected attributes).

WITH base AS (
    SELECT
        ID AS account_id,
        LIMIT_BAL AS credit_limit,
        EDUCATION AS education,
        PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
        BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
        PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6,
        dpnm AS defaulted
    FROM accounts
)
SELECT
    account_id,
    credit_limit,
    CASE WHEN education IN (1, 2, 3) THEN education ELSE 4 END AS education,
    PAY_1 AS status_last_month,
    -- how many of the last 6 months were paid late
    (PAY_1 > 0) + (PAY_2 > 0) + (PAY_3 > 0) + (PAY_4 > 0) + (PAY_5 > 0) + (PAY_6 > 0) AS months_late_6m,
    MAX(PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6) AS worst_delay_6m,
    -- utilisation: current balance as a share of the limit
    ROUND(1.0 * BILL_AMT1 / credit_limit, 4) AS utilisation,
    ROUND(1.0 * (BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6) / 6 / credit_limit, 4) AS avg_utilisation_6m,
    -- balance trend: is the customer's balance growing?
    ROUND(1.0 * (BILL_AMT1 - BILL_AMT6) / credit_limit, 4) AS balance_trend_6m,
    -- repayment ratio: how much of what was billed actually got paid
    ROUND(1.0 * (PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6)
          / MAX(BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6, 1), 4) AS repayment_ratio_6m,
    ROUND(1.0 * PAY_AMT1 / MAX(BILL_AMT2, 1), 4) AS last_payment_ratio,
    (PAY_AMT1 = 0) + (PAY_AMT2 = 0) + (PAY_AMT3 = 0) + (PAY_AMT4 = 0) + (PAY_AMT5 = 0) + (PAY_AMT6 = 0) AS zero_payment_months,
    defaulted
FROM base;
