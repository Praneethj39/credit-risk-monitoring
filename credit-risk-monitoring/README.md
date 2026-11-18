# Credit Card Default Risk Model with Drift Monitoring

Predicts which credit card customers will miss next month's payment, groups them into risk bands, and runs a monthly monitoring job that flags when the customer population drifts away from what the model was trained on.

**Data:** UCI "Default of Credit Card Clients" (30,000 card accounts, 6 months of balances, payments and repayment status; 22.1% defaulted).

## Pipeline

```
raw CSV -> SQLite -> SQL feature query -> train 3 models -> champion + model card
                                                      -> monthly batch scoring -> PSI drift + AUC checks -> alerts
                                                      -> FastAPI /score endpoint
```

| Step | File | What it does |
|---|---|---|
| Features | `src/features.sql`, `src/features.py` | 11 risk features in SQL: utilisation, months paid late, worst delay, repayment ratio, balance trend, zero-payment months |
| Train | `src/train.py` | Logistic regression vs random forest vs XGBoost, 60/20/20 split, picks champion on validation AUC, writes risk bands + model card |
| Monitor | `src/monitor.py` | Scores 6 monthly batches of 2,000 unseen accounts, computes PSI per feature and on the score, tracks AUC, writes `reports/monitoring_log.csv` and `reports/alerts.md` |
| Serve | `src/api.py` | `GET /health`, `POST /score` returns probability of default + risk band + model version |
| Test / CI | `tests/`, `.github/workflows/ci.yml` | 5 pytest checks (PSI maths, KS, no nulls, no protected attributes, API) run on every push |

Run everything: `pip install -r requirements.txt && python run_pipeline.py && pytest -q tests`

## Results (held-out test set, 6,000 accounts)

| Model | Val AUC | Test AUC | Gini | KS |
|---|---|---|---|---|
| Random forest (champion) | 0.774 | 0.787 | 0.574 | 0.438 |
| XGBoost | 0.772 | 0.790 | 0.579 | 0.445 |
| Logistic regression | 0.729 | 0.748 | 0.496 | 0.422 |

- Champion chosen on **validation** AUC (not test) so the test score stays an honest estimate.
- **Risk bands:** default rate climbs from 4.8% in the safest 10% to 71.5% in the riskiest 10%.
- The riskiest 20% of customers hold **53% of all defaults**. Declining them would cut the bad rate of the approved book from **22.1% to 13.0%**.
- Top drivers: last month's repayment status, worst delay in 6 months, number of late months.

## Monitoring results

Months 1-3 are normal; months 4-6 add a gradual stress scenario (10% / 25% / 40% of customers paying later, higher balances, smaller repayments).

| Month | Score PSI | Status | Features in alert | AUC |
|---|---|---|---|---|
| 1-3 | 0.006-0.009 | stable | 0 | 0.78-0.80 |
| 4 | 0.053 | stable | 0 | 0.79 |
| 5 | 0.275 | ALERT | 2 | 0.80 |
| 6 | 0.549 | ALERT | 4 | 0.77 |

Key point: PSI raised the alarm in month 5 while AUC still looked fine, so the shift is caught **before** the model's accuracy visibly drops, and the alert names the features causing it (repayment status, months late).

## Responsible-AI choices

- Sex, marital status and age are excluded from the features (protected attributes under fair-lending rules).
- Every score carries the model version; the model card records data, features, metrics and exclusions.

## Plain-language glossary (for interviews)

- **Probability of default (PD):** the model's estimate, 0 to 1, that a customer misses payment.
- **AUC:** if you pick one defaulter and one non-defaulter at random, the chance the model gives the defaulter a higher score. 0.5 = coin toss, 1.0 = perfect.
- **Gini:** 2 x AUC - 1. Banks quote this more often.
- **KS:** the biggest gap between how defaulters and non-defaulters are spread across scores. Higher = better separation.
- **PSI:** compares today's customers with the training customers bucket by bucket. Below 0.10 stable, 0.10-0.25 watch, above 0.25 investigate.
- **Drift:** the customers (or the data feeding the model) change over time, so the model is applied to people unlike those it learnt from.
