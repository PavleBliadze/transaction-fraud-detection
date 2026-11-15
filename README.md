# Transaction Fraud Detection

A fraud model for an online store that sells digital goods (gift cards, game keys, subscriptions). The goal is to estimate, at the moment of purchase, how likely it is that the payment method is stolen, and to turn that estimate into an accept / reject decision that the business can manage.

The whole project is in one notebook: [`fraud_detection.ipynb`](fraud_detection.ipynb).

## Problem

When a fraudster buys with a stolen card and the purchase goes through, the real owner disputes it and the money is returned. This is a chargeback, and for digital goods it is a full loss because the key is already delivered. Rejecting too much is also expensive, since every legitimate buyer that gets blocked is lost revenue.

So the model is judged on two things at once: how many chargebacks it prevents and how many honest buyers it blocks.

## Data

An event log with 369,853 transactions and 3,220 chargeback notifications, January to June 2021, extracted on 16 August 2021. The data is confidential and is not included in this repository.

A few things in the data shaped the whole approach:

- Chargebacks arrive late. The median delay is about 11 days, but 5% arrive after more than two months, so the labels of the last weeks are not final.
- Fraud is only visible on transactions that the existing system accepted (about 62% of traffic). Rejected transactions have no outcome.
- The fraud rate moves a lot over time: 0.4% in January, 2.1% in March.
- `userId` and `txId` are version 1 UUIDs, so they carry a creation timestamp. That gives account age and checkout duration for free.

## Approach

1. **Target.** Fraud is a chargeback that arrived within 60 days of the transaction. A transaction is clean only if it had at least 60 days to receive one. Late chargebacks are removed from training and evaluation instead of being labelled either way.
2. **Split.** By time, not at random: train on January to April, test on 1 May to 16 June. The test set is touched once, at the end.
3. **Features.** Price, unit price, quantity, account age, checkout duration, country mismatch, missing value flags, product type, weekday, and five high cardinality columns (product, issuer, payment country, IP country, email domain) encoded with a time-ordered target encoding plus frequency.
4. **Leakage control.** Every encoder is refitted inside each fold on training rows only, and the target encoding of a row uses only transactions that happened before it.
5. **Models.** LightGBM, XGBoost and CatBoost, tuned with Optuna on an expanding time window. The winner is picked by rank over four criteria: average PR AUC, stability between months, result on the last month, and fraud caught at 2% rejection.
6. **Threshold.** Scores are cross-fitted over 15 time blocks, calibrated with Platt scaling, and the cut is chosen where the marginal benefit of rejecting one more percent of traffic drops off.

## Results

XGBoost was selected. On the held out test period:

| Metric | Value |
| --- | ---: |
| PR AUC | 0.152 (about 11x a random model) |
| ROC AUC | 0.862 |
| Decision rule | reject if calibrated fraud probability is at least 13.9% |
| Transactions rejected | 1.28% |
| Rejected transactions that are really fraud | 24.7% |
| Chargebacks | 746 down to 580 (22% fewer) |
| Value lost to fraud | 23% lower |
| Legitimate buyers rejected | 0.98% |

The model sits on top of the existing rules and only scores transactions that those rules already let through. The main drivers by SHAP are the product, account age, card issuer and unit price, which are also the signals that stayed stable month to month in the exploration.

## Limitations

- Model quality cannot be measured on traffic that the existing system rejected.
- The 60 day label window is an approximation of the truth, not the truth.
- `checkout_seconds` depends on an assumption about when `eventTimestamp` is written. The notebook measures what happens without it (section 5.7).
- 93% of the fraud in the test period comes from new customers, where there is no history. User behavior and device features are the obvious next step (section 4.8).

## Reproducibility

The data is confidential and is not part of this repository, so the notebook cannot be re-run from here. All outputs are saved in the notebook, so the full analysis, tables and charts can be read directly on GitHub. `requirements.txt` lists the libraries that were used (Python 3.12).

## Notebook outline

1. Data overview
2. Chargebacks and the definition of the target
3. Train / test split and data exploration
4. Features and model
5. Evaluation and production view
6. Recommendations for production
