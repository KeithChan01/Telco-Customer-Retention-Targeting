# Telco Customer Retention Targeting

Churn model + profit simulation + Streamlit dashboard for prioritising telco retention calls under a contact limit.

A phone and internet company cannot call, or discount, every customer who might leave. If it can contact only
**10%** of its customers, who should they be so that as much monthly revenue as possible is protected?

Each customer is ranked by

```
revenue at risk = P(leave) × monthly bill
```

and the top of that list becomes the call queue.

## What's in the project

| Path | What it is |
|---|---|
| `retention_offers_analysis.ipynb` | Full analysis: exploration, feature tests, model comparison, segments, profit simulation, recommendation |
| `dashboard/` | **Retention Desk**, a Streamlit app that turns the model into a working call queue ([details](dashboard/README.md)) |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | IBM Telco Customer Churn sample data (7,043 customers) |
| `Project Report.pdf` | Written report |
| `Presentation.pdf` | Presentation slides |
| `requirements.txt` | Packages for the notebook |

## Approach

1. **Explore and prepare.** Drop `customerID` and `TotalCharges` (near-duplicate of tenure × bill), recode optional
   extras to Yes/No, and make a stratified 80/20 split (seed 3803) before fitting anything.
2. **Feature engineering.** Test price change, tenure × contract, log tenure, tenure bands, add-on counts and
   auto-pay. A feature is kept only if it lifts 10% lost-bill coverage on the training folds by 2+ points. None did.
3. **Baselines.** Four rules a manager could run without a model (e.g. newest month-to-month first, contract
   leave rate × bill, five-flag count × bill).
4. **Models.** Logistic regression, random forest, XGBoost and a neural network, compared on repeated training
   folds and then scored once on the 1,409-customer test group, with bootstrap intervals.
5. **Interpretation.** Permutation importance, coefficients and odds ratios, SHAP, a multicollinearity check, and a
   refit without `gender` and `SeniorCitizen`.
6. **Segments and offers.** *Stable*, *New high-value* (month-to-month, ≤12 months, bill ≥ $70) and unprotected
   fibre customers, each with its own retention script.
7. **Profit simulation.** Expected value per call given save rate, months retained, call cost and discount, and
   the quota that maximises net return.

## Key findings

- **Logistic regression is the recommended model.** No more complex model beat it by the pre-set 2-point margin,
  and the fold intervals for the challengers all span zero. Its real edge is over the simple manager rules.
- At a 10% quota on the test group the queue reaches **about 35% of lost monthly bills**
  (95% bootstrap interval 32–39%; 33% on the training folds).
- `gender` and `SeniorCitizen` can be dropped without reducing coverage, so the deployed model leaves them out.
- **The best quota depends on how long a saved customer stays.** At one month no quota pays; at about three months
  ~10% is right; at six months or more the profit-maximising quota is well above 10%, so the 10% limit becomes a
  staff-capacity constraint rather than an economic one.
- Judged on profit instead of coverage, the model choice does not change.

### Limitations

The score finds customers who **look like past leavers**, not customers who will **stay because of an offer**. The
data has no record of past offers or responses, so uplift cannot be modelled. It is also a single snapshot with no
dates, so the test group is not "future" customers. The dashboard's built-in ~10% holdout group is designed to
collect the uplift data needed for the next stage.

## Getting started

Requires Python 3.10+ (tested on 3.13).

```bash
git clone https://github.com/<your-username>/Telco-Customer-Retention-Targeting.git
cd Telco-Customer-Retention-Targeting
```

### Run the notebook

```bash
pip install -r requirements.txt
jupyter notebook retention_offers_analysis.ipynb
```

### Run the dashboard

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

On Windows you can double-click `dashboard/run_dashboard.bat` instead. The app reads the churn CSV from the project
root and fits the model at start-up (under a second).

The dashboard has three pages:

- **Call queue**: customers ranked by risk × bill, with the reason each was flagged, the suggested offer, and
  buttons to log the outcome (Accepted / Declined / No answer).
- **Quick score**: score a single customer as you type.
- **Planning**: quota and cost assumptions, net value by quota, and call outcomes so far.

## Data

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), a teaching sample of
7,043 customers with demographics, services, contract, billing and a `Churn` flag. It has no complaints, outages,
competitor offers or sign-up/leave dates.
