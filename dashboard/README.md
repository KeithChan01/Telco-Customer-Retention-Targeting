# Retention Desk (DBA3803 dashboard)

A local app that turns the `retention_offers_analysis.ipynb` model into a call queue for customer service.

## Run it (Windows)

Double-click `run_dashboard.bat`. The first run installs the packages, then the app opens in your browser.
Or in a terminal in this folder: `pip install -r requirements.txt` then `streamlit run app.py`.

The app reads `WA_Fn-UseC_-Telco-Customer-Churn.csv` from this folder or the Project folder above it and fits the
model at start-up (under a second).

## Pages

- **Call queue** — customers ranked by chance of leaving × monthly bill. Click a row to open the customer: leave
  risk, the offer, why they were flagged, and buttons to log the call (Accepted / Declined / No answer). Logging
  moves straight to the next customer. "No answer" goes back to the end of the To-call list.
- **Quick score** — score one customer as you type.
- **Planning** — quota and cost assumptions, net value by quota, and call outcomes so far.

## Files the app writes

| File | What's in it |
|---|---|
| `call_log.csv` | One row per logged call: time, customer, segment, offer, outcome, note |
| `queue_assignments.csv` | Each customer the first time they enter a queue, marked `call` or `holdout` |

About 10% of each queue is held back as a control group and never shown to agents. The choice is random but fixed
(the same customers are held back each time). At the next data extract, compare who left in the `call` and
`holdout` groups to measure whether calling keeps customers. That is the uplift data the notebook (§12) says is
missing.

## Matches the notebook

Same recoding, same 80/20 split (seed 3803), logistic regression C = 2.0, `gender` and `SeniorCitizen` left out,
§10 segment rules. At a 10% quota on the test group the queue reaches 35% of lost monthly bills.

## Code

| File | Purpose |
|---|---|
| `app.py` | Streamlit app (three pages) |
| `churn_model.py` | Cleaning, model fit, scoring, segments, reasons |
| `calllog.py` | Call log, holdout assignment |
| `logo.svg`, `.streamlit/config.toml` | Logo and theme |
