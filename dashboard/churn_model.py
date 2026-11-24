"""Scoring logic for the retention dashboard.

Mirrors retention_offers_v3.ipynb:
- same recoding of add-ons and MultipleLines
- same 80/20 split (seed 3803, stratified) so the model is fit on the notebook's training sample
- logistic regression, C = 2.0 (the value chosen on the training folds)
- gender and SeniorCitizen are left out: section 9 showed coverage is unchanged without them
- score = P(leave) x monthly bill; segments use the section 10 priority rules
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 3803
LOGISTIC_C = 2.0
DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]
CATEGORICAL = ["Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               *ADDONS, "Contract", "PaperlessBilling", "PaymentMethod"]
NUMERIC = ["tenure", "MonthlyCharges"]
FEATURES = [*NUMERIC, *CATEGORICAL]

# Allowed values for the single-customer form and for upload validation
CHOICES = {
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    **{a: ["Yes", "No"] for a in ADDONS},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

SEGMENT_ORDER = ["New high-value", "High-risk service", "Stable", "Other"]
SCRIPTS = {
    "New high-value": {"short": "Contract + extras",
                       "offer": "1-year contract with security + tech support included",
                       "avoid": "Don't lead with a discount."},
    "High-risk service": {"short": "Add security & support",
                          "offer": "Turn on security + tech support, with a one-time credit",
                          "avoid": "Don't use the contract script."},
    "Stable": {"short": "No call",
               "offer": "No save call. Renewal reminder only",
               "avoid": "Don't spend a discount here."},
    "Other": {"short": "Review",
              "offer": "Review the account. Neither script fits",
              "avoid": "Don't add extras they already have."},
}

SHORT_VALUE = {
    "Contract": {"Month-to-month": "Month-to-month", "One year": "1-year contract", "Two year": "2-year contract"},
    "InternetService": {"Fiber optic": "Fibre", "DSL": "DSL", "No": "No internet"},
    "PaymentMethod": {"Electronic check": "E-check", "Mailed check": "Mailed check",
                      "Bank transfer (automatic)": "Auto bank transfer", "Credit card (automatic)": "Auto card"},
}
SHORT_BINARY = {
    "OnlineSecurity": ("Has security", "No security"), "TechSupport": ("Has tech support", "No tech support"),
    "OnlineBackup": ("Has backup", "No backup"), "DeviceProtection": ("Has device cover", "No device cover"),
    "StreamingTV": ("Streams TV", "No TV streaming"), "StreamingMovies": ("Streams movies", "No movie streaming"),
    "Partner": ("Has partner", "No partner"), "Dependents": ("Has dependents", "No dependents"),
    "PhoneService": ("Has phone", "No phone"), "MultipleLines": ("Multiple lines", "Single line"),
    "PaperlessBilling": ("Paperless", "Paper bills"),
}

PRETTY = {
    "tenure": "Months with company",
    "MonthlyCharges": "Monthly bill",
    "Partner": "Partner", "Dependents": "Dependents", "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines", "InternetService": "Internet",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection", "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "Contract": "Contract", "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment",
}


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's recoding and return the model columns (plus id / churn if present)."""
    out = frame.copy()
    out.columns = [c.strip() for c in out.columns]
    missing = [c for c in FEATURES if c not in out.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    for column in ADDONS:
        out[column] = np.where(out[column].astype(str).str.strip() == "Yes", "Yes", "No")
    out["MultipleLines"] = np.where(out["MultipleLines"].astype(str).str.strip() == "Yes", "Yes", "No")
    for column in CATEGORICAL:
        out[column] = out[column].astype(str).str.strip()
    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")
    out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    bad = out[NUMERIC].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} row(s) have a blank or non-numeric tenure / MonthlyCharges.")
    if "customerID" not in out.columns:
        out["customerID"] = [f"ROW-{i + 1:05d}" for i in range(len(out))]
    keep = ["customerID", *FEATURES]
    if "Churn" in out.columns:
        out["churn"] = (out["Churn"].astype(str).str.strip() == "Yes").astype(int)
        keep.append("churn")
    return out[keep].reset_index(drop=True)


def segment(frame: pd.DataFrame) -> np.ndarray:
    stable = frame["Contract"].isin(["One year", "Two year"]) & (frame["tenure"] >= 24)
    new_hv = (frame["Contract"] == "Month-to-month") & (frame["tenure"] <= 12) & (frame["MonthlyCharges"] >= 70)
    risk_svc = ((frame["InternetService"] == "Fiber optic") & (frame["OnlineSecurity"] == "No")
                & (frame["TechSupport"] == "No"))
    return np.select([stable, new_hv, risk_svc], ["Stable", "New high-value", "High-risk service"], default="Other")


class RetentionModel:
    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir).resolve()
        for folder in (data_dir, data_dir.parent):
            if (folder / DATA_FILE).exists():
                raw = pd.read_csv(folder / DATA_FILE)
                break
        else:
            raise FileNotFoundError(DATA_FILE)
        data = clean(raw)
        X, y = data[FEATURES], data["churn"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        pre = ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])
        self.pipe = Pipeline([("preprocess", pre),
                              ("model", LogisticRegression(max_iter=2000, solver="lbfgs", C=LOGISTIC_C))])
        self.pipe.fit(X_train, y_train)

        self.feature_names = list(self.pipe.named_steps["preprocess"].get_feature_names_out())
        self.coef = self.pipe.named_steps["model"].coef_[0]
        Z_train = self.pipe.named_steps["preprocess"].transform(X_train)
        self.z_mean = Z_train.mean(axis=0)

        train_score = self.pipe.predict_proba(X_train)[:, 1] * X_train["MonthlyCharges"].to_numpy()
        self.train_scores = np.sort(train_score)
        self.cutoff_10 = float(np.quantile(train_score, 0.90))
        self.n_train = len(X_train)

        test = data.loc[X_test.index].copy()
        self.test_sample = raw.loc[X_test.index].reset_index(drop=True)
        self.test_frame = test.reset_index(drop=True)

    # ---- scoring -------------------------------------------------------
    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["p_leave"] = self.pipe.predict_proba(out[FEATURES])[:, 1]
        out["revenue_at_risk"] = out["p_leave"] * out["MonthlyCharges"]
        out["train_percentile"] = np.searchsorted(self.train_scores, out["revenue_at_risk"], side="right") / len(self.train_scores)
        out["segment"] = segment(out)
        out["drivers"] = self.drivers(out[FEATURES])
        out = out.sort_values("revenue_at_risk", ascending=False, kind="mergesort").reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Log-odds contribution of each encoded column relative to the average training customer."""
        Z = self.pipe.named_steps["preprocess"].transform(X)
        return (Z - self.z_mean) * self.coef

    def _label(self, name: str, row: pd.Series) -> str:
        block, col = name.split("__", 1)
        if block == "numeric":
            if col == "tenure":
                return f"{int(row['tenure'])} mo"
            return f"${row['MonthlyCharges']:.0f} bill"
        base = next(c for c in CATEGORICAL if col.startswith(c + "_"))
        value = row[base]
        if base in SHORT_BINARY:
            return SHORT_BINARY[base][0 if value == "Yes" else 1]
        return SHORT_VALUE[base].get(value, value)

    def drivers(self, X: pd.DataFrame, top: int = 3) -> list:
        contrib = self.contributions(X)
        result = []
        for i in range(len(X)):
            row = X.iloc[i]
            # merge encoded columns back to their source field
            by_field = {}
            for j, name in enumerate(self.feature_names):
                col = name.split("__", 1)[1]
                field = col if col in NUMERIC else next(c for c in CATEGORICAL if col.startswith(c + "_"))
                by_field.setdefault(field, [0.0, name])
                by_field[field][0] += contrib[i, j]
            # The bill already enters the score directly, and its coefficient is not stable on
            # its own (notebook section 9), so it is not shown as a reason to the agent.
            by_field.pop("MonthlyCharges", None)
            ranked = sorted(by_field.items(), key=lambda kv: kv[1][0], reverse=True)
            labels = [self._label(v[1], row) for f, v in ranked[:top] if v[0] > 0.05]
            result.append(" · ".join(labels) if labels else "No strong signal")
        return result

    def field_contributions(self, X_row: pd.DataFrame) -> pd.DataFrame:
        contrib = self.contributions(X_row)[0]
        rows = {}
        for j, name in enumerate(self.feature_names):
            col = name.split("__", 1)[1]
            field = col if col in NUMERIC else next(c for c in CATEGORICAL if col.startswith(c + "_"))
            rows[field] = rows.get(field, 0.0) + contrib[j]
        row = X_row.iloc[0]
        out = pd.DataFrame([
            {"field": PRETTY[f], "value": (f"{row[f]:.0f}" if f in NUMERIC else row[f]), "effect": v}
            for f, v in rows.items()
        ])
        return out.reindex(out["effect"].abs().sort_values(ascending=False).index).reset_index(drop=True)
