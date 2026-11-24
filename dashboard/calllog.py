"""Call outcomes and holdout assignments, stored as plain CSV files next to the app.

call_log.csv          one row per logged call (a customer can have several, e.g. No answer then Accepted)
queue_assignments.csv one row per customer the first time they enter a queue: group = call or holdout
"""

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOMES = ["Accepted", "Declined", "No answer"]
LOG_COLUMNS = ["timestamp", "customerID", "rank", "segment", "offer", "outcome", "note", "source"]
ASSIGN_COLUMNS = ["assigned_at", "customerID", "group", "segment", "p_leave", "monthly_bill", "source"]
HOLDOUT_SHARE = 0.10
SALT = "dba3803-holdout-v1"


def _read(path: Path, columns: list) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path, dtype={"customerID": str})
    except Exception:
        return pd.DataFrame(columns=columns)


def load_log(folder: Path) -> pd.DataFrame:
    return _read(folder / "call_log.csv", LOG_COLUMNS)


def append_log(folder: Path, row: dict) -> None:
    path = folder / "call_log.csv"
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)


def latest_status(log: pd.DataFrame) -> pd.Series:
    """Latest outcome per customer."""
    if log.empty:
        return pd.Series(dtype=str)
    return log.sort_values("timestamp").groupby("customerID")["outcome"].last()


def holdout_mask(queue: pd.DataFrame, share: float = HOLDOUT_SHARE) -> np.ndarray:
    """Deterministic random holdout: the same customers are held back every time the queue is built."""
    n = int(round(len(queue) * share))
    if n == 0:
        return np.zeros(len(queue), dtype=bool)
    keys = queue["customerID"].map(lambda c: hashlib.md5(f"{SALT}:{c}".encode()).hexdigest())
    chosen = set(keys.sort_values().index[:n])
    return queue.index.isin(chosen)


def record_assignments(folder: Path, queue: pd.DataFrame, source: str) -> None:
    """Save who was assigned to call vs holdout, once per customer, for later comparison."""
    path = folder / "queue_assignments.csv"
    existing = set(_read(path, ASSIGN_COLUMNS)["customerID"].astype(str))
    new = queue[~queue["customerID"].astype(str).isin(existing)]
    if new.empty:
        return
    out = pd.DataFrame({
        "assigned_at": datetime.now().isoformat(timespec="seconds"),
        "customerID": new["customerID"],
        "group": np.where(new["holdout"], "holdout", "call"),
        "segment": new["segment"],
        "p_leave": new["p_leave"].round(4),
        "monthly_bill": new["MonthlyCharges"],
        "source": source,
    })
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
