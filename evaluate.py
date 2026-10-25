"""Step 1: how well do High alerts match the true suspect rider-days?  YOU fill in the TODOs."""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

from pipeline import DAY_FEATURES, MODELS, rider_day_features, score_rider_days, simulate_taps

SEEDS = range(20)


def run_one(seed: int, cols: list[str], model: str = "iforest") -> dict:
    taps = simulate_taps(seed)
    scored = score_rider_days(rider_day_features(taps), cols=cols, seed=seed, model=model)

    y_true = scored["is_suspect"]
    y_pred = scored["severity"] == "High"
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "seed": seed,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def run_all(cols: list[str], model: str = "iforest") -> pd.DataFrame:
    rows = [run_one(seed, cols, model) for seed in SEEDS]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = {}
    for m in MODELS:
        df = run_all(DAY_FEATURES, model=m)
        results[m] = df[["precision", "recall"]].mean()
    print(pd.DataFrame(results).T.round(3))

