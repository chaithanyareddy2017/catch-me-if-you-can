"""Step 3: tell the officer WHY a rider-day was flagged.  YOU fill in the TODOs."""
import pandas as pd

from pipeline import DAY_FEATURES, rider_day_features, score_rider_days, simulate_taps

LABELS = {
    "trips": "trips today",
    "avg_gap_minutes": "minutes between taps",
    "avg_stops_per_trip": "stations per trip",
    "hub_tap_share": "share of taps at busy stations",
}


def top_reasons(scored: pd.DataFrame, k: int = 2) -> pd.Series:
    """One text per rider-day, e.g. 'trips today: 9 (typical 3); minutes between taps: 24 (typical 560)'."""
    typical = scored[DAY_FEATURES].median()
    spread = scored[DAY_FEATURES].std()

    z = (scored[DAY_FEATURES] - typical) / spread

    texts = []
    for i, row in scored.iterrows():
        picks = z.loc[i].abs().sort_values(ascending=False).index[:k]
        parts = []
        for name in picks:
            if name == "hub_tap_share":
                parts.append(f"{LABELS[name]}: {row[name]:.0%} (typical {typical[name]:.0%})")
            else:
                parts.append(f"{LABELS[name]}: {row[name]:.0f} (typical {typical[name]:.0f})")
        texts.append("; ".join(parts))
    return pd.Series(texts, index=scored.index)


if __name__ == "__main__":
    scored = score_rider_days(rider_day_features(simulate_taps(42)))
    scored["reasons"] = top_reasons(scored)
    high = scored[scored["severity"] == "High"]
    print(high[["rider_id", "date", "reasons"]].head(8).to_string(index=False))
