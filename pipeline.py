"""Catch Me If You Can: pickpocket-suspect triage on Hyderabad Metro Blue Line (Nagole to Raidurg).

All tap data is SIMULATED. Real rider-level smart-card records are not public.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Blue Line, east to west, 23 stations
STATIONS = [
    "Nagole", "Uppal", "Stadium", "NGRI", "Habsiguda", "Tarnaka", "Mettuguda",
    "Secunderabad East", "Parade Ground", "Paradise", "Rasoolpura", "Prakash Nagar",
    "Begumpet", "Ameerpet", "Madhura Nagar", "Yusufguda", "Jubilee Hills Road No. 5",
    "Jubilee Hills Check Post", "Peddamma Temple", "Madhapur", "Durgam Cheruvu",
    "HITEC City", "Raidurg",
]
# ASSUMED crowded stations (interchanges, rail link, IT hubs). Replace with real ridership if you get it.
HUBS = ["Ameerpet", "Parade Ground", "Secunderabad East", "HITEC City", "Raidurg"]
HUB_IDX = [STATIONS.index(s) for s in HUBS]
NON_HUB_IDX = [i for i in range(len(STATIONS)) if i not in HUB_IDX]
N = len(STATIONS)

DAY_FEATURES = ["trips", "avg_gap_minutes", "avg_stops_per_trip", "hub_tap_share"]
FEATURES = ["trips_per_day", "avg_gap_minutes", "avg_stops_per_trip", "hub_tap_share"]


def _near(rng, origin, lo=1, hi=3):
    """A destination 1-3 stations away from origin, staying on the line."""
    step = int(rng.integers(lo, hi + 1)) * int(rng.choice([-1, 1]))
    dest = origin + step
    return dest if 0 <= dest < N else origin - step


def simulate_taps(seed: int = 42, n_days: int = 7, end_date: str = "2026-10-17") -> pd.DataFrame:
    """One row per trip: rider, day, tap-in minute of day, origin and destination station."""
    rng = np.random.default_rng(seed)
    ids = [f"R{i+1:03d}" for i in range(50)]
    roles = np.array(["suspect"] * 3 + ["busy_normal"] * 3 + ["commuter"] * 44)
    roles = rng.permutation(roles)

    rows = []
    for rid, role in zip(ids, roles):
        if role == "commuter":
            home = int(rng.integers(0, N))
            work = int(rng.choice([i for i in range(N) if abs(i - home) >= 3]))
        for day in range(n_days):
            if role == "suspect" and rng.random() < 0.4:
                continue  # suspects are not on the line every day
            trips = []
            if role == "commuter":
                trips = [(rng.normal(540, 45), home, work), (rng.normal(1110, 60), work, home)]
                if rng.random() < 0.2:
                    a = int(rng.integers(0, N))
                    trips.append((rng.uniform(720, 900), a, _near(rng, a, 1, 6)))
            elif role == "busy_normal":  # e.g. field staff: many short hops, ordinary stations
                n = int(rng.integers(5, 8))
                t = rng.normal(540, 60) + np.cumsum(rng.normal(55, 15, n).clip(30))
                for ti in t:
                    a = int(rng.choice(NON_HUB_IDX))
                    trips.append((ti, a, _near(rng, a)))
            else:  # suspect: rides back and forth around crowded stations at peak hours
                n = int(rng.integers(7, 11)) if rng.random() < 0.5 else int(rng.integers(4, 7))
                start = rng.choice([510, 1050]) + rng.normal(0, 20)
                t = start + np.cumsum(rng.normal(22, 7, n).clip(8))
                for ti in t:
                    a = int(rng.choice(HUB_IDX))
                    trips.append((ti, a, _near(rng, a, 1, 2)))
            for ti, a, d in trips:
                rows.append((rid, day, float(np.clip(ti, 300, 1380)), a, d, role == "suspect"))

    taps = pd.DataFrame(rows, columns=["rider_id", "day", "tap_in_min", "origin", "dest", "is_suspect"])
    dates = pd.date_range(end=end_date, periods=n_days).strftime("%Y-%m-%d")
    taps["date"] = taps["day"].map(lambda d: dates[d])
    taps["origin_station"] = taps["origin"].map(lambda i: STATIONS[i])
    taps["dest_station"] = taps["dest"].map(lambda i: STATIONS[i])
    return taps


def build_features(taps: pd.DataFrame) -> pd.DataFrame:
    n_days = taps["day"].nunique()
    taps = taps.sort_values(["rider_id", "day", "tap_in_min"])
    gaps = taps.groupby(["rider_id", "day"])["tap_in_min"].diff().dropna()
    g = taps.loc[gaps.index].assign(gap=gaps).groupby("rider_id")["gap"].mean()

    by = taps.groupby("rider_id")
    feats = pd.DataFrame({
        "trips_per_day": by.size() / n_days,
        "avg_gap_minutes": g,
        "avg_stops_per_trip": by.apply(lambda d: (d["origin"] - d["dest"]).abs().mean(), include_groups=False),
        "hub_tap_share": by.apply(lambda d: d["origin"].isin(HUB_IDX).mean(), include_groups=False),
        "top_station": by["origin_station"].agg(lambda s: s.value_counts().index[0]),
        "is_suspect": by["is_suspect"].first(),
    }).reset_index()
    return feats.fillna({"avg_gap_minutes": feats["avg_gap_minutes"].max()})


def score_riders(feats: pd.DataFrame, review_budget: int = 4, seed: int = 42) -> pd.DataFrame:
    """Isolation Forest score. review_budget = how many riders analysts can check per day."""
    X = StandardScaler().fit_transform(feats[FEATURES])
    model = IsolationForest(n_estimators=300, contamination=review_budget / len(feats), random_state=seed)
    model.fit(X)
    out = feats.copy()
    out["suspicion_score"] = -model.score_samples(X)  # higher = more unusual
    out["flagged"] = model.predict(X) == -1
    out = out.sort_values("suspicion_score", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def _hhmm(m: float) -> str:
    return f"{int(m // 60):02d}:{int(m % 60):02d}"


def rider_day_features(taps: pd.DataFrame) -> pd.DataFrame:
    """One row per rider per date."""
    taps = taps.sort_values(["rider_id", "date", "tap_in_min"])
    taps = taps.assign(gap=taps.groupby(["rider_id", "date"])["tap_in_min"].diff())
    g = taps.groupby(["rider_id", "date"])
    rd = pd.DataFrame({
        "trips": g.size(),
        "avg_gap_minutes": g["gap"].mean(),
        "avg_stops_per_trip": g.apply(lambda d: (d["origin"] - d["dest"]).abs().mean(), include_groups=False),
        "hub_tap_share": g.apply(lambda d: d["origin"].isin(HUB_IDX).mean(), include_groups=False),
        "is_suspect": g["is_suspect"].first(),
    }).reset_index()
    rd["avg_gap_minutes"] = rd["avg_gap_minutes"].fillna(rd["avg_gap_minutes"].max())
    return rd


MODELS = ["iforest", "lof", "ocsvm", "rule"]


def score_rider_days(rd: pd.DataFrame, alert_rate: float = 0.04, seed: int = 42,
                     cols=None, model: str = "iforest") -> pd.DataFrame:
    """Score every rider-day. model: iforest | lof | ocsvm | rule (rule = just rank by trip count).
    alert_rate = share of rider-days analysts accept as alerts."""
    cols = cols or DAY_FEATURES
    X = StandardScaler().fit_transform(rd[cols])
    n = len(rd)
    if model == "iforest":
        m = IsolationForest(n_estimators=300, contamination=alert_rate, random_state=seed).fit(X)
        score, flagged = -m.score_samples(X), m.predict(X) == -1
    elif model == "lof":
        m = LocalOutlierFactor(n_neighbors=20, contamination=alert_rate)
        flagged = m.fit_predict(X) == -1
        score = -m.negative_outlier_factor_
    elif model == "ocsvm":
        m = OneClassSVM(nu=alert_rate, gamma="scale").fit(X)
        score, flagged = -m.decision_function(X), m.predict(X) == -1
    elif model == "rule":  # dumb baseline: the busiest rider-days are the alerts
        score = rd["trips"].to_numpy(dtype=float) - rd["avg_gap_minutes"].to_numpy() / 1e5  # ties: shorter gap first
        flagged = np.zeros(n, dtype=bool)
        flagged[np.argsort(-score)[: int(round(alert_rate * n))]] = True
    else:
        raise ValueError(f"unknown model {model!r}")

    out = rd.copy()
    out["suspicion_score"] = score
    lo, hi = out["suspicion_score"].min(), out["suspicion_score"].max()
    out["pattern_score"] = ((out["suspicion_score"] - lo) / (hi - lo) * 100).round().astype(int)
    medium_cut = out["suspicion_score"].quantile(0.90)
    out["severity"] = np.where(flagged, "High", np.where(out["suspicion_score"] >= medium_cut, "Medium", "Low"))
    return out


LABELS = {
    "trips": "trips today",
    "avg_gap_minutes": "minutes between taps",
    "avg_stops_per_trip": "stations per trip",
    "hub_tap_share": "share of taps at busy stations",
}


def top_reasons(rows: pd.DataFrame, k: int = 2, ref: pd.DataFrame | None = None,
                always: tuple = ("hub_tap_share",)) -> pd.Series:
    """Plain-language reasons: the k features furthest from typical, plus any in `always`
    (busy-station share is always shown because it separates busy normal riders from suspects)."""
    ref = rows if ref is None else ref
    typical, spread = ref[DAY_FEATURES].median(), ref[DAY_FEATURES].std()
    z = (rows[DAY_FEATURES] - typical) / spread
    texts = []
    for i, row in rows.iterrows():
        picks = list(z.loc[i].abs().sort_values(ascending=False).index[:k])
        picks += [a for a in always if a not in picks]
        parts = []
        for name in picks:
            fmt = (lambda v: f"{v:.0%}") if name == "hub_tap_share" else (lambda v: f"{v:.0f}")
            parts.append(f"{LABELS[name]}: {fmt(row[name])} (typical {fmt(typical[name])})")
        texts.append("; ".join(parts))
    return pd.Series(texts, index=rows.index)


def station_cards(taps: pd.DataFrame, scored_days: pd.DataFrame, station: str, date: str, k: int = 5) -> list[dict]:
    """Cards for ONE station on ONE date: riders who tapped here that day, ranked by their day pattern."""
    idx = STATIONS.index(station)
    day = taps[taps["date"] == date]
    here = day[(day["origin"] == idx) | (day["dest"] == idx)]
    cand = scored_days[(scored_days["date"] == date)
                       & scored_days["rider_id"].isin(here["rider_id"])
                       & (scored_days["severity"] != "Low")]
    cand = cand.sort_values("suspicion_score", ascending=False).head(k)
    med_trips = scored_days[scored_days["date"] == date]["trips"].median()
    cards = []
    why = top_reasons(cand, ref=scored_days) if len(cand) else pd.Series(dtype=str)
    for idx_row, r in cand.iterrows():
        mine = day[day["rider_id"] == r["rider_id"]].sort_values("tap_in_min")
        n_in = int((mine["origin"] == idx).sum())
        n_out = int((mine["dest"] == idx).sum())
        if n_in:
            times = ", ".join(_hhmm(t) for t in mine[mine["origin"] == idx]["tap_in_min"])
            here_txt = f"Tapped in {n_in}x at {station} ({times})"
            here_txt += f", tapped out {n_out}x here." if n_out else "."
        else:
            here_txt = f"Tapped out {n_out}x at {station}."
        summary = f"{here_txt} Why flagged: {why[idx_row]}."
        cards.append({
            "card_id": "C-" + r["rider_id"][1:].zfill(4),
            "severity": r["severity"],
            "pattern_score": int(r["pattern_score"]),
            "trips_today": int(r["trips"]),
            "avg_minutes_between_taps": round(float(r["avg_gap_minutes"]), 1),
            "busy_station_share": round(float(r["hub_tap_share"]), 2),
            "summary": summary,
            "timeline": [{"time": _hhmm(t.tap_in_min), "from": t.origin_station, "to": t.dest_station}
                         for t in mine.itertuples()],
        })
    return cards


def export_json(taps: pd.DataFrame, scored_days: pd.DataFrame, path: str = "station_cards.json") -> None:
    import json
    dates = sorted(taps["date"].unique())
    data = {"line": "Hyderabad Metro Blue Line", "stations": STATIONS, "dates": dates, "simulated": True,
            "cards": {s: {d: station_cards(taps, scored_days, s, d) for d in dates} for s in STATIONS}}
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


if __name__ == "__main__":
    hits = []
    for s in range(10):
        taps = simulate_taps(s)
        scored = score_riders(build_features(taps), seed=s)
        top3 = set(scored.head(3)["rider_id"])
        true = set(scored[scored.is_suspect]["rider_id"])
        hits.append(len(top3 & true))
    print("true suspects in top 3, per seed:", hits, "| mean", np.mean(hits))
    scored = score_riders(build_features(simulate_taps(42)))
    print(scored.head(7)[["rank", "rider_id", *FEATURES, "top_station", "is_suspect"]].round(2).to_string(index=False))