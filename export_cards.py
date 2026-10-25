"""Regenerate station_cards.json (compact) for the Lovable app: python export_cards.py"""
import json

import pipeline as P

taps = P.simulate_taps(42)
scored_days = P.score_rider_days(P.rider_day_features(taps), seed=42)
P.export_json(taps, scored_days, "station_cards.json")

with open("station_cards.json") as f:
    data = json.load(f)
with open("station_cards.json", "w") as f:
    json.dump(data, f, separators=(",", ":"))

n = sum(1 for s in data["cards"].values() for day in s.values() if day)
print(f"Wrote station_cards.json: {len(data['stations'])} stations, {len(data['dates'])} dates, {n} station-days with cards")
