"""Python side of the project as a small web server (FastAPI).

Run:   uvicorn api:app --reload --port 8000
Docs:  http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import pipeline as P

SEED = 42
taps = P.simulate_taps(SEED)
scored_days = P.score_rider_days(P.rider_day_features(taps), seed=SEED)
DATES = sorted(taps["date"].unique().tolist())

app = FastAPI(title="Catch Me If You Can: Blue Line station alerts (simulated data)")
# Local demo only: lets the React app on another port call this server.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


@app.get("/api/meta")
def meta():
    return {"line": "Hyderabad Metro Blue Line", "stations": P.STATIONS, "dates": DATES, "simulated": True}


@app.get("/api/cards")
def cards(station: str = Query(...), date: str = Query(...)):
    if station not in P.STATIONS:
        raise HTTPException(404, f"Unknown station: {station}")
    if date not in DATES:
        raise HTTPException(404, f"Unknown date: {date}")
    return {"station": station, "date": date, "cards": P.station_cards(taps, scored_days, station, date)}
