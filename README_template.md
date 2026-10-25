# Catch Me If You Can: Pickpocket-Suspect Triage on the Hyderabad Metro Blue Line

<!-- WRITE: 2 sentences. What it does, for whom. Add the live dashboard link and the other repo link. -->

**Live dashboard:** <Lovable published link> | **Dashboard code:** https://github.com/chaithanyareddy2017/rider-rhythm-insights

> All tap data is **simulated**. Real rider-level smart-card records are not public.

## 1. Problem
<!-- WRITE: why pickpocketing is hard to spot in transit data, why a human-review triage list (not automatic accusations) is the right goal. 4-5 lines. -->

## 2. Architecture
```
Simulator -> rider-day features -> anomaly model -> explanations -> FastAPI server -> React dashboard (Lovable)
```
<!-- WRITE: one sentence per box. Name the files: pipeline.py, api.py, evaluate.py, explain.py. -->

## 3. Data (simulated)
<!-- WRITE: 23 Blue Line stations Nagole to Raidurg, 50 riders, 7 days, 3 suspects, 3 busy normal riders,
     commuters. Say suspects are half obvious (7-10 trips) and half subtle (4-6 trips). Say which stations
     are assumed busy and that this is an assumption. -->

## 4. Features (per rider per day)
<!-- WRITE: a 4-row table: trips, avg_gap_minutes, avg_stops_per_trip, hub_tap_share, each with one line on why it matters. -->

## 5. Models and results
Alert rate fixed at 4% of rider-days (analyst review capacity). Mean over 20 seeds.

**Easy suspects (all suspect days have 7-10 trips)**
| Model | Precision | Recall |
|---|---|---|
| Isolation Forest | 0.650 | 0.750 |
| Local Outlier Factor | 0.511 | 0.614 |
| One-Class SVM | 0.172 | 0.216 |
| Rule: most trips | 0.876 | 0.971 |

**Subtle suspects (half of suspect days have 4-6 trips)**
| Model | Precision | Recall |
|---|---|---|
| Isolation Forest | 0.739 | 0.826 |
| Local Outlier Factor | 0.461 | 0.522 |
| One-Class SVM | 0.226 | 0.233 |
| Rule: most trips | 0.557 | 0.612 |

<!-- WRITE: 4-5 lines. Why the rule won on easy data, why that is the reason you changed the simulator,
     and why Isolation Forest wins on subtle data. Add the +/- std values from your own run. -->

## 6. Ablation
<!-- WRITE: re-run evaluate.py WITHOUT hub_tap_share on the SUBTLE data and fill this in.
     Earlier run on easy data: all features 0.650 +/- 0.083 precision, without hub_tap_share 0.546 +/- 0.093. -->

## 7. Explanations
<!-- WRITE: show 2 example cards: one real suspect and one false alarm (a busy normal rider with 0% busy-station share).
     Explain how the "Why flagged" text is built (z-score vs typical rider-day). -->

## 8. Limitations
<!-- WRITE: be blunt. At least: (1) data is simulated from rules close to the model's features, so results are optimistic;
     (2) about 3 in 10 High alerts are false alarms, e.g. long end-to-end commuters and busy normal riders;
     (3) "busy stations" are assumed; (4) alert rate is a staffing choice, not learned. -->

## 9. Privacy and ethics
<!-- WRITE: a score is a lead for a human, not proof of theft; real tap records are personal data (India's DPDP Act, 2023):
     data minimisation, purpose limitation, retention limits, no automated action against a rider;
     bias risk (commuters with unusual but legitimate patterns, shift workers, delivery staff); audit log of reviews. -->

## 10. How to run
```
pip install -r requirements.txt
python evaluate.py                         # metrics over 20 seeds
uvicorn api:app --reload --port 8000       # Python API, docs at /docs
python export_cards.py                     # regenerate the bundled JSON
```
Dashboard: in the React repo run `npm install`, create `.env.local` with `VITE_API_URL=http://localhost:8000`, then `npm run dev`.

## 11. Future work
<!-- WRITE: 3 bullets. Real HMRL data under an agreement, per-rider baselines, trajectory/graph models, drift monitoring. -->
