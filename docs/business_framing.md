# Business Framing: 16_weather_prediction

**Status:** Confirmed  
**Mode:** Exploratory  
**Last updated:** 2026-10-20  
**Source:** `notebooks/01_business_framing.ipynb`, written to `ml_spec.yaml`. Edit the notebook, not this file.

## Summary

This is an exploratory project. It should find out: Can I build an end to end online inference system (feature store, champion/challenger gate, monitoring) that serves next day maximum temperature forecasts for 50 cities, with a model that beats a simple persistence forecast (tomorrow's maximum equals the last complete day's maximum)? It ends with one of 3 answers, each leading to a different action. Is ML needed: not sure yet.

## What We Want to Learn

Can I build an end to end online inference system (feature store, champion/challenger gate, monitoring) that serves next day maximum temperature forecasts for 50 cities, with a model that beats a simple persistence forecast (tomorrow's maximum equals the last complete day's maximum)?

## What Each Answer Leads To

| Answer | Action |
|---|---|
| The model clearly beats persistence (for example 10% or more lower error). | Promote it as champion, build the full serving stack around it, and use the gate to judge later challengers. |
| The model only matches persistence. | Still build the system, but serve the persistence forecast as the first champion, then try richer features (lags from nearby cities, hourly data) as challengers. |
| The model is worse than persistence. | Look for leakage or a flawed setup first, and ship nothing until it is explained. |

## Who Uses the Result

The project owner, who builds the system and takes the answer action; portfolio readers see the result but do not act on the forecasts.

## What to Predict

Each city's maximum air temperature (2 m, degrees Celsius) for its next local calendar day, predicted whenever a request arrives, from that city's weather up to and including the last complete day (normally yesterday). The exact target, time window, and metric are defined in the ML problem framing.

## How Success Is Judged

Beat the best simple baseline (persistence from the last complete day) by at least 10% lower error on CV, then promote the model as champion and build the serving stack around it. There is no current approach to beat, so V1 is compared with the best simple baseline.

## Is ML Needed?

Not sure yet. That is what the experiment should find out; a persistence or seasonal baseline may already come close.

## Constraints

* **Latency:** Online, on request. Forecasts are served when a request arrives, not as a nightly batch.
* **Data access:** Open-Meteo free tier only (non commercial, no API key, per minute, hourly, and daily request limits).
* Only the 50 cities in the snapshot; requests for other locations are out of scope in V1.

## Out of Scope

* Competing with professional weather forecasts, which use numerical weather models this project does not.
* Locations other than the 50 cities.
* Forecasts beyond the next day.

## Risks Found During Framing

* The first draft of the goal was only "build the system end to end", which gives the gate and the monitoring no quality bar. The learning question now includes beating a persistence forecast.
* The first draft assumed forecasts are made the evening before, but requests can arrive at any time, when the current day is incomplete. Inputs stop at the last complete day, and training must copy that one day gap so offline results match what the live system can do.
* The model uses only past weather at each city, so it will be far less accurate than professional forecasts. Recorded as out of scope; the comparison that matters is against persistence.

## Assumptions

* The forecast is for the next day only (horizon of one day), for the daily maximum temperature at 2 m.
* The archive API returns the last complete day promptly in production (recent days come from the ECMWF IFS model with no delay; ERA5 runs about 5 days behind).
* Persistence from the last complete day is the naive forecast with a 2 day lag.
* The weather of April to September 2026 is like the weather the system will forecast next.
* The archive's values for past days do not change after they are first published.
* The forecast is for the next calendar day only (one day after the request, two days after the last complete day), for the daily maximum temperature at 2 m.

## Open Questions

| ID | Question | Needed by stage |
|---|---|---|
| Q1 | What does the system serve when the last complete day is not yet in the archive (stale inputs)? | system |
