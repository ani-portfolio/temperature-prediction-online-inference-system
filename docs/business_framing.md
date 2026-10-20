# Business Framing: 16_weather_prediction

**Status:** Draft  
**Mode:** Exploratory  
**Last updated:** 2026-10-20  
**Source:** `notebooks/01_business_framing.ipynb`, written to `ml_spec.yaml`. Edit the notebook, not this file.

## Summary

This is an exploratory project. It should find out: Can I build an end to end online inference system (feature store, champion/challenger gate, monitoring) that serves next day maximum temperature forecasts for 50 cities, with a model that beats a simple persistence forecast (tomorrow's maximum equals today's maximum)? It ends with one of 3 answers, each leading to a different action.

## What We Want to Learn

Can I build an end to end online inference system (feature store, champion/challenger gate, monitoring) that serves next day maximum temperature forecasts for 50 cities, with a model that beats a simple persistence forecast (tomorrow's maximum equals today's maximum)?

## What Each Answer Leads To

| Answer | Action |
|---|---|
| The model clearly beats persistence (for example 10% or more lower error). | Promote it as champion, build the full serving stack around it, and use the gate to judge later challengers. |
| The model only matches persistence. | Still build the system, but serve the persistence forecast as the first champion, then try richer features (lags from nearby cities, hourly data) as challengers. |
| The model is worse than persistence. | Look for leakage or a flawed setup first, and ship nothing until it is explained. |

## Who Uses the Result

Not decided yet.

## What to Predict

Not decided yet. The exact target, time window, and metric are defined in the ML problem framing.

## How Success Is Judged

Not decided yet. There is no current approach to beat, so V1 is compared with the best simple baseline.

## Is ML Needed?

Not decided yet.

## Constraints

None recorded.

## Out of Scope

Nothing listed yet.

## Risks Found During Framing

* The first draft of the goal was only "build the system end to end", which gives the gate and the monitoring no quality bar. The learning question now includes beating a persistence forecast.

## Assumptions

* The forecast is for the next day only (horizon of one day), for the daily maximum temperature at 2 m.

## Open Questions

None.
