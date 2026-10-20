# Model Card: 16_weather_prediction

**Evaluated:** 2026-10-20  
**Source:** `notebooks/09_evaluation.ipynb`, written from `ml_spec.yaml`. Rerun the notebook to update it.

## Purpose

An exploratory project. It should find out: Can I build an end to end online inference system (feature store, champion/challenger gate, monitoring) that serves next day maximum temperature forecasts for 50 cities, with a model that beats a simple persistence forecast (tomorrow's maximum equals the last complete day's maximum)?

| Answer | Action |
|---|---|
| The model clearly beats persistence (for example 10% or more lower error). | Promote it as champion, build the full serving stack around it, and use the gate to judge later challengers. |
| The model only matches persistence. | Still build the system, but serve the persistence forecast as the first champion, then try richer features (lags from nearby cities, hourly data) as challengers. |
| The model is worse than persistence. | Look for leakage or a flawed setup first, and ship nothing until it is explained. |

There is no current approach, so the champion is compared with the best simple baseline. The V1 check in the evidence below says which answer the result supports, and the table above says what follows from it.

## Model

* **Champion:** lightgbm (lightgbm), chosen by its mean CV score in model training.
* **Predicts:** The city's maximum air temperature at 2 m (degrees Celsius) on the target date, its local calendar day.
* **Unit and timing:** One city on one target date (the city's next local day, seen from a request); Whenever a request arrives during the day before the target date. The current day is incomplete, so inputs stop at the last complete day, two days before the target date.
* **Inputs:** 27 features. Excluded: date, grid_latitude, grid_longitude, country, timezone, rain_sum, temperature_2m_min, temperature_2m_mean, apparent_temperature_max, apparent_temperature_min, precipitation_sum, snowfall_sum, wind_speed_10m_max, wind_gusts_10m_max, wind_direction_10m_dominant, shortwave_radiation_sum, sunshine_duration, weather_code.
* **Saved model:** `models/champion_lightgbm.joblib`

## Data

* **Split:** time; test set of 9,100 rows, 2026-04-03 to 2026-10-01.

## Results on the Test Set

Primary metric: mae. Intervals are 95% bootstrap intervals, resampling whole date values.

| Model | Kind | Test | 95% interval | CV |
|---|---|---|---|---|
| lightgbm (champion) | champion | 2.0799 | 2.0376 to 2.1269 | 2.1630 |
| ridge | ml | 2.1460 | 2.1005 to 2.1922 | 2.2301 |
| naive | baseline | 2.4434 | 2.3944 to 2.5112 | 2.5156 |
| seasonal_naive | baseline | 3.2353 | 3.1749 to 3.3064 | 3.3270 |
| moving_average | baseline | 2.6066 | 2.5247 to 2.6745 | 2.7740 |

## Comparisons

Paired bootstrap differences; positive is better, also for error metrics.

| Comparison | Difference | 95% interval | Relative |
|---|---|---|---|
| lightgbm against naive | +0.3635 | +0.3228 to +0.4058 | +14.9% |

## Evidence

| Question | Answer |
|---|---|
| Does the champion beat naive on the test set? | Yes: by +0.3635 (+14.9%), 95% interval +0.3228 to +0.4058; the interval is above zero. |
| Is the V1 bar met? | met. On CV: met on cv (beats_best_baseline_cv pass). On test: met on test (beats_best_baseline_test pass). |
| How does the test score compare with CV? | Test 2.0799, CV 2.1630 (-0.0831). Close, so the CV estimate held. |
| Is any other model better on the test set? | No: the best other is ridge with 2.1460. The champion was fixed before the test set was opened; switching now would make its test score optimistic. |
| Are there weak slices? | 16: city = Beijing, city = Berlin, city = Cape Town, city = Chicago, city = Denver, city = London, city = Madrid, city = Moscow, city = New York, city = Paris, city = Santiago, city = Sao Paulo, city = Seattle, city = Tokyo, city = Toronto, latitude (quantiles of training) = (41.008, inf]. |
| Has the data shifted? | Prediction drift (PSI) 0.311, a large shift; drift of the actual values 0.256, a large shift. The actual values moved as well, so the shift comes from the data; the bias metric shows whether the model kept up. |

## Slices

* city = Beijing: champion 2.7588, reference 3.1511 (far below the overall score)
* city = Berlin: champion 3.2106, reference 4.0214 (far below the overall score)
* city = Cape Town: champion 2.6407, reference 3.2808 (far below the overall score)
* city = Chicago: champion 3.3793, reference 4.5104 (far below the overall score)
* city = Denver: champion 3.7027, reference 4.5923 (far below the overall score)
* city = London: champion 2.7452, reference 3.0582 (far below the overall score)
* city = Madrid: champion 2.6465, reference 3.0951 (far below the overall score)
* city = Moscow: champion 3.3172, reference 3.7357 (far below the overall score)
* city = New York: champion 3.5408, reference 4.2819 (far below the overall score)
* city = Paris: champion 3.2318, reference 3.4148 (far below the overall score)
* city = Santiago: champion 2.8661, reference 3.6445 (far below the overall score)
* city = Sao Paulo: champion 2.7896, reference 3.5808 (far below the overall score)
* city = Seattle: champion 2.7141, reference 3.3874 (far below the overall score)
* city = Tokyo: champion 2.7341, reference 3.1819 (far below the overall score)
* city = Toronto: champion 3.0691, reference 3.6753 (far below the overall score)
* latitude (quantiles of training) = (41.008, inf]: champion 2.6596, reference 3.119 (far below the overall score)

## Limitations and Assumptions

* The forecast is for the next day only (horizon of one day), for the daily maximum temperature at 2 m.
* The archive API returns the last complete day promptly in production (recent days come from the ECMWF IFS model with no delay; ERA5 runs about 5 days behind).
* Persistence from the last complete day is the naive forecast with a 2 day lag.
* The weather of April to September 2026 is like the weather the system will forecast next.
* The archive's values for past days do not change after they are first published.

## Open Questions

* Q1: What does the system serve when the last complete day is not yet in the archive (stale inputs)?

## Decision

Not recorded yet. The evidence above is for the model owner to decide go, iterate, or no go; record the decision with its reason in the decisions cell of the evaluation notebook.
<!-- written by ml-offline-evaluation. If you edit this card, the next run keeps it and writes model_card_new.md: 676fd9e1efa9cc49 -->
