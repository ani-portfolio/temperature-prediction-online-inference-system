"""Download a snapshot of daily historical weather from the Open-Meteo archive API.

Run from the project root:
    .venv/bin/python -m src.download_weather --start 2024-01-01 --end 2026-10-01
"""

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

API_URL = "https://archive-api.open-meteo.com/v1/archive"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = RAW_DIR / "open_meteo_cache"
SECONDS_BETWEEN_CITIES = 15  # one city costs ~150 weighted calls; the free tier allows 600 per minute

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "sunshine_duration",
    "weather_code",
]

# (city, country, latitude, longitude): spread across continents and climate zones.
CITIES = [
    # Europe
    ("London", "United Kingdom", 51.5074, -0.1278),
    ("Paris", "France", 48.8566, 2.3522),
    ("Berlin", "Germany", 52.5200, 13.4050),
    ("Madrid", "Spain", 40.4168, -3.7038),
    ("Rome", "Italy", 41.9028, 12.4964),
    ("Stockholm", "Sweden", 59.3293, 18.0686),
    ("Moscow", "Russia", 55.7558, 37.6173),
    ("Istanbul", "Turkey", 41.0082, 28.9784),
    ("Reykjavik", "Iceland", 64.1466, -21.9426),
    ("Athens", "Greece", 37.9838, 23.7275),
    # North America
    ("New York", "United States", 40.7128, -74.0060),
    ("Los Angeles", "United States", 34.0522, -118.2437),
    ("Chicago", "United States", 41.8781, -87.6298),
    ("Houston", "United States", 29.7604, -95.3698),
    ("Phoenix", "United States", 33.4484, -112.0740),
    ("Miami", "United States", 25.7617, -80.1918),
    ("Seattle", "United States", 47.6062, -122.3321),
    ("Denver", "United States", 39.7392, -104.9903),
    ("Anchorage", "United States", 61.2181, -149.9003),
    ("Toronto", "Canada", 43.6532, -79.3832),
    ("Vancouver", "Canada", 49.2827, -123.1207),
    ("Mexico City", "Mexico", 19.4326, -99.1332),
    # South America
    ("Sao Paulo", "Brazil", -23.5505, -46.6333),
    ("Buenos Aires", "Argentina", -34.6037, -58.3816),
    ("Lima", "Peru", -12.0464, -77.0428),
    ("Bogota", "Colombia", 4.7110, -74.0721),
    ("Santiago", "Chile", -33.4489, -70.6693),
    ("Manaus", "Brazil", -3.1190, -60.0217),
    # Africa
    ("Cairo", "Egypt", 30.0444, 31.2357),
    ("Lagos", "Nigeria", 6.5244, 3.3792),
    ("Nairobi", "Kenya", -1.2921, 36.8219),
    ("Johannesburg", "South Africa", -26.2041, 28.0473),
    ("Cape Town", "South Africa", -33.9249, 18.4241),
    ("Casablanca", "Morocco", 33.5731, -7.5898),
    ("Addis Ababa", "Ethiopia", 8.9806, 38.7578),
    # Middle East and Asia
    ("Dubai", "United Arab Emirates", 25.2048, 55.2708),
    ("Riyadh", "Saudi Arabia", 24.7136, 46.6753),
    ("Tehran", "Iran", 35.6892, 51.3890),
    ("Mumbai", "India", 19.0760, 72.8777),
    ("Delhi", "India", 28.6139, 77.2090),
    ("Chennai", "India", 13.0827, 80.2707),
    ("Bangkok", "Thailand", 13.7563, 100.5018),
    ("Singapore", "Singapore", 1.3521, 103.8198),
    ("Jakarta", "Indonesia", -6.2088, 106.8456),
    ("Beijing", "China", 39.9042, 116.4074),
    ("Shanghai", "China", 31.2304, 121.4737),
    ("Tokyo", "Japan", 35.6762, 139.6503),
    ("Seoul", "South Korea", 37.5665, 126.9780),
    # Oceania
    ("Sydney", "Australia", -33.8688, 151.2093),
    ("Auckland", "New Zealand", -36.8485, 174.7633),
]


# Each city's IANA timezone, as Open-Meteo returned it (timezone=auto): the predictor needs the local date to
# know which day is tomorrow without a lookup. The unit test checks it against the raw data.
TIMEZONES = {
    "London": "Europe/London",
    "Paris": "Europe/Paris",
    "Berlin": "Europe/Berlin",
    "Madrid": "Europe/Madrid",
    "Rome": "Europe/Rome",
    "Stockholm": "Europe/Stockholm",
    "Moscow": "Europe/Moscow",
    "Istanbul": "Europe/Istanbul",
    "Reykjavik": "Atlantic/Reykjavik",
    "Athens": "Europe/Athens",
    "New York": "America/New_York",
    "Los Angeles": "America/Los_Angeles",
    "Chicago": "America/Chicago",
    "Houston": "America/Chicago",
    "Phoenix": "America/Phoenix",
    "Miami": "America/New_York",
    "Seattle": "America/Los_Angeles",
    "Denver": "America/Denver",
    "Anchorage": "America/Anchorage",
    "Toronto": "America/Toronto",
    "Vancouver": "America/Vancouver",
    "Mexico City": "America/Mexico_City",
    "Sao Paulo": "America/Sao_Paulo",
    "Buenos Aires": "America/Argentina/Buenos_Aires",
    "Lima": "America/Lima",
    "Bogota": "America/Bogota",
    "Santiago": "America/Santiago",
    "Manaus": "America/Manaus",
    "Cairo": "Africa/Cairo",
    "Lagos": "Africa/Lagos",
    "Nairobi": "Africa/Nairobi",
    "Johannesburg": "Africa/Johannesburg",
    "Cape Town": "Africa/Johannesburg",
    "Casablanca": "Africa/Casablanca",
    "Addis Ababa": "Africa/Addis_Ababa",
    "Dubai": "Asia/Dubai",
    "Riyadh": "Asia/Riyadh",
    "Tehran": "Asia/Tehran",
    "Mumbai": "Asia/Kolkata",
    "Delhi": "Asia/Kolkata",
    "Chennai": "Asia/Kolkata",
    "Bangkok": "Asia/Bangkok",
    "Singapore": "Asia/Singapore",
    "Jakarta": "Asia/Jakarta",
    "Beijing": "Asia/Shanghai",
    "Shanghai": "Asia/Shanghai",
    "Tokyo": "Asia/Tokyo",
    "Seoul": "Asia/Seoul",
    "Sydney": "Australia/Sydney",
    "Auckland": "Pacific/Auckland",
}

def build_url(batch, start, end):
    params = {
        "latitude": ",".join(str(lat) for _, _, lat, _ in batch),
        "longitude": ",".join(str(lon) for _, _, _, lon in batch),
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
    }
    return f"{API_URL}?{urllib.parse.urlencode(params)}"


def fetch_json(url, retries=8):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == retries - 1:
                raise
            reason = error.read().decode(errors="replace")
            wait = 3600 if "hour" in reason.lower() else 65
            print(f"Rate limited ({reason.strip()}); waiting {wait}s", flush=True)
            time.sleep(wait)


def to_frame(city, payload):
    name, country, lat, lon = city
    frame = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    frame.insert(0, "city", name)
    frame.insert(1, "country", country)
    frame.insert(2, "latitude", lat)
    frame.insert(3, "longitude", lon)
    frame.insert(4, "grid_latitude", payload["latitude"])
    frame.insert(5, "grid_longitude", payload["longitude"])
    frame.insert(6, "elevation", payload["elevation"])
    frame.insert(7, "timezone", payload["timezone"])
    return frame


def download(start, end, cached_only=False):
    """Fetch one city per request, caching each response so an interrupted run resumes. With cached_only, use only
    the cities already in the cache and make no requests."""
    cache = CACHE_DIR / f"{start}_{end}"
    cache.mkdir(parents=True, exist_ok=True)
    frames, urls = [], []
    for city in CITIES:
        url = build_url([city], start, end)
        path = cache / f"{city[0].replace(' ', '_').lower()}.json"
        if not path.exists():
            if cached_only:
                continue
            path.write_text(json.dumps(fetch_json(url)))
            print(f"Downloaded {city[0]}", flush=True)
            time.sleep(SECONDS_BETWEEN_CITIES)
        frames.append(to_frame(city, json.loads(path.read_text())))
        urls.append(url)
    return pd.concat(frames, ignore_index=True), urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-10-01")
    parser.add_argument("--cached-only", action="store_true", help="build the table from cached cities only")
    args = parser.parse_args()

    df, urls = download(args.start, args.end, cached_only=args.cached_only)
    stem = f"open_meteo_daily_{args.start}_{args.end}"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / f"{stem}.csv", index=False)
    meta = {
        "source": API_URL,
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "start_date": args.start,
        "end_date": args.end,
        "daily_variables": DAILY_VARIABLES,
        "units": {"temperature": "celsius", "precipitation": "mm", "snowfall": "cm", "wind": "km/h",
                  "shortwave_radiation_sum": "MJ/m2", "sunshine_duration": "seconds"},
        "n_cities": int(df["city"].nunique()),
        "cities_missing": sorted({c[0] for c in CITIES} - set(df["city"])),
        "n_rows": len(df),
        "request_urls": urls,
    }
    (RAW_DIR / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {len(df):,} rows for {df['city'].nunique()} cities to {RAW_DIR / (stem + '.csv')}")


if __name__ == "__main__":
    main()
