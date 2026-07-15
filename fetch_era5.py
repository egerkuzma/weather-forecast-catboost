"""
Download ERA5 reanalysis (via the free Open-Meteo archive API) for Kursk and a
ring of upwind points, and cache it as era5_grid.csv.gz.

Why a grid and not just neighbouring stations: a 24h forecast is decided by which
air mass arrives, and at a typical 5-7 m/s transport speed air covers 400-600 km
in a day. Sampling every compass bearing at 300 and 600 km therefore brackets the
air that will be over Kursk at target time, whichever way the wind blows.

Run once:  python fetch_era5.py
"""

import gzip
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

OUT_PATH = Path("era5_grid.csv.gz")
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

KURSK_LAT, KURSK_LON = 51.76, 36.19
START_DATE, END_DATE = "2012-09-26", "2025-12-10"
RADII_KM = (300, 600)
BEARINGS = range(0, 360, 45)
HOURLY_VARS = ["temperature_2m", "pressure_msl"]

EARTH_RADIUS_KM = 6371.0


def offset_point(lat: float, lon: float, distance_km: float, bearing_deg: float) -> tuple[float, float]:
    """Great-circle destination point, `distance_km` from (lat, lon) along `bearing_deg`."""
    angular = distance_km / EARTH_RADIUS_KM
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular) + math.cos(lat1) * math.sin(angular) * math.cos(bearing)
    )
    lon2 = math.radians(lon) + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    return round(math.degrees(lat2), 3), round(math.degrees(lon2), 3)


def build_grid() -> list[tuple[str, float, float]]:
    """(name, lat, lon) for the centre plus one ring per radius."""
    grid = [("center", KURSK_LAT, KURSK_LON)]
    for km in RADII_KM:
        for bearing in BEARINGS:
            lat, lon = offset_point(KURSK_LAT, KURSK_LON, km, bearing)
            grid.append((f"{bearing:03d}_{km}", lat, lon))
    return grid


def fetch_point(lat: float, lon: float, retries: int = 4) -> pd.DataFrame:
    query = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    })
    url = f"{ARCHIVE_URL}?{query}"

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                payload = json.loads(response.read())
            hourly = payload["hourly"]
            frame = pd.DataFrame(hourly)
            frame["time"] = pd.to_datetime(frame["time"])
            return frame.set_index("time")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"    retry {attempt + 1}/{retries - 1} after {type(exc).__name__}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main() -> None:
    grid = build_grid()
    print(f"Fetching {len(grid)} points x {len(HOURLY_VARS)} vars, {START_DATE}..{END_DATE}")

    columns: dict[str, pd.Series] = {}
    for i, (name, lat, lon) in enumerate(grid, 1):
        frame = fetch_point(lat, lon)
        columns[f"era5_temp_{name}"] = frame["temperature_2m"]
        columns[f"era5_pressure_{name}"] = frame["pressure_msl"]
        nulls = int(frame["temperature_2m"].isna().sum())
        print(f"  [{i:>2}/{len(grid)}] {name:<9} ({lat:>6.2f},{lon:>6.2f})  "
              f"{len(frame):,} h, {nulls} nulls")
        time.sleep(1)  # stay well inside the free tier's rate limit

    grid_df = pd.DataFrame(columns).sort_index()
    grid_df.index.name = "time"

    with gzip.open(OUT_PATH, "wt") as handle:
        grid_df.to_csv(handle, float_format="%.2f")

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nSaved {OUT_PATH} -- {grid_df.shape[0]:,} hours x {grid_df.shape[1]} cols ({size_mb:.1f} MB)")
    print(f"Range: {grid_df.index.min()} -> {grid_df.index.max()}")


if __name__ == "__main__":
    main()
