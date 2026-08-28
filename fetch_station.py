"""
Download the Kursk (UUOK) METAR archive from the Iowa State ASOS service and
cache it as station.csv.gz.

Why not RP5: the RP5 export is only reachable through a web form, so `data.csv`
could not be reproduced by a script -- and the file it produces is stamped in
*local* time, which changed from UTC+4 to UTC+3 in 2014. The same METAR reports
are served over a plain URL here, in UTC, for the whole period. That removes a
manual step and a timezone discontinuity at once.

Run once:  python fetch_station.py
"""

import gzip
import io
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

OUT_PATH = Path("station.csv.gz")
SERVICE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

STATION = "UUOK"
START_DATE, END_DATE = "2012-09-26", "2025-12-10"

# METAR is reported in Fahrenheit / knots / inches of mercury by this service.
F_TO_C_OFFSET, F_TO_C_SCALE = 32.0, 1.8
KNOTS_TO_MS = 0.514444
INHG_TO_HPA = 33.8639


def fetch_raw() -> pd.DataFrame:
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    query = urllib.parse.urlencode(
        [
            ("station", STATION),
            ("data", "tmpf"),
            ("data", "dwpf"),
            ("data", "drct"),
            ("data", "sknt"),
            ("data", "alti"),
            ("data", "relh"),
            ("year1", start.year), ("month1", start.month), ("day1", start.day),
            ("year2", end.year), ("month2", end.month), ("day2", end.day),
            ("tz", "UTC"),
            ("format", "onlycomma"),
            ("missing", "empty"),
            ("trace", "empty"),
            ("latlon", "no"),
            # 3 = routine hourly METAR, 4 = SPECI. Both are kept; the hourly grid
            # downstream decides what to do with the extra reports.
            ("report_type", "3"), ("report_type", "4"),
        ]
    )
    with urllib.request.urlopen(f"{SERVICE_URL}?{query}", timeout=900) as response:
        payload = response.read().decode()
    return pd.read_csv(io.StringIO(payload))


def main() -> None:
    print(f"Fetching {STATION} METAR {START_DATE}..{END_DATE} (UTC)")
    raw = fetch_raw()

    df = pd.DataFrame({"time": pd.to_datetime(raw["valid"])})
    df["temp"] = (raw["tmpf"] - F_TO_C_OFFSET) / F_TO_C_SCALE
    df["dew_point"] = (raw["dwpf"] - F_TO_C_OFFSET) / F_TO_C_SCALE
    df["pressure"] = raw["alti"] * INHG_TO_HPA
    df["humidity"] = raw["relh"]
    df["wind_speed"] = raw["sknt"] * KNOTS_TO_MS
    df["wind_dir"] = raw["drct"]

    df = df.dropna(subset=["temp"]).sort_values("time").set_index("time")

    with gzip.open(OUT_PATH, "wt") as handle:
        df.to_csv(handle, float_format="%.2f")

    print(f"Saved {OUT_PATH} -- {len(df):,} reports, "
          f"{df.index.min()} -> {df.index.max()} "
          f"({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
