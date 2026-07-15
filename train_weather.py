"""
Forecast Kursk airport temperature 24 hours ahead with CatBoost.

Data: an RP5 METAR export (data.csv, semicolon-delimited with quoted fields),
optionally joined with an ERA5 reanalysis grid (era5_grid.csv.gz, see fetch_era5.py).

Note on the station: Kursk airport reports roughly hourly by day but goes quiet
for ~6h most nights. The hourly grid is therefore interpolated to keep lags
well-defined, but training and scoring are restricted to rows where the
temperature was genuinely measured -- both at feature time and at target time.

WARNING -- the ERA5 features are NOT deployable, they are a research upper bound:

  * ERA5 is published with a ~5 day lag, so at real forecast time t its values
    simply do not exist yet.
  * ERA5 is a reanalysis built in 12h 4D-Var windows, so the field at time t
    already absorbs observations from up to ~9h *after* t -- part of the very
    interval being forecast.

Measured test MAE: station-only 2.344, +ERA5 at t 2.014, +ERA5 lagged 12h 2.339.
The entire ERA5 gain disappears once it is lagged, i.e. it comes from being
contemporaneous, not from upwind advection signal. See ablation.py. For a live
forecaster, replace these with actual NWP forecast output (model output
statistics) or real-time neighbouring station observations.

Writes artifacts/ for plots.py to render; run `python plots.py` afterwards.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("data.csv")
ERA5_PATH = Path("era5_grid.csv.gz")  # optional; produced by fetch_era5.py
ARTIFACT_DIR = Path("artifacts")

VAL_START = pd.Timestamp("2023-01-01")  # tail of train, held out for early stopping
TEST_START = pd.Timestamp("2024-01-01")
HORIZON = 24  # hours ahead to forecast
INTERP_LIMIT = 8  # max consecutive hours to interpolate (covers the ~6h night gap)

METEO_COLS = [
    "temp",
    "pressure",
    "humidity",
    "wind_speed",
    "dew_point",
    "wind_u",
    "wind_v",
]
OBSERVED = "temp_observed"
TARGET = "target"
TARGET_OBSERVED = "target_observed"
NON_FEATURE_COLS = {OBSERVED, TARGET, TARGET_OBSERVED}

# Columns a row must have to be trainable. The rest (cloud, wind vector,
# visibility) may be NaN -- CatBoost splits on missingness natively, so dropping
# those rows would cost data for no gain.
REQUIRED_COLS = ["temp", "pressure", "temp_lag_1", "temp_lag_24", "pressure_lag_24"]

# RP5 spells the 16-point compass out in prose. DD is the direction the wind
# blows *from*, in degrees clockwise from north.
WIND_AZIMUTH = {
    "north": 0,
    "north-northeast": 22.5,
    "north-east": 45,
    "east-northeast": 67.5,
    "east": 90,
    "east-southeast": 112.5,
    "south-east": 135,
    "south-southeast": 157.5,
    "south": 180,
    "south-southwest": 202.5,
    "south-west": 225,
    "west-southwest": 247.5,
    "west": 270,
    "west-northwest": 292.5,
    "north-west": 315,
    "north-northwest": 337.5,
}

# Cloud cover (c) and visibility (VV) are parsed out of the RP5 export in earlier
# revisions and were dropped: CatBoost importance ~0.3, and removing them moves
# test MAE by 0.005 against a seed standard deviation of 0.008 -- i.e. nothing,
# for ~25 lines of free-text parsing.

# Kursk keeps Moscow time, which was UTC+4 until 26.10.2014 02:00 MSK and UTC+3
# after. data.csv is stamped in local time, ERA5 in UTC, so the offset both
# matters and *changes mid-dataset*. Verified empirically: correlation between
# station and ERA5 centre temperature peaks at exactly these offsets (0.988/0.990).
MSK_SWITCH_UTC = pd.Timestamp("2014-10-25 22:00")


def parse_wind(direction: pd.Series, speed: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert prose wind direction + speed into u/v vector components.

    Advection depends on direction and strength together, and a raw azimuth in
    degrees wraps discontinuously at 0/360. Components avoid both problems.
    "Calm" is a true zero vector; "variable" leaves direction undefined (NaN).
    """
    name = direction.str.extract(r"blowing from the ([a-z\-]+)$", expand=False)
    azimuth = name.map(WIND_AZIMUTH)

    radians = np.deg2rad(azimuth.astype(float))
    u = -speed * np.sin(radians)  # eastward component
    v = -speed * np.cos(radians)  # northward component

    calm = direction.str.startswith("Calm")
    u[calm] = 0.0
    v[calm] = 0.0
    return u, v


def load_data(path: Path) -> pd.DataFrame:
    """Load and clean the RP5 METAR export (semicolon-delimited with quotes).

    Returns a contiguous hourly grid. Short gaps are interpolated so that lags
    stay well-defined; the OBSERVED column records which hours are real
    measurements rather than interpolated filler.
    """
    df_raw = pd.read_csv(
        path,
        sep=";",
        engine="python",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )

    # Fixed-column selection by index: the header is unreliable (the whole record
    # is wrapped in quotes, and the wind field may contain "Calm, no wind").
    df = df_raw.iloc[:, [0, 1, 3, 4, 6, 12, 5]].copy()
    df.columns = ["timestamp", "temp", "pressure", "humidity", "wind_speed", "dew_point",
                  "wind_dir"]

    # Strip quotes/spaces then convert
    df = df.apply(lambda col: col.astype(str).str.replace('"', "", regex=False).str.strip())

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    for col in ["temp", "pressure", "humidity", "wind_speed", "dew_point"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive the wind vector before resampling, so that the hourly mean averages
    # vectors rather than compass labels.
    df["wind_u"], df["wind_v"] = parse_wind(df["wind_dir"], df["wind_speed"])
    df = df[METEO_COLS]

    df = df.dropna(subset=["temp"])

    # Resample to an hourly grid. This grid must stay contiguous: every shift()
    # in add_features() is positional, so a dropped row would silently change
    # what "24 hours" means.
    df = df.resample("1h").mean()

    # Record real measurements before any filling, then interpolate short gaps.
    # Linear-in-time beats ffill here: ffill would assert a flat temperature all
    # night and erase the nocturnal minimum.
    df[OBSERVED] = df["temp"].notna()
    df[METEO_COLS] = df[METEO_COLS].interpolate(method="time", limit=INTERP_LIMIT)

    return df


def load_era5(path: Path, grid_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load the ERA5 grid and re-stamp it from UTC into Kursk local time."""
    era = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()

    offsets = np.where(era.index < MSK_SWITCH_UTC, 4, 3)
    era.index = era.index + pd.to_timedelta(offsets, unit="h")
    era = era[~era.index.duplicated(keep="first")]

    return era.reindex(grid_index)


def add_era5_features(feat: pd.DataFrame) -> pd.DataFrame:
    """Advection features: how the upwind air differs from the air overhead now.

    The raw upwind temperature matters, but the *contrast* against Kursk is what
    signals a change: +8°C at 600km upwind is a warm front inbound, while the same
    temperature everywhere means nothing is coming.
    """
    if "era5_temp_center" not in feat.columns:
        return feat

    ring = [c for c in feat.columns if c.startswith("era5_temp_") and c != "era5_temp_center"]
    for col in ring:
        feat[col.replace("era5_temp_", "era5_tadv_")] = feat[col] - feat["era5_temp_center"]

    # ERA5's own 24h tendency at the centre -- the reanalysis sees the synoptic
    # pattern the single station cannot.
    feat["era5_temp_center_diff_24h"] = feat["era5_temp_center"].diff(24)
    feat["era5_pressure_center_diff_24h"] = feat["era5_pressure_center"].diff(24)
    return feat


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features, lags, deltas, and target."""
    feat = df.copy()

    # Cyclical encodings
    hours = feat.index.hour
    doy = feat.index.dayofyear
    feat["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Lags (positional == hourly, guaranteed by the contiguous grid from load_data)
    for lag in [1, 3, 6, 24]:
        feat[f"temp_lag_{lag}"] = feat["temp"].shift(lag)
        feat[f"pressure_lag_{lag}"] = feat["pressure"].shift(lag)

    # Tendencies. Pressure fall/rise is the classic signal for an approaching
    # front, and the advected air mass shows up as a temperature/wind trend.
    for window in [3, 6, 24]:
        feat[f"pressure_diff_{window}h"] = feat["pressure"].diff(window)
        feat[f"temp_diff_{window}h"] = feat["temp"].diff(window)

    for window in [6, 24]:
        feat[f"wind_u_mean_{window}h"] = feat["wind_u"].rolling(window, min_periods=1).mean()
        feat[f"wind_v_mean_{window}h"] = feat["wind_v"].rolling(window, min_periods=1).mean()

    feat["dew_spread"] = feat["temp"] - feat["dew_point"]

    feat = add_era5_features(feat)

    # Target: temperature HORIZON hours after this row's timestamp, plus whether
    # that temperature was actually measured.
    feat[TARGET] = feat["temp"].shift(-HORIZON)
    feat[TARGET_OBSERVED] = feat[OBSERVED].shift(-HORIZON, fill_value=False)

    return feat


def feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def split_labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for training/scoring: real measurement now AND at target time."""
    usable = df[OBSERVED] & df[TARGET_OBSERVED] & df[REQUIRED_COLS].notna().all(axis=1)
    return df.loc[usable]


def build_dataset(era5_lag_hours: int | None = 0) -> pd.DataFrame:
    """Station grid joined with ERA5. era5_lag_hours=None skips ERA5 entirely."""
    df = load_data(DATA_PATH)

    if era5_lag_hours is not None and ERA5_PATH.exists():
        era = load_era5(ERA5_PATH, df.index)
        if era5_lag_hours:
            era = era.shift(era5_lag_hours)
        df = df.join(era)

    return add_features(df)


def make_model() -> CatBoostRegressor:
    # Optimise MAE directly -- it is the reported metric, and it is far less
    # sensitive to the outlier days that dominate an RMSE objective.
    return CatBoostRegressor(
        iterations=5000,
        learning_rate=0.05,
        depth=8,
        loss_function="MAE",
        eval_metric="MAE",
        early_stopping_rounds=200,
        random_seed=42,
        verbose=500,
    )


def baseline_predictions(test_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Trivial forecasts any model must beat to be worth its complexity."""
    return {
        "Persistence (temp now)": test_df["temp"],
        "Temp 24h ago": test_df["temp_lag_24"],
        "Test mean": pd.Series(test_df[TARGET].mean(), index=test_df.index),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    ARTIFACT_DIR.mkdir(exist_ok=True)

    df = build_dataset()
    has_era5 = "era5_temp_center" in df.columns
    print(f"Hourly grid: {len(df):,} rows ({df.index.min()} -> {df.index.max()})")
    print(f"Genuinely measured hours: {df[OBSERVED].sum():,} ({df[OBSERVED].mean() * 100:.1f}%)")
    print(f"ERA5 grid: {'joined' if has_era5 else 'ABSENT (run fetch_era5.py)'}")

    labeled = split_labeled(df)
    train_df = labeled.loc[labeled.index < VAL_START]
    val_df = labeled.loc[(labeled.index >= VAL_START) & (labeled.index < TEST_START)]
    test_df = labeled.loc[labeled.index >= TEST_START]
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("A data split is empty. Check date coverage.")

    cols = feature_names(df)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,} | "
          f"Features: {len(cols)}")

    model = make_model()
    model.fit(train_df[cols], train_df[TARGET],
              eval_set=(val_df[cols], val_df[TARGET]), use_best_model=True)
    print(f"Best iteration: {model.get_best_iteration()} of {model.tree_count_} trees kept")

    preds = model.predict(test_df[cols])
    mae = mean_absolute_error(test_df[TARGET], preds)
    print(f"\nTest MAE: {mae:.3f}")

    # --- artifacts ------------------------------------------------------------
    # Indexed by valid_time (the hour being forecast), not feature time: the two
    # differ by HORIZON, and conflating them is exactly the bug that made an
    # earlier version's chart look good while comparing the wrong hours.
    predictions = pd.DataFrame(
        {
            "valid_time": test_df.index + pd.Timedelta(hours=HORIZON),
            "actual": test_df[TARGET].to_numpy(),
            "predicted": preds,
        }
    )
    for name, series in baseline_predictions(test_df).items():
        predictions[name] = series.to_numpy()
    predictions.to_csv(ARTIFACT_DIR / "predictions.csv", index=False)

    metrics = {
        "horizon_hours": HORIZON,
        "test_mae": round(float(mae), 4),
        "best_iteration": int(model.get_best_iteration()),
        "n_features": len(cols),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "test_start": str(test_df.index.min()),
        "test_end": str(test_df.index.max()),
        "has_era5": has_era5,
        "baselines": {},
    }
    print("\nMAE comparison (lower is better):")
    print(f"  {'Model (CatBoost)':<24} {mae:.3f}")
    for name, series in baseline_predictions(test_df).items():
        b_mae = float(mean_absolute_error(test_df[TARGET], series))
        metrics["baselines"][name] = round(b_mae, 4)
        print(f"  {name:<24} {b_mae:.3f}   (model {(b_mae - mae) / b_mae * 100:+.1f}%)")

    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    importance = pd.DataFrame(
        {"feature": cols, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)

    # --- forward forecast -----------------------------------------------------
    # The final HORIZON rows of the grid -- a NaN target alone does not identify
    # them, since interpolation gaps also leave the target unset.
    cutoff = df.index.max() - pd.Timedelta(hours=HORIZON - 1)
    future = df.loc[(df.index >= cutoff) & df[REQUIRED_COLS].notna().all(axis=1)]
    if not future.empty:
        forecast = pd.DataFrame({
            "valid_time": future.index + pd.Timedelta(hours=HORIZON),
            "forecast_temp": model.predict(future[cols]).round(2),
        })
        forecast.to_csv(ARTIFACT_DIR / "forecast.csv", index=False)
        print(f"\nForward forecast: {len(forecast)}h past last observation")

    print(f"Artifacts written to {ARTIFACT_DIR}/ -- run `python plots.py` to render charts.")


if __name__ == "__main__":
    main()
