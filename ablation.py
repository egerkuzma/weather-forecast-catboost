"""
Is the ERA5 gain real, or is it borrowed from the future?

ERA5 is a reanalysis assimilated in 12h 4D-Var windows, so the field at time t
carries observations from up to ~9h after t. Lagging ERA5 removes that peek: if
the advantage survives, it is genuine upwind advection signal; if it collapses to
the station-only level, the headline number cannot be earned in real time.

Every configuration is run over several seeds, because CatBoost's seed noise
(sd ~0.008 MAE) is larger than several of the effects people are tempted to read
into a single run.

Writes artifacts/ablation.json for plots.py.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import train_weather as tw

SEEDS = (1, 2, 3)


def evaluate(dataset: pd.DataFrame, label: str) -> dict[str, float]:
    cols = tw.feature_names(dataset)
    labeled = tw.split_labeled(dataset)
    train_df = labeled.loc[labeled.index < tw.VAL_START]
    val_df = labeled.loc[(labeled.index >= tw.VAL_START) & (labeled.index < tw.TEST_START)]
    test_df = labeled.loc[labeled.index >= tw.TEST_START]

    scores = []
    for seed in SEEDS:
        model = tw.make_model()
        model.set_params(verbose=0, random_seed=seed)
        model.fit(train_df[cols], train_df[tw.TARGET],
                  eval_set=(val_df[cols], val_df[tw.TARGET]), use_best_model=True)
        scores.append(float(mean_absolute_error(test_df[tw.TARGET], model.predict(test_df[cols]))))

    result = {"mean": float(np.mean(scores)), "sd": float(np.std(scores)), "n_features": len(cols)}
    print(f"  {label:<26} MAE {result['mean']:.3f} +/- {result['sd']:.3f}  ({len(cols)} features)")
    return result


def main() -> None:
    tw.ARTIFACT_DIR.mkdir(exist_ok=True)
    print(f"ERA5 lag sweep over {len(SEEDS)} seeds -- does the gain survive the lag?")

    results = {"Station only": evaluate(tw.build_dataset(era5_lag_hours=None), "station only")}
    for lag in (0, 12, 24):
        label = f"+ERA5 at t-{lag}h" + ("  <- headline" if lag == 0 else "")
        results[f"+ERA5 t-{lag}h"] = evaluate(tw.build_dataset(era5_lag_hours=lag), label)

    baseline = results["Station only"]["mean"]
    headline = results["+ERA5 t-0h"]["mean"]
    lagged = results["+ERA5 t-12h"]["mean"]
    noise = max(results["Station only"]["sd"], 1e-6)
    print(f"\nERA5 at t   gains {baseline - headline:+.3f} MAE ({(baseline - headline) / noise:.0f} sd)")
    print(f"ERA5 at t-12h gains {baseline - lagged:+.3f} MAE ({(baseline - lagged) / noise:.0f} sd)")

    out = tw.ARTIFACT_DIR / "ablation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
