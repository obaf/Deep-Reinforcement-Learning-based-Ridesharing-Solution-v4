"""Stratified sample of cleaned + centroid-attached HVFHV data.

Produces ``data/processed/hvfhv_2019_sample_200k.parquet`` with roughly
200,000 trips, balanced across (hour-of-day x weekday) strata. Pool-
flagged rows (``shared_request_flag == 'Y'``) are kept first; the
remaining slots are filled from non-pool rows.

If a robustness centroids file is present, a smaller balanced sample is
also produced (``data/processed/hvfhv_robustness_sample.parquet``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _add_strata_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pickup = pd.to_datetime(df["pickup_datetime"])
    df["hour_of_day"] = pickup.dt.hour
    df["weekday"] = pickup.dt.weekday
    return df


def stratified_sample(
    df: pd.DataFrame,
    target_total: int,
    seed: int = 0,
    weekday_only: bool = True,
) -> pd.DataFrame:
    df = _add_strata_cols(df)
    if weekday_only:
        df = df[df["weekday"].between(0, 4)].copy()  # Mon-Fri

    flag = df["shared_request_flag"].astype(str).str.strip().str.upper() == "Y"
    df["_is_pool"] = flag.values

    if weekday_only:
        n_strata = 24 * 5
    else:
        n_strata = 24 * 7
    target_per_stratum = max(1, target_total // n_strata)

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    pool_cap = max(1, target_per_stratum // 2)
    grouped = df.groupby(["hour_of_day", "weekday"], sort=True)
    for (hh, wd), g in grouped:
        pool_rows = g[g["_is_pool"]]
        nonpool_rows = g[~g["_is_pool"]]
        nP = min(pool_cap, len(pool_rows))
        nN = min(target_per_stratum - nP, len(nonpool_rows))
        if nP > 0:
            keep_pool = pool_rows.iloc[
                rng.choice(len(pool_rows), size=nP, replace=False)
            ]
        else:
            keep_pool = pool_rows.iloc[0:0]
        if nN > 0:
            keep_nonpool = nonpool_rows.iloc[
                rng.choice(len(nonpool_rows), size=nN, replace=False)
            ]
        else:
            keep_nonpool = nonpool_rows.iloc[0:0]
        parts.append(pd.concat([keep_pool, keep_nonpool], ignore_index=True))

    out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
    out = out.drop(columns=["_is_pool"])
    out = out.sort_values("pickup_datetime").reset_index(drop=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int, default=200_000,
                   help="approximate total rows in the main sample")
    p.add_argument("--robustness-target", type=int, default=20_000,
                   help="approximate total rows in the robustness sample")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    main_in = INTERIM_DIR / "hvfhv_2019_centroids.parquet"
    if not main_in.exists():
        raise FileNotFoundError(
            f"{main_in} not found. Run scripts/03_zone_to_centroid.py first."
        )
    print(f"Reading {main_in.name} ...", flush=True)
    df = pd.read_parquet(main_in)
    print(f"  {len(df):,} rows in cleaned + centroid file", flush=True)

    sample = stratified_sample(df, args.target, seed=args.seed,
                               weekday_only=True)
    out_main = PROCESSED_DIR / "hvfhv_2019_sample_200k.parquet"
    sample.to_parquet(out_main, index=False)
    print(f"Wrote {out_main} ({len(sample):,} rows)", flush=True)

    rob_files = list(INTERIM_DIR.glob("hvfhv_2019-*_centroids.parquet"))
    for rob_in in rob_files:
        if rob_in.name == "hvfhv_2019_centroids.parquet":
            continue
        print(f"\nReading robustness file {rob_in.name} ...", flush=True)
        df_rob = pd.read_parquet(rob_in)
        sample_rob = stratified_sample(
            df_rob, args.robustness_target, seed=args.seed, weekday_only=True
        )
        label = rob_in.name.replace("_centroids.parquet", "")
        out_rob = PROCESSED_DIR / f"{label}_sample.parquet"
        sample_rob.to_parquet(out_rob, index=False)
        print(f"Wrote {out_rob} ({len(sample_rob):,} rows)", flush=True)


if __name__ == "__main__":
    main()
