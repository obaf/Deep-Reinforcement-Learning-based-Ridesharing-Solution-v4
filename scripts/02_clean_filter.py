"""Clean and filter the downloaded HVFHV monthly parquet files.

Filters applied in order:
    - drop NaN in {request_datetime, pickup_datetime, dropoff_datetime,
      PULocationID, DOLocationID, trip_miles, base_passenger_fare}
    - trip_miles in [0.1, 100]
    - trip_time = (dropoff - pickup) seconds in [30, 18000]
    - base_passenger_fare in [2.50, 300.00]
    - dropoff_datetime > pickup_datetime
    - keep only Uber (HV0003) and Lyft (HV0005)

August 2019 is filtered separately into a sibling parquet so it can be
used as a robustness slice without contaminating the training set.

Output: data/interim/hvfhv_2019_clean.parquet
        data/interim/hvfhv_2019-08_clean.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

REQUIRED_COLS = [
    "hvfhs_license_num",
    "request_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "base_passenger_fare",
    "shared_request_flag",
    "shared_match_flag",
]


def load_and_clean(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        print(f"  missing: {parquet_path}, skipping", flush=True)
        return pd.DataFrame(columns=REQUIRED_COLS)

    print(f"  reading {parquet_path.name} ...", flush=True)
    keep_cols = [c for c in REQUIRED_COLS if c is not None]
    try:
        df = pd.read_parquet(parquet_path, columns=keep_cols)
    except Exception:
        df = pd.read_parquet(parquet_path)
        df = df[[c for c in keep_cols if c in df.columns]]

    n0 = len(df)
    df = df.dropna(subset=[
        "request_datetime", "pickup_datetime", "dropoff_datetime",
        "PULocationID", "DOLocationID", "trip_miles", "base_passenger_fare",
    ])
    df = df[(df["trip_miles"] >= 0.1) & (df["trip_miles"] <= 100.0)]

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
    df["request_datetime"] = pd.to_datetime(df["request_datetime"])
    trip_time = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()
    df = df[(trip_time >= 30) & (trip_time <= 18_000)]

    df = df[
        (df["base_passenger_fare"] >= 2.50)
        & (df["base_passenger_fare"] <= 300.00)
    ]
    df = df[df["dropoff_datetime"] > df["pickup_datetime"]]
    df = df[df["hvfhs_license_num"].isin(["HV0003", "HV0005"])]

    print(
        f"    kept {len(df):,} of {n0:,} rows "
        f"({100.0 * len(df) / max(1, n0):.2f}%)",
        flush=True,
    )
    return df.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-months", nargs="+",
        default=["2019-03", "2019-04", "2019-05"],
        help="months to concatenate into the main training cleaned file",
    )
    p.add_argument(
        "--robustness-months", nargs="+",
        default=["2019-08"],
        help="months to concatenate into the robustness cleaned file",
    )
    args = p.parse_args()

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    train_dfs = []
    for ym in args.train_months:
        train_dfs.append(load_and_clean(RAW_DIR / f"fhvhv_tripdata_{ym}.parquet"))
    if any(len(d) > 0 for d in train_dfs):
        train_df = pd.concat([d for d in train_dfs if len(d) > 0],
                             ignore_index=True)
        out_train = INTERIM_DIR / "hvfhv_2019_clean.parquet"
        train_df.to_parquet(out_train, index=False)
        print(f"Wrote {out_train} ({len(train_df):,} rows)", flush=True)
    else:
        print("No training months produced data; skipping write.", flush=True)

    rob_dfs = []
    for ym in args.robustness_months:
        rob_dfs.append(load_and_clean(RAW_DIR / f"fhvhv_tripdata_{ym}.parquet"))
    if any(len(d) > 0 for d in rob_dfs):
        rob_df = pd.concat([d for d in rob_dfs if len(d) > 0],
                           ignore_index=True)
        rob_label = "_".join(args.robustness_months)
        out_rob = INTERIM_DIR / f"hvfhv_{rob_label}_clean.parquet"
        rob_df.to_parquet(out_rob, index=False)
        print(f"Wrote {out_rob} ({len(rob_df):,} rows)", flush=True)


if __name__ == "__main__":
    main()
