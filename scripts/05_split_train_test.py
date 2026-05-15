"""Split the balanced sample into env-compatible train/test Excel files.

Holds out the calendar week of 2019-04-08 .. 2019-04-14 as the test set;
all remaining rows are training. Column names are remapped to what
``RidePoolEnv`` expects:

    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
    tpep_pickup_datetime, tpep_dropoff_datetime,
    trip_distance, total_amount,
    shared_request_flag, shared_match_flag

If a robustness sample file exists, it is converted into a single
``data/processed/robustness.xlsx`` so it can be evaluated by
``agent.inference``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


COLUMN_MAP = {
    "pickup_datetime": "tpep_pickup_datetime",
    "dropoff_datetime": "tpep_dropoff_datetime",
    "trip_miles": "trip_distance",
    "base_passenger_fare": "total_amount",
}


def _to_env_format(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP).copy()
    keep = [
        "pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon",
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "trip_distance", "total_amount",
        "shared_request_flag", "shared_match_flag",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--test-week-start", default="2019-04-08",
        help="ISO date of Monday at the start of the held-out test week",
    )
    args = p.parse_args()

    main_in = PROCESSED_DIR / "hvfhv_2019_sample_200k.parquet"
    if not main_in.exists():
        raise FileNotFoundError(
            f"{main_in} not found. Run scripts/04_sample_balanced.py first."
        )
    df = pd.read_parquet(main_in)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    test_start = pd.Timestamp(args.test_week_start)
    test_end = test_start + pd.Timedelta(days=7)

    is_test = (df["pickup_datetime"] >= test_start) & (df["pickup_datetime"] < test_end)
    train_df = df[~is_test].copy()
    test_df = df[is_test].copy()
    print(f"Train rows: {len(train_df):,}", flush=True)
    print(f"Test rows  : {len(test_df):,} (window {test_start.date()} .. "
          f"{test_end.date()})", flush=True)

    train_out = PROCESSED_DIR / "train.xlsx"
    test_out = PROCESSED_DIR / "test.xlsx"
    _to_env_format(train_df).to_excel(train_out, index=False)
    _to_env_format(test_df).to_excel(test_out, index=False)
    print(f"Wrote {train_out}", flush=True)
    print(f"Wrote {test_out}", flush=True)

    for rob_in in PROCESSED_DIR.glob("hvfhv_2019-*_sample.parquet"):
        if rob_in.name == "hvfhv_2019_sample_200k.parquet":
            continue
        rob_df = pd.read_parquet(rob_in)
        rob_out = PROCESSED_DIR / "robustness.xlsx"
        _to_env_format(rob_df).to_excel(rob_out, index=False)
        print(f"Wrote {rob_out} ({len(rob_df):,} rows)", flush=True)


if __name__ == "__main__":
    main()
