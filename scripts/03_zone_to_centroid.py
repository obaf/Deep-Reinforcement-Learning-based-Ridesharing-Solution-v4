"""Replace TLC zone IDs with zone-centroid lat/lon.

Reads ``data/interim/hvfhv_2019_clean.parquet`` and the zone shapefile,
then writes ``data/interim/hvfhv_2019_centroids.parquet`` with
``pickup_lat / pickup_lon / dropoff_lat / dropoff_lon`` and the original
``PULocationID`` / ``DOLocationID`` columns dropped.

A NYC bounding-box filter is applied:
    lat in [40.4, 41.0], lon in [-74.3, -73.6]

Both training and robustness cleaned files are processed if they exist.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
ZONES_DIR = PROJECT_ROOT / "data" / "raw" / "taxi_zones"


def _resolve_shapefile() -> Path:
    """Locate ``taxi_zones.shp`` under ``data/raw/taxi_zones/``."""
    direct = ZONES_DIR / "taxi_zones.shp"
    if direct.exists():
        return direct
    matches = list(ZONES_DIR.rglob("taxi_zones.shp"))
    if matches:
        return matches[0]
    return direct

NYC_BBOX = {
    "lat_min": 40.4,
    "lat_max": 41.0,
    "lon_min": -74.3,
    "lon_max": -73.6,
}


def _zone_centroids() -> pd.DataFrame:
    """Return a DataFrame of LocationID -> (centroid_lat, centroid_lon)."""
    shp = _resolve_shapefile()
    if not shp.exists():
        raise FileNotFoundError(
            f"Zone shapefile not found at {shp}. "
            "Run scripts/01_download_tlc.py first."
        )
    zones = gpd.read_file(str(shp))
    zones_proj = zones.to_crs(2263)
    cent = zones_proj.geometry.centroid.to_crs(4326)
    out = pd.DataFrame({
        "LocationID": zones["LocationID"].values,
        "centroid_lat": cent.y.values,
        "centroid_lon": cent.x.values,
    })
    return out


def attach_centroids(df: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    pu = centroids.rename(
        columns={
            "LocationID": "PULocationID",
            "centroid_lat": "pickup_lat",
            "centroid_lon": "pickup_lon",
        }
    )
    do = centroids.rename(
        columns={
            "LocationID": "DOLocationID",
            "centroid_lat": "dropoff_lat",
            "centroid_lon": "dropoff_lon",
        }
    )
    df = df.merge(pu, on="PULocationID", how="inner")
    df = df.merge(do, on="DOLocationID", how="inner")

    n0 = len(df)
    df = df[
        df["pickup_lat"].between(NYC_BBOX["lat_min"], NYC_BBOX["lat_max"])
        & df["pickup_lon"].between(NYC_BBOX["lon_min"], NYC_BBOX["lon_max"])
        & df["dropoff_lat"].between(NYC_BBOX["lat_min"], NYC_BBOX["lat_max"])
        & df["dropoff_lon"].between(NYC_BBOX["lon_min"], NYC_BBOX["lon_max"])
    ]
    print(
        f"    bbox filter kept {len(df):,} of {n0:,} rows "
        f"({100.0 * len(df) / max(1, n0):.2f}%)",
        flush=True,
    )
    return df.drop(columns=["PULocationID", "DOLocationID"]).reset_index(drop=True)


def process(in_path: Path, out_path: Path, centroids: pd.DataFrame) -> None:
    if not in_path.exists():
        print(f"  skip: {in_path} not found", flush=True)
        return
    print(f"  reading {in_path.name} ...", flush=True)
    df = pd.read_parquet(in_path)
    df = attach_centroids(df, centroids)
    df.to_parquet(out_path, index=False)
    print(f"  wrote {out_path} ({len(df):,} rows)", flush=True)


def main() -> None:
    print("Computing zone centroids ...", flush=True)
    centroids = _zone_centroids()
    print(f"  {len(centroids)} zones", flush=True)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    process(
        INTERIM_DIR / "hvfhv_2019_clean.parquet",
        INTERIM_DIR / "hvfhv_2019_centroids.parquet",
        centroids,
    )
    for p in INTERIM_DIR.glob("hvfhv_2019-*_clean.parquet"):
        out = INTERIM_DIR / p.name.replace("_clean.parquet", "_centroids.parquet")
        process(p, out, centroids)


if __name__ == "__main__":
    main()
