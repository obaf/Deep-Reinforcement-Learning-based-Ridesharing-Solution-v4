"""Download NYC TLC HVFHV monthly parquet files plus the taxi-zone
lookup CSV and shapefile.

Source: NYC TLC, https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Defaults to March/April/May 2019 plus August 2019. Already-downloaded
files are skipped unless ``--force`` is passed.

Usage:
    python scripts/01_download_tlc.py
    python scripts/01_download_tlc.py --months 2019-03 2019-04 2019-05 2019-08
    python scripts/01_download_tlc.py --force
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZONES_DIR = RAW_DIR / "taxi_zones"

PARQUET_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "fhvhv_tripdata_{ym}.parquet"
)
ZONE_LOOKUP_URL = (
    "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
)
ZONE_SHP_ZIP_URL = (
    "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
)

DEFAULT_MONTHS = ["2019-03", "2019-04", "2019-05", "2019-08"]


def _download(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  exists ({size_mb:.1f} MB), skipping: {dest.name}", flush=True)
        return dest

    print(f"  GET {url}", flush=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            disable=total == 0,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dest)
    return dest


def _unzip(zip_path: Path, target_dir: Path, force: bool = False) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if (target_dir / "taxi_zones.shp").exists() and not force:
        print(f"  shapefile already extracted in {target_dir.name}, skipping",
              flush=True)
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    print(f"  extracted shapefile into {target_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--months",
        nargs="+",
        default=DEFAULT_MONTHS,
        help="YYYY-MM tags to fetch (default: 2019-03 04 05 08)",
    )
    p.add_argument("--force", action="store_true",
                   help="re-download even if file exists")
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target dir: {RAW_DIR}", flush=True)

    failures: list[str] = []
    for ym in args.months:
        url = PARQUET_URL.format(ym=ym)
        dest = RAW_DIR / f"fhvhv_tripdata_{ym}.parquet"
        try:
            _download(url, dest, force=args.force)
        except Exception as e:
            failures.append(f"{ym}: {e}")
            print(f"  FAILED {ym}: {e}", flush=True)

    try:
        _download(ZONE_LOOKUP_URL, RAW_DIR / "taxi_zone_lookup.csv",
                  force=args.force)
    except Exception as e:
        failures.append(f"zone lookup: {e}")
        print(f"  FAILED zone lookup: {e}", flush=True)

    zip_path = RAW_DIR / "taxi_zones.zip"
    try:
        _download(ZONE_SHP_ZIP_URL, zip_path, force=args.force)
        _unzip(zip_path, ZONES_DIR, force=args.force)
    except Exception as e:
        failures.append(f"zones zip: {e}")
        print(f"  FAILED zones zip: {e}", flush=True)

    if failures:
        print("\nThe following items failed:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        sys.exit(1)
    print("\nAll downloads complete.", flush=True)


if __name__ == "__main__":
    main()
