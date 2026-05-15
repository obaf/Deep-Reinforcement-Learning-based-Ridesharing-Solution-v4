"""Greedy inference (epsilon = 0) on the test set.

Writes an execution plan with one row per assignment decision. Outcome
categories: ``solo``, ``shared``, ``vmt_reject_solo``, ``fallback_solo``.
Per-rider delay is recorded for descriptive reporting only.

Run as ``python -m agent.inference``.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agent.dqn import DQN
from env.ride_pool_env import (
    RidePoolEnv,
    haversine,
    compute_pair_geometry,
)
from env.reward import (
    FARE_PER_MILE,
    SHARED_DISCOUNT,
    TRAVEL_SPEED_MPH,
)


COST_PER_MILE = FARE_PER_MILE
SPEED_MPH = TRAVEL_SPEED_MPH


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_test_xlsx() -> Path:
    return _project_root() / "data" / "processed" / "test.xlsx"


def _default_checkpoint() -> Path:
    return _project_root() / "models" / "model.pth"


def _default_plan_xlsx() -> Path:
    return _project_root() / "analysis" / "trip_execution_plan_v4.xlsx"


def _pair_geometry_with_delays(row_i: pd.Series, row_j: pd.Series) -> dict:
    """Return route geometry and per-rider delays (in minutes) for a pair."""
    dist_i = haversine(
        row_i["pickup_lat"], row_i["pickup_lon"],
        row_i["dropoff_lat"], row_i["dropoff_lon"],
    )
    dist_j = haversine(
        row_j["pickup_lat"], row_j["pickup_lon"],
        row_j["dropoff_lat"], row_j["dropoff_lon"],
    )

    p1p2 = haversine(
        row_i["pickup_lat"], row_i["pickup_lon"],
        row_j["pickup_lat"], row_j["pickup_lon"],
    )
    p2d1 = haversine(
        row_j["pickup_lat"], row_j["pickup_lon"],
        row_i["dropoff_lat"], row_i["dropoff_lon"],
    )
    p2d2 = haversine(
        row_j["pickup_lat"], row_j["pickup_lon"],
        row_j["dropoff_lat"], row_j["dropoff_lon"],
    )

    if p2d1 < p2d2:
        d1d2 = haversine(
            row_i["dropoff_lat"], row_i["dropoff_lon"],
            row_j["dropoff_lat"], row_j["dropoff_lon"],
        )
        shared_vmt = p1p2 + p2d1 + d1d2
        delay_i = ((p1p2 + p2d1) - dist_i) / SPEED_MPH * 60.0
        delay_j = ((p2d1 + d1d2) - dist_j) / SPEED_MPH * 60.0
    else:
        d2d1 = haversine(
            row_j["dropoff_lat"], row_j["dropoff_lon"],
            row_i["dropoff_lat"], row_i["dropoff_lon"],
        )
        shared_vmt = p1p2 + p2d2 + d2d1
        delay_i = ((p1p2 + p2d2 + d2d1) - dist_i) / SPEED_MPH * 60.0
        delay_j = (p2d2 - dist_j) / SPEED_MPH * 60.0

    return {
        "dist_i": dist_i,
        "dist_j": dist_j,
        "shared_vmt": shared_vmt,
        "delay_i": delay_i,
        "delay_j": delay_j,
    }


def run_inference(env: RidePoolEnv, model: torch.nn.Module) -> pd.DataFrame:
    """Run one greedy episode and record per-decision details."""
    model.eval()
    records: list[dict] = []
    obs, _ = env.reset()
    done = False

    while not done:
        current_idx = env.current_idx
        row_i = env.df.iloc[current_idx]
        dist_i = haversine(
            row_i["pickup_lat"], row_i["pickup_lon"],
            row_i["dropoff_lat"], row_i["dropoff_lon"],
        )

        mask = env.action_mask(obs)
        with torch.no_grad():
            q = (
                model(torch.from_numpy(obs).float().unsqueeze(0))
                .squeeze(0)
                .numpy()
            )
        q_masked = np.where(mask, q, -np.inf)
        action = int(np.argmax(q_masked))

        partner_idx: int | None = None
        dist_j = 0.0
        delay_i = delay_j = 0.0
        geom_pre: dict | None = None

        j = current_idx + action if action != 0 else None
        if action != 0 and j is not None and j < env.n_trips and j not in env.assigned:
            geom_pre = _pair_geometry_with_delays(row_i, env.df.iloc[j])

        obs, _reward, done, _, info = env.step(action)

        if action == 0:
            outcome = "solo"
            actual_vmt = dist_i
        elif j is None or j >= env.n_trips or geom_pre is None:
            outcome = "fallback_solo"
            actual_vmt = dist_i
        elif info.get("vmt_inc_reject"):
            outcome = "vmt_reject_solo"
            actual_vmt = dist_i
            dist_j = geom_pre["dist_j"]
            delay_i = geom_pre["delay_i"]
            delay_j = geom_pre["delay_j"]
        elif info.get("shared", 0) == 2:
            outcome = "shared"
            partner_idx = int(j)
            dist_j = geom_pre["dist_j"]
            actual_vmt = geom_pre["shared_vmt"]
            delay_i = geom_pre["delay_i"]
            delay_j = geom_pre["delay_j"]
        else:
            outcome = "fallback_solo"
            actual_vmt = dist_i

        if outcome == "shared":
            cost_solo_i = dist_i * COST_PER_MILE
            cost_solo_j = dist_j * COST_PER_MILE
            cost_actual_i = cost_solo_i * SHARED_DISCOUNT
            cost_actual_j = cost_solo_j * SHARED_DISCOUNT
        else:
            cost_solo_i = dist_i * COST_PER_MILE
            cost_solo_j = dist_j * COST_PER_MILE if dist_j > 0 else 0.0
            cost_actual_i = cost_solo_i
            cost_actual_j = cost_solo_j if dist_j > 0 else 0.0

        solo_vmt_total = dist_i + dist_j if outcome == "shared" else dist_i
        cost_solo_total = cost_solo_i + cost_solo_j
        cost_actual_total = cost_actual_i + cost_actual_j
        max_delay = max(delay_i, delay_j) if outcome == "shared" else 0.0

        row_j = (
            env.df.iloc[partner_idx]
            if partner_idx is not None
            else None
        )
        records.append({
            "decision_id": len(records),
            "trip_i_idx": int(current_idx),
            "action": action,
            "partner_idx": "" if partner_idx is None else partner_idx,
            "outcome": outcome,
            "dist_i_mi": dist_i,
            "dist_j_mi": dist_j,
            "solo_vmt_total_mi": solo_vmt_total,
            "actual_vmt_mi": actual_vmt,
            "vmt_saved_mi": solo_vmt_total - actual_vmt,
            "delay_i_min": delay_i,
            "delay_j_min": delay_j,
            "max_delay_min": max_delay,
            "cost_solo_i_usd": cost_solo_i,
            "cost_solo_j_usd": cost_solo_j,
            "cost_actual_i_usd": cost_actual_i,
            "cost_actual_j_usd": cost_actual_j,
            "cost_solo_total_usd": cost_solo_total,
            "cost_actual_total_usd": cost_actual_total,
            "cost_saved_usd": cost_solo_total - cost_actual_total,
            "pickup_lat_i":  row_i["pickup_lat"],
            "pickup_lon_i":  row_i["pickup_lon"],
            "dropoff_lat_i": row_i["dropoff_lat"],
            "dropoff_lon_i": row_i["dropoff_lon"],
            "pickup_lat_j":  row_j["pickup_lat"]  if row_j is not None else "",
            "pickup_lon_j":  row_j["pickup_lon"]  if row_j is not None else "",
            "dropoff_lat_j": row_j["dropoff_lat"] if row_j is not None else "",
            "dropoff_lon_j": row_j["dropoff_lon"] if row_j is not None else "",
            "pickup_datetime_i": row_i["tpep_pickup_datetime"],
        })

    return pd.DataFrame(records)


def summarise(plan: pd.DataFrame, n_trips_total: int) -> dict:
    """Aggregate the execution plan into headline metrics."""
    counts = Counter(plan["outcome"].tolist())
    n_decisions = len(plan)
    n_shared = counts.get("shared", 0)
    n_solo = counts.get("solo", 0)
    n_fallback = counts.get("fallback_solo", 0)
    n_vmt_reject = counts.get("vmt_reject_solo", 0)

    riders_via_share = 2 * n_shared
    riders_solo = n_solo + n_fallback + n_vmt_reject
    riders_total = riders_via_share + riders_solo

    total_solo_vmt = float(plan["solo_vmt_total_mi"].sum())
    total_actual_vmt = float(plan["actual_vmt_mi"].sum())
    total_vmt_saved = total_solo_vmt - total_actual_vmt
    vmt_reduction_pct = (
        100.0 * total_vmt_saved / total_solo_vmt if total_solo_vmt > 0 else 0.0
    )

    valid_share = plan[plan["outcome"] == "shared"]
    if len(valid_share) > 0:
        delays = pd.concat(
            [valid_share["delay_i_min"], valid_share["delay_j_min"]],
            ignore_index=True,
        )
        avg_delay = float(delays.mean())
        median_delay = float(delays.median())
        max_delay = float(delays.max())
        p95_delay = float(delays.quantile(0.95))
        share_with_delay = 100.0 * float((delays > 0.5).mean())
    else:
        avg_delay = median_delay = max_delay = p95_delay = 0.0
        share_with_delay = 0.0

    total_solo_cost = float(plan["cost_solo_total_usd"].sum())
    total_actual_cost = float(plan["cost_actual_total_usd"].sum())
    total_cost_saved = total_solo_cost - total_actual_cost
    cost_reduction_pct = (
        100.0 * total_cost_saved / total_solo_cost if total_solo_cost > 0 else 0.0
    )

    if riders_via_share > 0:
        savings_total = (
            valid_share[["cost_solo_i_usd", "cost_solo_j_usd"]].values.sum()
            - valid_share[["cost_actual_i_usd", "cost_actual_j_usd"]].values.sum()
        )
        avg_saving_per_shared_rider = savings_total / riders_via_share
    else:
        avg_saving_per_shared_rider = 0.0

    return {
        "n_trips_total": n_trips_total,
        "n_decisions": n_decisions,
        "n_solo": n_solo,
        "n_fallback_solo": n_fallback,
        "n_vmt_reject_solo": n_vmt_reject,
        "n_shared": n_shared,
        "riders_total": riders_total,
        "riders_via_share": riders_via_share,
        "riders_solo": riders_solo,
        "share_rate_riders_pct": (
            100.0 * riders_via_share / riders_total if riders_total > 0 else 0.0
        ),
        "total_solo_vmt_mi": total_solo_vmt,
        "total_actual_vmt_mi": total_actual_vmt,
        "total_vmt_saved_mi": total_vmt_saved,
        "vmt_reduction_pct": vmt_reduction_pct,
        "avg_delay_min": avg_delay,
        "median_delay_min": median_delay,
        "p95_delay_min": p95_delay,
        "max_delay_min": max_delay,
        "pct_shared_riders_with_any_delay": share_with_delay,
        "total_solo_cost_usd": total_solo_cost,
        "total_actual_cost_usd": total_actual_cost,
        "total_cost_saved_usd": total_cost_saved,
        "cost_reduction_pct": cost_reduction_pct,
        "avg_saving_per_shared_rider_usd": float(avg_saving_per_shared_rider),
    }


def load_model(checkpoint_path: Path, obs_dim: int, n_actions: int) -> DQN:
    """Restore a DQN from a checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train first via "
            f"`python -m agent.train`."
        )
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu",
                          weights_only=False)
    except TypeError:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")

    model = DQN(obs_dim, n_actions)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        ep = ckpt.get("episode", "unknown")
        eps = ckpt.get("epsilon", float("nan"))
        print(
            f"Loaded checkpoint at episode {ep} (epsilon at save = {eps})."
        )
    else:
        model.load_state_dict(ckpt)
        print("Loaded legacy state-dict checkpoint.")
    model.eval()
    return model


def main(
    data_file: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    plan_path: str | Path | None = None,
) -> None:
    data_file = Path(data_file) if data_file is not None else _default_test_xlsx()
    checkpoint_path = (
        Path(checkpoint_path) if checkpoint_path is not None
        else _default_checkpoint()
    )
    plan_path = Path(plan_path) if plan_path is not None else _default_plan_xlsx()
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    env = RidePoolEnv(str(data_file))
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = load_model(checkpoint_path, obs_dim, n_actions)

    plan = run_inference(env, model)
    try:
        plan.to_excel(plan_path, index=False)
        print(f"Wrote {plan_path}  ({len(plan):,} rows).")
    except PermissionError:
        fallback = plan_path.with_name(plan_path.stem + ".new" + plan_path.suffix)
        plan.to_excel(fallback, index=False)
        print(f"  [locked] {plan_path.name} is open elsewhere -- wrote "
              f"{fallback.name} ({len(plan):,} rows) instead.")

    stats = summarise(plan, n_trips_total=env.n_trips)
    print("\n--- Summary ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:42s} {v:,.4f}")
        else:
            print(f"  {k:42s} {v:,}")


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Run greedy inference (epsilon = 0) with the trained "
            "ride-pooling DQN. Produces a per-decision execution plan "
            "(.xlsx) and prints headline stats to stdout."
        ),
    )
    parser.add_argument(
        "--trip-file", "--data-file",
        dest="trip_file",
        default=None,
        help=(
            "Path to an .xlsx trip file. Required columns: pickup_lat, "
            "pickup_lon, dropoff_lat, dropoff_lon. Optional: "
            "shared_request_flag. Defaults to data/processed/test.xlsx."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=("Path to a model.pth checkpoint. Defaults to "
              "models/model.pth."),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=("Path to write the execution-plan .xlsx. Defaults to "
              "analysis/trip_execution_plan_v4.xlsx."),
    )
    args = parser.parse_args()
    main(
        data_file=args.trip_file,
        checkpoint_path=args.checkpoint,
        plan_path=args.out,
    )


if __name__ == "__main__":
    _cli()
