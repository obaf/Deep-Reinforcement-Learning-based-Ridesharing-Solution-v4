"""Ride-pooling Gymnasium environment."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym

from env.reward import (
    SOLO_REWARD,
    W_VMT_SAVED,
    W_COST_SAVED,
    SHARED_DISCOUNT,
    VMT_REJECT_PEN,
    INVALID_ACT_PEN,
    FARE_PER_MILE,
    TRAVEL_SPEED_MPH,
)


WINDOW_SIZE = 10
N_FEATURES = 5


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two lat/lon points."""
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_pair_geometry(row_i, row_j):
    """Return (dist_i, dist_j, shared_vmt, p1p2, route_choice).

    ``route_choice`` is 1 if rider i is dropped first (p1 -> p2 -> d1 -> d2),
    and 2 if rider j is dropped first (p1 -> p2 -> d2 -> d1). All distances
    are in haversine miles.
    """
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
        return dist_i, dist_j, shared_vmt, p1p2, 1
    d2d1 = haversine(
        row_j["dropoff_lat"], row_j["dropoff_lon"],
        row_i["dropoff_lat"], row_i["dropoff_lon"],
    )
    shared_vmt = p1p2 + p2d2 + d2d1
    return dist_i, dist_j, shared_vmt, p1p2, 2


def _shared_flag(row) -> float:
    """Convert the HVFHV shared_request_flag value into a float (1.0 or 0.0)."""
    val = row.get("shared_request_flag", None) if hasattr(row, "get") else None
    if val is None:
        try:
            val = row["shared_request_flag"]
        except (KeyError, IndexError):
            return 0.0
    if isinstance(val, str):
        return 1.0 if val.strip().upper() == "Y" else 0.0
    if pd.isna(val):
        return 0.0
    try:
        return 1.0 if float(val) > 0.5 else 0.0
    except (TypeError, ValueError):
        return 0.0


class RidePoolEnv(gym.Env):
    """Sequential single-vehicle ride-pooling assignment environment.

    Observation
    -----------
    Flat float32 vector of length ``WINDOW_SIZE * N_FEATURES`` (= 50).
    Per candidate ``j`` in the next-10 lookahead window:

        f0  haversine pickup-pickup distance, miles
        f1  haversine dropoff-dropoff distance, miles
        f2  availability flag (0 if j already assigned, else 1)
        f3  VMT saved by sharing with j
        f4  shared_request_flag from the HVFHV record (1 if 'Y' else 0)

    Action set
    ----------
    Discrete(11): action 0 = solo; actions 1..10 = pair with the candidate
    at index ``current_idx + action``.

    Reward
    ------
    * solo:                              reward = SOLO_REWARD = 0
    * invalid action:                    reward = -INVALID_ACT_PEN (fall back to solo)
    * pair, VMT-increasing:              reward = -VMT_REJECT_PEN (fall back to solo)
    * pair, VMT-saving:
        delta_VMT  = (solo_i + solo_j) - shared_vmt
        delta_Cost = (solo_i + solo_j) * FARE_PER_MILE * (1 - SHARED_DISCOUNT)
        reward = W_VMT_SAVED * delta_VMT + W_COST_SAVED * delta_Cost
    """

    metadata = {"render_modes": []}

    def __init__(self, data_file: str | Path):
        super().__init__()
        path = Path(data_file)
        df = pd.read_excel(path)
        df = df.dropna(
            subset=["pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"]
        ).reset_index(drop=True)

        self.df = df
        self.n_trips = len(df)
        self.window_size = WINDOW_SIZE
        self.n_features = N_FEATURES

        self.action_space = gym.spaces.Discrete(self.window_size + 1)
        self.observation_space = gym.spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(self.window_size * self.n_features,),
            dtype=np.float32,
        )
        self.current_idx = 0
        self.assigned: set[int] = set()
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_idx = 0
        self.assigned = set()
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.window_size, self.n_features), dtype=np.float32)
        if self.current_idx >= self.n_trips:
            return obs.flatten()

        row_i = self.df.iloc[self.current_idx]
        c = 0
        j = self.current_idx + 1
        while c < self.window_size and j < self.n_trips:
            row_j = self.df.iloc[j]
            p_dist = haversine(
                row_i["pickup_lat"], row_i["pickup_lon"],
                row_j["pickup_lat"], row_j["pickup_lon"],
            )
            d_dist = haversine(
                row_i["dropoff_lat"], row_i["dropoff_lon"],
                row_j["dropoff_lat"], row_j["dropoff_lon"],
            )
            is_avail = 0.0 if j in self.assigned else 1.0
            d_i, d_j, shared_vmt, _p1p2, _route = compute_pair_geometry(row_i, row_j)
            vmt_saved_est = (d_i + d_j) - shared_vmt
            shared_hint = _shared_flag(row_j)

            obs[c] = [p_dist, d_dist, is_avail, vmt_saved_est, shared_hint]
            c += 1
            j += 1
        return obs.flatten()

    def action_mask(self, obs: np.ndarray | None = None) -> np.ndarray:
        """Boolean mask over the discrete action space.

        ``mask[0]`` is always True (solo is always feasible). For each
        candidate ``a`` in 1..window_size, the mask is True only if the
        candidate slot is in-window, the candidate has not been assigned
        yet, and pairing with this candidate would not increase total VMT.
        """
        if obs is None:
            obs = self._get_obs()
        feats = np.asarray(obs, dtype=np.float32).reshape(
            self.window_size, self.n_features,
        )
        avail = feats[:, 2] > 0.5
        vmt_saved = feats[:, 3] > 0.0
        feasible_share = avail & vmt_saved
        mask = np.ones(self.window_size + 1, dtype=bool)
        mask[1:] = feasible_share
        return mask

    def step(self, action: int):
        if self.current_idx >= self.n_trips:
            return self._get_obs(), 0.0, True, False, {
                "vmt": 0.0,
                "shared": 0,
                "vmt_inc_reject": 0,
                "vmt_saved": 0.0,
                "max_delay_min": 0.0,
            }

        row_i = self.df.iloc[self.current_idx]
        solo_vmt_i = haversine(
            row_i["pickup_lat"], row_i["pickup_lon"],
            row_i["dropoff_lat"], row_i["dropoff_lon"],
        )

        reward = 0.0
        vmt = 0.0
        shared = 0
        vmt_increase_flag = 0
        vmt_saved_step = 0.0
        max_delay_min = 0.0

        if action == 0:
            vmt = solo_vmt_i
            reward = SOLO_REWARD
            self.assigned.add(self.current_idx)
        else:
            j = self.current_idx + action
            if j >= self.n_trips or j in self.assigned:
                vmt = solo_vmt_i
                reward = SOLO_REWARD - INVALID_ACT_PEN
                self.assigned.add(self.current_idx)
            else:
                row_j = self.df.iloc[j]
                d_i, d_j, shared_vmt, p1p2, route = compute_pair_geometry(row_i, row_j)
                solo_vmt_j = d_j
                solo_total = solo_vmt_i + solo_vmt_j

                if route == 1:
                    p2d1 = haversine(
                        row_j["pickup_lat"], row_j["pickup_lon"],
                        row_i["dropoff_lat"], row_i["dropoff_lon"],
                    )
                    d1d2 = haversine(
                        row_i["dropoff_lat"], row_i["dropoff_lon"],
                        row_j["dropoff_lat"], row_j["dropoff_lon"],
                    )
                    delay_i = ((p1p2 + p2d1) - solo_vmt_i) / TRAVEL_SPEED_MPH * 60.0
                    delay_j = ((p2d1 + d1d2) - solo_vmt_j) / TRAVEL_SPEED_MPH * 60.0
                else:
                    p2d2 = haversine(
                        row_j["pickup_lat"], row_j["pickup_lon"],
                        row_j["dropoff_lat"], row_j["dropoff_lon"],
                    )
                    d2d1 = haversine(
                        row_j["dropoff_lat"], row_j["dropoff_lon"],
                        row_i["dropoff_lat"], row_i["dropoff_lon"],
                    )
                    delay_i = ((p1p2 + p2d2 + d2d1) - solo_vmt_i) / TRAVEL_SPEED_MPH * 60.0
                    delay_j = (p2d2 - solo_vmt_j) / TRAVEL_SPEED_MPH * 60.0
                max_delay_min = max(delay_i, delay_j)

                if shared_vmt > solo_total:
                    vmt = solo_vmt_i
                    reward = -VMT_REJECT_PEN
                    vmt_increase_flag = 1
                    self.assigned.add(self.current_idx)
                else:
                    vmt_saved = solo_total - shared_vmt
                    cost_solo_total = (d_i + d_j) * FARE_PER_MILE
                    cost_shared = cost_solo_total * SHARED_DISCOUNT
                    cost_saved = cost_solo_total - cost_shared

                    vmt = shared_vmt
                    reward = (
                        W_VMT_SAVED * vmt_saved
                        + W_COST_SAVED * cost_saved
                    )
                    shared = 2
                    vmt_saved_step = vmt_saved
                    self.assigned.add(self.current_idx)
                    self.assigned.add(j)

        while (
            self.current_idx < self.n_trips
            and self.current_idx in self.assigned
        ):
            self.current_idx += 1

        done = self.current_idx >= self.n_trips

        info = {
            "vmt": vmt,
            "shared": shared,
            "vmt_inc_reject": vmt_increase_flag,
            "vmt_saved": vmt_saved_step,
            "max_delay_min": max_delay_min,
        }
        return self._get_obs(), reward, done, False, info
