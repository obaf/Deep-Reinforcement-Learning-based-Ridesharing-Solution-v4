"""Prioritised experience replay with N-step returns."""
from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


class SumTree:
    """Fixed-capacity sum-tree of priorities backed by a flat array."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.size = 0

    def __len__(self) -> int:
        return self.size

    @property
    def total(self) -> float:
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        if self.size == 0:
            return 1.0
        leaves = self.tree[self.capacity - 1:self.capacity - 1 + self.size]
        if leaves.size == 0:
            return 1.0
        m = float(leaves.max())
        return m if m > 0.0 else 1.0

    def set_size(self, size: int) -> None:
        self.size = int(size)

    def update(self, leaf_idx: int, priority: float) -> None:
        """Update a single leaf priority and propagate the change to the root."""
        change = float(priority) - float(self.tree[leaf_idx])
        self.tree[leaf_idx] = float(priority)
        parent = (leaf_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def batch_update(
        self,
        leaf_idxs: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        """Apply many priority updates at once."""
        leaf_idxs = np.asarray(leaf_idxs, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float64)
        if leaf_idxs.size == 0:
            return
        changes = priorities - self.tree[leaf_idxs]
        self.tree[leaf_idxs] = priorities

        parents = (leaf_idxs - 1) // 2
        while True:
            valid = parents >= 0
            if not valid.any():
                break
            valid_parents = parents[valid]
            valid_changes = changes[valid]
            np.add.at(self.tree, valid_parents, valid_changes)
            parents = np.where(parents == 0, -1, (parents - 1) // 2)

    def get_leaf(self, value: float) -> tuple[int, float]:
        """Return (leaf_idx, priority) for a sample point ``value`` in [0, total]."""
        idx = 0
        tree = self.tree
        size = len(tree)
        while True:
            left = 2 * idx + 1
            if left >= size:
                break
            right = left + 1
            left_p = tree[left]
            if value <= left_p:
                idx = left
            else:
                value -= left_p
                idx = right
        return idx, float(tree[idx])


class NStepPrioritizedReplay:
    """Prioritised replay buffer with N-step return accumulation."""

    def __init__(
        self,
        capacity: int = 50_000,
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        eps: float = 1e-6,
        obs_dim: int | None = None,
    ):
        self.tree = SumTree(int(capacity))
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._n_buf: deque[tuple] = deque(maxlen=self.n_step)
        self._gamma_pow = np.power(
            self.gamma, np.arange(self.n_step, dtype=np.float64),
        )

        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim) if obs_dim is not None else None
        self._states: np.ndarray | None = None
        self._next_states: np.ndarray | None = None
        self._actions = np.empty(self.capacity, dtype=np.int64)
        self._rewards = np.empty(self.capacity, dtype=np.float32)
        self._dones = np.empty(self.capacity, dtype=np.float32)
        self._n_eff = np.empty(self.capacity, dtype=np.int64)
        self._write = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def reset_n_step(self) -> None:
        self._n_buf.clear()

    def _ensure_obs_buffers(self, obs_like: np.ndarray) -> None:
        if self._states is not None:
            return
        self.obs_dim = int(obs_like.shape[-1])
        self._states = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self._next_states = np.empty(
            (self.capacity, self.obs_dim), dtype=np.float32,
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Append a transition and flush completed N-step segments."""
        self._n_buf.append((state, action, float(reward), next_state, bool(done)))

        if len(self._n_buf) >= self.n_step:
            self._flush_one(self.n_step)

        if done:
            while len(self._n_buf) > 0:
                self._flush_one(len(self._n_buf))

    def _flush_one(self, horizon: int) -> None:
        s0, a0, _, _, _ = self._n_buf[0]
        cum_r = 0.0
        last_done = False
        last_next_state = None
        for k in range(horizon):
            _, _, r_k, ns_k, d_k = self._n_buf[k]
            cum_r += self._gamma_pow[k] * r_k
            last_next_state = ns_k
            last_done = d_k
            if d_k:
                horizon = k + 1
                break
        self._store(s0, int(a0), float(cum_r), last_next_state,
                    bool(last_done), int(horizon))
        self._n_buf.popleft()

    def _store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        horizon: int,
    ) -> None:
        self._ensure_obs_buffers(state)
        slot = self._write
        self._states[slot] = state
        self._next_states[slot] = next_state
        self._actions[slot] = action
        self._rewards[slot] = reward
        self._dones[slot] = float(done)
        self._n_eff[slot] = horizon

        leaf_idx = slot + self.capacity - 1
        priority = self.tree.max_priority
        self.tree.update(leaf_idx, priority)

        self._write = (self._write + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1
            self.tree.set_size(self._size)

    def sample(self, batch_size: int, beta: float) -> dict:
        if self._size == 0:
            raise RuntimeError("sample() called on empty replay buffer")
        batch_size = int(batch_size)
        beta = float(beta)

        total = self.tree.total
        if total <= 0.0:
            total = 1e-8

        segment = total / batch_size
        lo = segment * np.arange(batch_size, dtype=np.float64)
        hi = segment * np.arange(1, batch_size + 1, dtype=np.float64)
        vs = np.random.uniform(lo, hi)

        leaf_idxs = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        get_leaf = self.tree.get_leaf
        for i, v in enumerate(vs):
            leaf_idxs[i], priorities[i] = get_leaf(float(v))

        data_idxs = leaf_idxs - (self.capacity - 1)
        if (data_idxs >= self._size).any() or (data_idxs < 0).any():
            populated = max(1, self._size)
            data_idxs = np.clip(data_idxs, 0, populated - 1)
            leaf_idxs = data_idxs + (self.capacity - 1)
            priorities = self.tree.tree[leaf_idxs]

        probs = np.clip(priorities / total, 1e-12, None)
        n = self._size
        weights = (n * probs) ** (-beta)
        weights = weights / weights.max()

        return {
            "states": self._states[data_idxs],
            "actions": self._actions[data_idxs],
            "rewards": self._rewards[data_idxs],
            "next_states": self._next_states[data_idxs],
            "dones": self._dones[data_idxs],
            "n_step": self._n_eff[data_idxs],
            "indices": leaf_idxs,
            "weights": weights.astype(np.float32),
        }

    def update_priorities(
        self,
        indices: Iterable[int],
        td_errors: Iterable[float],
    ) -> None:
        """Refresh priorities for the supplied leaves."""
        idxs = np.asarray(indices, dtype=np.int64)
        errs = np.asarray(td_errors, dtype=np.float64)
        priorities = (np.abs(errs) + self.eps) ** self.alpha
        self.tree.batch_update(idxs, priorities)
