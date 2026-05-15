"""Training loop for the Double Dueling DQN with PER and 3-step returns."""
from __future__ import annotations

import datetime
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Allow tuning the torch thread count via env var.
_DESIRED_THREADS = int(os.environ.get("TRIP_SIM_TORCH_THREADS", 4))
torch.set_num_threads(_DESIRED_THREADS)
torch.set_num_interop_threads(1)

from agent.dqn import DQN
from agent.replay import NStepPrioritizedReplay
from env.ride_pool_env import RidePoolEnv


HIDDEN_DIM = 128
N_HIDDEN_LAYERS = 3
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 256
REPLAY_CAPACITY = 50_000
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
N_STEP = 3
TAU = 0.005
GRAD_CLIP_NORM = 10.0
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.97
EPISODES_MAX = 600
PLATEAU_PATIENCE = 80
MIN_EPS_BEFORE_PLATEAU = 60
MOVING_AVG_WINDOW = 10
WARMUP_TRANSITIONS = 1_000


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_train_xlsx() -> Path:
    return _project_root() / "data" / "processed" / "train.xlsx"


def _default_checkpoint() -> Path:
    return _project_root() / "models" / "model.pth"


def _default_log_dir() -> Path:
    return _project_root() / "logs"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    target_model: nn.Module,
    optimizer: optim.Optimizer,
    epsilon: float,
    episode: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_model_state_dict": target_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epsilon": epsilon,
            "episode": episode,
        },
        path,
    )


def _load_checkpoint_into(
    ckpt_path: Path,
    model: DQN,
    target_model: DQN,
    optimizer: optim.Optimizer,
) -> tuple[DQN, DQN, optim.Optimizer, float, int]:
    """Restore from a checkpoint if present; otherwise start fresh."""
    if not ckpt_path.exists():
        target_model.load_state_dict(model.state_dict())
        print("No checkpoint found - starting fresh.", flush=True)
        return model, target_model, optimizer, EPSILON_START, 0

    print(f"Loading checkpoint from {ckpt_path}", flush=True)
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            target_model.load_state_dict(
                ckpt.get("target_model_state_dict", ckpt["model_state_dict"])
            )
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            eps = float(ckpt.get("epsilon", EPSILON_MIN))
            start_ep = int(ckpt.get("episode", 0)) + 1
            print(f"Resumed at episode {start_ep} with epsilon={eps:.3f}")
            return model, target_model, optimizer, eps, start_ep
        except (RuntimeError, KeyError) as e:
            print(f"Could not load checkpoint into current architecture ({e}).")
            print("Starting a fresh model under the current architecture.")
            return model, target_model, optimizer, EPSILON_START, 0

    try:
        model.load_state_dict(ckpt)
        target_model.load_state_dict(model.state_dict())
        print("Loaded legacy state_dict-only checkpoint.")
        return model, target_model, optimizer, EPSILON_MIN, 0
    except (RuntimeError, KeyError) as e:
        print(f"Could not load legacy checkpoint ({e}). Starting fresh.")
        return model, target_model, optimizer, EPSILON_START, 0


def _beta_at(step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return PER_BETA_END
    frac = min(1.0, max(0.0, step / float(total_steps)))
    return PER_BETA_START + frac * (PER_BETA_END - PER_BETA_START)


def train(
    data_file: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    log_dir: str | Path | None = None,
    episodes: int = EPISODES_MAX,
    seed: int | None = 0,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    data_file = Path(data_file) if data_file is not None else _default_train_xlsx()
    checkpoint_path = (
        Path(checkpoint_path) if checkpoint_path is not None else _default_checkpoint()
    )
    log_dir = Path(log_dir) if log_dir is not None else _default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_file}. Run scripts/01..05 first."
        )

    env = RidePoolEnv(str(data_file))
    n_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    model = DQN(obs_dim, n_actions, hidden_dim=HIDDEN_DIM,
                n_hidden_layers=N_HIDDEN_LAYERS)
    target_model = DQN(obs_dim, n_actions, hidden_dim=HIDDEN_DIM,
                       n_hidden_layers=N_HIDDEN_LAYERS)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model, target_model, optimizer, epsilon, start_episode = _load_checkpoint_into(
        checkpoint_path, model, target_model, optimizer
    )

    buffer = NStepPrioritizedReplay(
        capacity=REPLAY_CAPACITY,
        n_step=N_STEP,
        gamma=GAMMA,
        alpha=PER_ALPHA,
    )

    log_path = log_dir / "training_log.csv"
    snapshot_path = log_dir / "reward_snapshots.csv"

    log_is_new = not log_path.exists()
    snap_is_new = not snapshot_path.exists()
    log_file = open(log_path, "a", buffering=1)
    snap_file = open(snapshot_path, "a", buffering=1)
    if log_is_new:
        log_file.write(
            "timestamp_iso,wall_time_s,episode,reward,vmt,shared,vmt_saved,"
            "vmt_increase_rejects,epsilon,beta,"
            f"reward_ma{MOVING_AVG_WINDOW},vmt_ma{MOVING_AVG_WINDOW},"
            f"shared_ma{MOVING_AVG_WINDOW},vmt_saved_ma{MOVING_AVG_WINDOW}\n"
        )
    if snap_is_new:
        snap_file.write(
            "timestamp_iso,wall_time_s,episode,episode_reward,reward_ma,"
            "best_reward_ma,epsilon,plateau_episodes_since_best_ma,note\n"
        )

    reward_hist: deque[float] = deque(maxlen=MOVING_AVG_WINDOW)
    vmt_hist: deque[float] = deque(maxlen=MOVING_AVG_WINDOW)
    shared_hist: deque[int] = deque(maxlen=MOVING_AVG_WINDOW)
    vmt_saved_hist: deque[float] = deque(maxlen=MOVING_AVG_WINDOW)

    best_reward_ma = float("-inf")
    plateau_episodes = 0
    last_reward_ma = 0.0

    training_start_time = time.time()
    last_snapshot_wall = training_start_time
    last_eval_time = time.time()

    approx_steps_per_episode = max(1, env.n_trips)
    beta_anneal_total = approx_steps_per_episode * max(1, episodes // 2)
    global_step = 0

    def write_snapshot(ep: int, episode_reward: float, ma_value: float, note: str) -> None:
        nonlocal last_snapshot_wall
        now = time.time()
        ts_iso = datetime.datetime.now().isoformat(timespec="seconds")
        snap_file.write(
            f"{ts_iso},{now - training_start_time:.2f},{ep},{episode_reward:.4f},"
            f"{ma_value:.4f},{best_reward_ma:.4f},{epsilon:.6f},"
            f"{plateau_episodes},{note}\n"
        )
        last_snapshot_wall = now

    end_episode = start_episode + episodes
    for ep in range(start_episode, end_episode):
        obs, _ = env.reset()
        buffer.reset_n_step()
        done = False

        total_reward = 0.0
        total_vmt = 0.0
        total_shared = 0
        total_vmt_saved = 0.0
        n_vmt_inc_reject = 0

        while not done:
            mask = env.action_mask(obs)
            feasible = np.flatnonzero(mask)
            if random.random() < epsilon:
                action = int(np.random.choice(feasible))
            else:
                with torch.no_grad():
                    q = (
                        model(torch.from_numpy(obs).float().unsqueeze(0))
                        .squeeze(0).numpy()
                    )
                q_masked = np.where(mask, q, -np.inf)
                action = int(np.argmax(q_masked))

            next_obs, reward, done, _, info = env.step(action)
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            global_step += 1

            total_reward += float(reward)
            total_vmt += float(info["vmt"])
            total_shared += int(info["shared"])
            total_vmt_saved += float(info.get("vmt_saved", 0.0))
            n_vmt_inc_reject += int(info.get("vmt_inc_reject", 0))

            if len(buffer) >= max(BATCH_SIZE, WARMUP_TRANSITIONS):
                beta = _beta_at(global_step, beta_anneal_total)
                batch = buffer.sample(BATCH_SIZE, beta)

                states = torch.from_numpy(batch["states"]).float()
                actions = torch.from_numpy(batch["actions"]).long()
                rewards = torch.from_numpy(batch["rewards"]).float()
                next_states = torch.from_numpy(batch["next_states"]).float()
                dones = torch.from_numpy(batch["dones"]).float()
                n_step = torch.from_numpy(batch["n_step"]).float()
                weights = torch.from_numpy(batch["weights"]).float()

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_actions = model(next_states).argmax(dim=1, keepdim=True)
                    next_q = (
                        target_model(next_states)
                        .gather(1, next_actions)
                        .squeeze(1)
                    )
                    discount = (GAMMA ** n_step) * (1.0 - dones)
                    target = rewards + discount * next_q

                td_errors = target - q_values
                loss = (weights * td_errors.pow(2)).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

                with torch.no_grad():
                    for tp, p in zip(target_model.parameters(), model.parameters()):
                        tp.data.mul_(1.0 - TAU).add_(p.data, alpha=TAU)

                buffer.update_priorities(
                    batch["indices"], td_errors.detach().abs().cpu().numpy()
                )

            if time.time() - last_eval_time > 600:
                save_checkpoint(checkpoint_path, model, target_model,
                                optimizer, epsilon, ep)
                last_eval_time = time.time()
                print(
                    f"  [periodic] saved checkpoint @ ep {ep}, "
                    f"avg VMT/trip so far: "
                    f"{total_vmt / (env.current_idx + 1):.2f}, "
                    f"shares: {total_shared}",
                    flush=True,
                )

            if time.time() - last_snapshot_wall >= 60.0:
                write_snapshot(ep, total_reward, last_reward_ma,
                               "within_episode_partial")

        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY
            epsilon = max(epsilon, EPSILON_MIN)

        reward_hist.append(total_reward)
        vmt_hist.append(total_vmt)
        shared_hist.append(total_shared)
        vmt_saved_hist.append(total_vmt_saved)
        reward_ma = sum(reward_hist) / len(reward_hist)
        vmt_ma = sum(vmt_hist) / len(vmt_hist)
        shared_ma = sum(shared_hist) / len(shared_hist)
        vmt_saved_ma = sum(vmt_saved_hist) / len(vmt_saved_hist)

        if reward_ma > best_reward_ma + 1e-6:
            best_reward_ma = reward_ma
            plateau_episodes = 0
            trend_note = "episode_end_new_best_ma"
        else:
            plateau_episodes += 1
            trend_note = "episode_end"

        last_reward_ma = reward_ma
        ts_iso = datetime.datetime.now().isoformat(timespec="seconds")
        wall_s = time.time() - training_start_time
        cur_beta = _beta_at(global_step, beta_anneal_total)

        snap_file.write(
            f"{ts_iso},{wall_s:.2f},{ep},{total_reward:.4f},{reward_ma:.4f},"
            f"{best_reward_ma:.4f},{epsilon:.6f},{plateau_episodes},{trend_note}\n"
        )
        last_snapshot_wall = time.time()

        log_file.write(
            f"{ts_iso},{wall_s:.2f},{ep},{total_reward:.4f},"
            f"{total_vmt:.4f},{total_shared},{total_vmt_saved:.4f},"
            f"{n_vmt_inc_reject},{epsilon:.6f},{cur_beta:.4f},"
            f"{reward_ma:.4f},{vmt_ma:.4f},{shared_ma:.4f},{vmt_saved_ma:.4f}\n"
        )

        print(
            f"[{ts_iso}] Episode {ep} | Reward: {total_reward:.2f} | "
            f"VMT: {total_vmt:.2f} | VMT_saved: {total_vmt_saved:.2f} | "
            f"Shares: {total_shared} | VMT_inc_rej: {n_vmt_inc_reject} | "
            f"Eps: {epsilon:.3f} | Beta: {cur_beta:.3f} || "
            f"MA{MOVING_AVG_WINDOW} -> Reward: {reward_ma:.2f} | "
            f"VMT: {vmt_ma:.2f} | Shares: {shared_ma:.2f} | "
            f"VMT_saved: {vmt_saved_ma:.2f}",
            flush=True,
        )

        save_checkpoint(checkpoint_path, model, target_model,
                        optimizer, epsilon, ep)

        if (
            ep - start_episode >= MIN_EPS_BEFORE_PLATEAU
            and plateau_episodes >= PLATEAU_PATIENCE
        ):
            print(
                f"Early stop at episode {ep}: "
                f"{MOVING_AVG_WINDOW}-episode reward MA unchanged for "
                f"{PLATEAU_PATIENCE} episodes (best MA: {best_reward_ma:.2f}).",
                flush=True,
            )
            break

    log_file.close()
    snap_file.close()


def _parse_cli_args() -> dict:
    import argparse
    p = argparse.ArgumentParser(description="Train the Double Dueling DQN.")
    p.add_argument("--data-file", default=None,
                   help="path to the training Excel "
                        "(default: data/processed/train.xlsx)")
    p.add_argument("--checkpoint", default=None,
                   help="path to model.pth "
                        "(default: models/model.pth)")
    p.add_argument("--log-dir", default=None,
                   help="directory for training_log.csv + reward_snapshots.csv "
                        "(default: logs/)")
    p.add_argument("--episodes", type=int, default=EPISODES_MAX,
                   help=f"max episodes to run (default: {EPISODES_MAX})")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (default: 0; set to -1 to disable)")
    args = p.parse_args()
    return {
        "data_file": args.data_file,
        "checkpoint_path": args.checkpoint,
        "log_dir": args.log_dir,
        "episodes": args.episodes,
        "seed": None if args.seed < 0 else args.seed,
    }


if __name__ == "__main__":
    train(**_parse_cli_args())
