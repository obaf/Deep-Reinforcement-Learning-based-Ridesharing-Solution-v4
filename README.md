# Ride-Pooling DRL on NYC HVFHV: Double Dueling DQN with PER and N-step returns

A reinforcement-learning dispatcher that decides, request by request,
whether an incoming ride-hail trip should go solo or be matched with
one of the next ten queued requests. Trained on a balanced stratified
sample of NYC TLC HVFHV (Uber, Lyft) trip records.

The policy is a **Double Dueling Deep Q-Network**, trained with
**Prioritised Experience Replay (PER)** and **3-step bootstrap returns**.
Eleven discrete actions (solo plus ten lookahead candidates) are filtered
through **hard action masking** so the Q-function is not poisoned by
large feasibility penalties during exploration.

This repository ships with a trained checkpoint and the held-out test
workbook, so a fresh clone can produce a trip execution plan without
re-running the data pipeline.

---

## Contents

1. [Quick start](#quick-start)
2. [Repository layout](#repository-layout)
3. [Setup](#setup)
4. [Run inference with the shipped model](#run-inference-with-the-shipped-model)
5. [Train from scratch](#train-from-scratch)
6. [Problem formulation](#problem-formulation)
7. [Algorithm](#algorithm)
8. [Hyperparameters](#hyperparameters)
9. [Trip execution plan: schema](#trip-execution-plan-schema)
10. [Experiments, results and discussion](#experiments-results-and-discussion)
11. [Data source and attribution](#data-source-and-attribution)
12. [Citation](#citation)
13. [Licence](#licence)

---

## Quick start

Two supported workflows. Both assume Python 3.10 or newer and a fresh
virtual environment with `requirements.txt` installed.

**1. Inference using the shipped model**

```bash
python -m agent.inference
```

Reads `data/processed/test.xlsx`, loads `models/model.pth`, and writes
`analysis/trip_execution_plan_v4.xlsx`. A summary table is printed to
stdout.

**2. Training from scratch**

```bash
python scripts/01_download_tlc.py
python scripts/02_clean_filter.py
python scripts/03_zone_to_centroid.py
python scripts/04_sample_balanced.py
python scripts/05_split_train_test.py
python -m agent.train
python -m agent.inference
```

The five pipeline scripts are idempotent. Total wall-clock from a clean
working tree is roughly thirty minutes for the data pipeline plus around
an hour for training on a modern laptop CPU.

---

## Repository layout

```text
.
|-- README.md
|-- requirements.txt
|-- requirements.lock.txt
|
|-- scripts/                  Data pipeline (run in order)
|   |-- 01_download_tlc.py    Fetch HVFHV parquet and zone shapefile
|   |-- 02_clean_filter.py    Drop NaNs, apply plausibility filters
|   |-- 03_zone_to_centroid.py  Resolve zone IDs to centroid lat/lon
|   |-- 04_sample_balanced.py   Stratified balanced sample
|   |-- 05_split_train_test.py  Train, test, optional robustness split
|
|-- env/                      Gymnasium environment
|   |-- ride_pool_env.py      State, action, step, action_mask
|   |-- reward.py             Reward constants
|
|-- agent/                    DQN agent and training loop
|   |-- dqn.py                Dueling network
|   |-- replay.py             Sum-tree PER plus N-step buffer
|   |-- train.py              Training loop and checkpointing
|   |-- inference.py          Greedy episode and plan emitter
|
|-- analysis/                 Post-inference analysis
|   |-- generating_stats_v3.docx
|   |-- Stats.docx
|   |-- trip_execution_plan_v4.xlsx
|
|-- data/processed/
|   |-- test.xlsx             Held-out test workbook (shipped)
|   |-- train.xlsx            Training workbook (shipped)
|
|-- models/
|   |-- model.pth             Trained checkpoint (shipped)
```

The `data/raw/`, `data/interim/`, and `logs/` folders are gitignored and
will be created locally by the pipeline.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                  # Windows
source .venv/bin/activate               # macOS or Linux
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.lock.txt` carries the full transitive snapshot if exact
reproduction of the original environment is required.

CPU is the assumed compute target. The network has roughly 70k
parameters, so a single modern CPU core trains it comfortably. For CUDA
training, install the matching `torch` wheel and remove the
`torch.set_num_threads` line at the top of `agent/train.py`.

---

## Run inference with the shipped model

```bash
python -m agent.inference
```

CLI flags:

| Flag                       | Default                                       | Notes                                                  |
| -------------------------- | --------------------------------------------- | ------------------------------------------------------ |
| `--trip-file PATH`         | `data/processed/test.xlsx`                    | Any workbook with the four required columns.           |
| `--checkpoint PATH`        | `models/model.pth`                            | Use an alternative checkpoint.                         |
| `--out PATH`               | `analysis/trip_execution_plan_v4.xlsx`        | Destination for the execution plan.                    |

Required columns in the trip file:

| Column         | Type  |
| -------------- | ----- |
| `pickup_lat`   | float |
| `pickup_lon`   | float |
| `dropoff_lat`  | float |
| `dropoff_lon`  | float |

Optional, used when present:

| Column                  | Type             | Description                                                                 |
| ----------------------- | ---------------- | --------------------------------------------------------------------------- |
| `shared_request_flag`   | `Y`/`N` or 0/1   | Platform-side pool opt-in flag, used as a state feature.                    |
| `tpep_pickup_datetime`  | datetime         | Carried into the output for downstream visualisation tools.                 |

Trip order in the workbook is the order of decisions. If the destination
file is locked (open in Excel), the script writes to a `*.new.xlsx`
sibling instead of failing.

After inference, an optional aggregate analysis docx and figures can be
produced with:

```bash
python -m analysis.make_outputs
```

---

## Train from scratch

```bash
python -m agent.train
```

Loads `data/processed/train.xlsx`, walks one full pass of the trip
stream per episode, and writes per-episode metrics to
`logs/training_log.csv` plus a fresh `models/model.pth` after each
episode. Training stops when the moving-average reward has not improved
for `PLATEAU_PATIENCE = 80` episodes, or at the `EPISODES_MAX = 600`
cap.

CLI flags:

| Flag                | Default                                  | Notes                                  |
| ------------------- | ---------------------------------------- | -------------------------------------- |
| `--episodes N`      | 600                                      | Episode cap.                           |
| `--data-file PATH`  | `data/processed/train.xlsx`              | Override the training workbook.        |

Resuming is automatic: `agent.train` saves the online network, target
network, optimiser state, episode counter, and current `epsilon` to the
checkpoint after every episode, and restores them when re-run.

Typical wall-clock on a modern laptop CPU:

| Workload                                              | Time      |
| ----------------------------------------------------- | --------- |
| Smaller subset (10k rows), 50 episodes                | ~5 min    |
| Full `train.xlsx` (~135k rows), 100 episodes          | ~50 min   |
| Full training to plateau                              | ~1-2 h    |

---

## Problem formulation

The dispatcher is modelled as an episodic Markov Decision Process. One
episode is one pass over the trip stream in arrival order.

**State** $s \in \mathbb{R}^{50}$. For each of the next ten candidate
trips after the current leader, five features:

| Feature           | Meaning                                                              |
| ----------------- | -------------------------------------------------------------------- |
| `p_dist`          | Pickup-to-pickup haversine distance (mi).                            |
| `d_dist`          | Dropoff-to-dropoff haversine distance (mi).                          |
| `is_avail`        | 1 if the candidate is still in the lookahead window and unassigned.  |
| `vmt_saved_est`   | Estimated shared-VMT savings against the solo-solo baseline (mi).    |
| `shared_flag`     | 1 if the candidate's `shared_request_flag` is `Y`, else 0.           |

**Action** $a \in \{0, 1, \dots, 10\}$. Action 0 serves the leader
solo; action $k$ pairs the leader with the candidate at queue offset
$k$.

**Reward**:

```
r_solo    = 0
r_shared  = W_VMT_SAVED  * (solo_total_vmt - shared_vmt)
          + W_COST_SAVED * (cost_solo_total - cost_actual_total)
```

with `W_VMT_SAVED = 5.0`, `W_COST_SAVED = 1.0`, `SHARED_DISCOUNT = 0.65`
(i.e. a successful share charges 65% of the solo fare). A pairing whose
shared route exceeds the sum of the two solos triggers a hard
`VMT_REJECT_PEN = 1000.0` negative reward and falls back to solo. Pickup
delay is computed and reported but does not enter the reward.

**Action masking**. The environment exposes
`RidePoolEnv.action_mask(obs)` which reads availability and
`vmt_saved_est` directly off the observation and zeroes out infeasible
actions. Training and inference both apply it before the argmax:

```python
mask = env.action_mask(obs)
q_masked = np.where(mask, q, -np.inf)
action = int(np.argmax(q_masked))
```

This avoids the failure mode where huge feasibility penalties dominate
early-exploration replay batches and collapse the policy to
always-solo.

---

## Algorithm

The agent is a value-based DQN with three modifications layered on top
of the original DQN architecture.

**Dueling network** (`agent/dqn.py`). The trunk feeds two heads: a
scalar value $V(s)$ and an 11-vector advantage $A(s, \cdot)$. The output
is recombined as

$$Q(s, a) \;=\; V(s) \;+\; \Bigl(A(s, a) \;-\; \tfrac{1}{|\mathcal{A}|}\textstyle\sum_{a'} A(s, a')\Bigr).$$

Trunk: 3 hidden layers of 128 units, LayerNorm, ReLU, Kaiming init.

**Double DQN target**. The online network selects the next action; the
target network evaluates it:

$$a^{*} \;=\; \arg\max_{a'} \, Q_{\text{online}}(s', a'; \theta),$$

$$y \;=\; r \;+\; \gamma \, Q_{\text{target}}(s', a^{*}; \theta^{-}).$$

The target net is a Polyak-averaged copy of the online net
($\tau = 0.005$ per gradient step).

**Prioritised Experience Replay** (`agent/replay.py`). A sum-tree of
capacity 50,000 stores transitions with priority

$$p_i \;=\; \bigl(|\delta_i| + \varepsilon\bigr)^{\alpha},$$

where $\delta_i$ is the TD-error of transition $i$. Importance sampling
corrects for the non-uniform sampling with weights

$$w_i \;=\; \Bigl(N \cdot \dfrac{p_i}{\sum_j p_j}\Bigr)^{-\beta},$$

and $\beta$ is annealed from 0.4 to 1.0 across training.

**3-step bootstrap returns**. The N-step buffer accumulates three
consecutive transitions and writes a single compound experience with
return

$$R^{(3)}_t \;=\; r_t \;+\; \gamma \, r_{t+1} \;+\; \gamma^{2} \, r_{t+2}$$

and the state three steps ahead. This trades a small amount of bias for
noticeably better signal-to-noise on the gradient.

Combined, this is four of the six Rainbow ingredients. Distributional
value estimation and noisy nets are not used.

---

## Hyperparameters

Single source of truth at the top of `agent/train.py`.

| Parameter                                  | Value             |
| ------------------------------------------ | ----------------- |
| `HIDDEN_DIM`                               | 128               |
| `N_HIDDEN_LAYERS`                          | 3                 |
| `GAMMA`                                    | 0.99              |
| `LR`                                       | 1e-4 (Adam)       |
| `BATCH_SIZE`                               | 256               |
| `REPLAY_CAPACITY`                          | 50,000            |
| `PER_ALPHA`                                | 0.6               |
| `PER_BETA_START`, `PER_BETA_END`           | 0.4 -> 1.0        |
| `N_STEP`                                   | 3                 |
| `TAU`                                      | 0.005             |
| `GRAD_CLIP_NORM`                           | 10.0              |
| `EPSILON_START`, `EPSILON_MIN`, decay      | 1.0 -> 0.05, x0.97/episode |
| `EPISODES_MAX`                             | 600               |
| `PLATEAU_PATIENCE`                         | 80                |
| `MIN_EPS_BEFORE_PLATEAU`                   | 60                |
| `MOVING_AVG_WINDOW`                        | 10                |
| `WARMUP_TRANSITIONS`                       | 1,000             |

The PyTorch thread count can be overridden with the
`TRIP_SIM_TORCH_THREADS` environment variable (default 4).

---

## Trip execution plan: schema

`analysis/trip_execution_plan_v4.xlsx` has one row per decision, in the
order the agent walked through the trip stream.

| Column                                     | Description                                                         |
| ------------------------------------------ | ------------------------------------------------------------------- |
| `decision_id`                              | Sequential, 0-indexed.                                              |
| `trip_i_idx`                               | Row index of the leader trip in the source workbook.                |
| `action`                                   | 0 = solo; `k` = pair with `trip_i_idx + k`.                         |
| `partner_idx`                              | Row index of the partner (blank for non-shared outcomes).           |
| `outcome`                                  | `solo`, `shared`, `fallback_solo`, or `vmt_reject_solo`.            |
| `dist_i_mi`, `dist_j_mi`                   | Per-rider solo haversine distance (mi).                             |
| `solo_vmt_total_mi`                        | Counterfactual fleet VMT if both rode solo.                         |
| `actual_vmt_mi`                            | Fleet VMT actually driven under the chosen action.                  |
| `vmt_saved_mi`                             | Difference; non-negative by construction.                           |
| `delay_i_min`, `delay_j_min`               | Per-rider extra in-vehicle time (min). Non-zero on shared rows.     |
| `max_delay_min`                            | Max of the two delays on shared rows.                               |
| `cost_solo_i_usd`, `cost_solo_j_usd`       | Per-rider solo fare at `FARE_PER_MILE = $9.25`.                     |
| `cost_actual_i_usd`, `cost_actual_j_usd`   | Per-rider actual fare (with 35% discount on shared).                |
| `cost_solo_total_usd`, `cost_actual_total_usd`, `cost_saved_usd` | Totals.                                       |
| `pickup_lat_i`, `pickup_lon_i`, `dropoff_lat_i`, `dropoff_lon_i` | Leader coordinates.                           |
| `pickup_lat_j`, `pickup_lon_j`, `dropoff_lat_j`, `dropoff_lon_j` | Partner coordinates (blank for non-shared).   |
| `pickup_datetime_i`                        | Leader pickup datetime carried through from the source file.        |

Outcome semantics:

* `solo`: agent chose action 0.
* `shared`: agent paired the leader with a candidate whose merged route
  is strictly shorter than the sum of solos. Both riders are served on
  one vehicle.
* `vmt_reject_solo`: agent picked a candidate, but the merged route
  would have increased fleet VMT. The env serves the leader solo and
  applies the VMT-reject penalty. The candidate stays in the queue.
* `fallback_solo`: agent picked a candidate that was already assigned
  or had fallen out of the lookahead window.

---

## Experiments, results and discussion

The numbers below were produced from `analysis/trip_execution_plan_v4.xlsx`,
the workbook written by `agent/inference.py` after a greedy evaluation of
the trained checkpoint on the held-out test week (2019-04-08 to
2019-04-14, 15,103 HVFHV trips). Each subsection shows the Excel
formulas that recover the headline figure directly from the workbook
(open `trip_execution_plan_v4.xlsx`, default column order, data starts
on row 2), followed by the corresponding result table.

### Experimental setup

Training proceeded under cumulative checkpointing, with $\varepsilon$
annealed from 1.0 toward $\varepsilon_{\min} = 0.05$ at a per-episode
multiplicative rate of 0.97. The target network was updated by Polyak
averaging (soft-update coefficient 0.005), gradients were clipped at
norm 10, and the prioritised buffer held the last 50,000 N-step
transitions. Training was terminated by the early-stop controller after
the 10-episode reward moving average remained below its best value for
80 consecutive episodes; the checkpoint analysed below is the policy
held at that early-stop point.

### Decision composition

On the held-out test week, the greedy policy produced 13,067 decisions
covering 15,103 riders. Of those, 4,072 riders were served via shared
trips (i.e. 2,036 successful pairings) and 11,031 were served solo,
giving a rider-level share rate of 26.96%.

**Excel formulas.**

| Row label | Formula |
| --- | --- |
| Successful share | `=COUNTIF(E:E,"shared")` |
| Solo service (agent chose solo) | `=COUNTIF(E:E,"solo")` |
| Solo (no feasible partner in window) | `=COUNTIF(E:E,"fallback_solo")` |
| Solo (after VMT-increase rejection) | `=COUNTIF(E:E,"vmt_reject_solo")` |
| Total decisions | `=COUNTA(E:E)-1` |

**Result.**

| Decision outcome | Count |
| --- | --- |
| Successful share | 2,036 |
| Solo service (chosen) | 11,031 |
| Solo (no feasible partner) | 0 |
| Solo (after VMT-increase rejection) | 0 |
| Total decisions | 13,067 |

### Vehicle miles travelled

Compared to a counterfactual in which every rider is served solo, the
agent's policy drives the fleet a total of 43,493.8 mi against the
solo baseline of 46,019.7 mi. The net change is 2,525.9 mi fewer
miles, a 5.49% reduction in total VMT. The hard VMT-rejection rule
guarantees that no shared trip ever appears with negative VMT savings,
so the magnitude of the reduction reflects how often the agent finds
genuinely overlapping requests in the queue.

**Excel formulas.**

| Row label | Formula |
| --- | --- |
| Solo-only baseline VMT (mi) | `=SUM(H:H)` |
| Policy VMT (mi) | `=SUM(I:I)` |
| VMT saved by sharing (mi) | `=SUM(H:H)-SUM(I:I)` |
| VMT reduction (%) | `=100*(SUM(H:H)-SUM(I:I))/SUM(H:H)` |

**Result.**

| Quantity | Value |
| --- | --- |
| Solo-only baseline VMT | 46,019.7 mi |
| Policy VMT | 43,493.8 mi |
| VMT saved by sharing | 2,525.9 mi |
| VMT reduction (%) | 5.49% |

### Rider cost

At a per-mile fare of $9.25 and a shared-trip discount of 35% on the
solo fare, the all-solo baseline charges riders $425,682 in total. The
trained policy lowers that figure to $363,547, a reduction of $62,135
(14.60%). On average, each rider actually matched into a shared trip
paid $15.26 less than they would have under solo service.

**Excel formulas.**

| Row label | Formula |
| --- | --- |
| Solo-only baseline cost (USD) | `=SUM(R:R)` |
| Policy cost (USD) | `=SUM(S:S)` |
| Total cost saved (USD) | `=SUM(R:R)-SUM(S:S)` |
| Cost reduction (%) | `=100*(SUM(R:R)-SUM(S:S))/SUM(R:R)` |
| Average saving per shared rider (USD) | `=(SUMIF(E:E,"shared",N:N)+SUMIF(E:E,"shared",O:O)-SUMIF(E:E,"shared",P:P)-SUMIF(E:E,"shared",Q:Q))/(2*COUNTIF(E:E,"shared"))` |

**Result.**

| Quantity | Value |
| --- | --- |
| Solo-only baseline cost | $425,682 |
| Policy cost | $363,547 |
| Total cost saved | $62,135 |
| Cost reduction (%) | 14.60% |
| Average saving per shared rider | $15.26 |

### Pickup delay (descriptive)

Although delay no longer enters the training reward, the environment
still records the in-vehicle time penalty for every shared rider so the
operational consequence of the two-objective reward can be reported. On
the 2,036 successful shared trips, the worst observed delay across
either rider was 37.06 min; the mean was 2.43 min, the median 0.69 min,
and the 95th percentile 10.47 min. 54.00% of pooled riders experienced
more than half a minute of additional in-vehicle time.

**Excel formulas.** First build a stacked-delay scratch column over
shared rows. In an empty area (column AD), paste the formula below
into `AD2` and drag down to row `2*N + 1`, where
`N = COUNTIF(E:E,"shared")`:

```
=IFERROR(
  INDEX(K:K, SMALL(IF($E$2:$E$100000="shared", ROW($E$2:$E$100000)), ROWS($AD$2:AD2))),
  INDEX(L:L, SMALL(IF($E$2:$E$100000="shared", ROW($E$2:$E$100000)), ROWS($AD$2:AD2) - COUNTIF(E:E,"shared")))
)
```

Enter as an array formula on legacy Excel via Ctrl+Shift+Enter; modern
Excel handles it as a regular formula. (Simpler manual alternative:
filter the table by `outcome = "shared"`, copy column `K` (`delay_i_min`)
into `AD`, then copy column `L` (`delay_j_min`) directly below it.)
The five statistics are:

| Row label | Formula |
| --- | --- |
| Mean extra in-vehicle time (min) | `=AVERAGE(AD:AD)` |
| Median extra in-vehicle time (min) | `=MEDIAN(AD:AD)` |
| 95th-percentile extra in-vehicle time (min) | `=PERCENTILE.INC(AD:AD,0.95)` |
| Maximum extra in-vehicle time (min) | `=MAX(AD:AD)` |
| Riders with any noticeable delay (> 0.5 min, %) | `=100*COUNTIF(AD:AD,">0.5")/COUNT(AD:AD)` |

**Result.**

| Statistic (across all pooled riders) | Value |
| --- | --- |
| Mean extra in-vehicle time | 2.43 min |
| Median extra in-vehicle time | 0.69 min |
| 95th-percentile extra in-vehicle time | 10.47 min |
| Maximum extra in-vehicle time | 37.06 min |
| Riders with any noticeable delay (> 0.5 min) | 54.00% |

### Discussion

Three important observations:

First, the VMT direction is the right one: the policy reduces total
fleet mileage rather than increasing it, which is the headline
operational outcome the reward was meant to produce. The
hard VMT-rejection rule guarantees that no shared trip ever appears
with negative VMT savings, and the magnitude of the reduction in the
VMT table reflects how often the agent finds genuinely overlapping
requests in the queue.

Second, the cost saving in the rider-cost table is a direct consequence
of the fare-discount mechanism. Every successful share produces a
fixed fractional saving, so the total amount transferred to riders
scales linearly with the share count.

Third, the delay distribution is not directly regulated by the reward
but remains operationally tame: VMT-saving pairs are by construction
also short-detour pairs, so removing the explicit delay penalty did not
produce a long-detour failure mode.

---

## Data source and attribution

Training and test data are derived from the **NYC Taxi and Limousine
Commission High-Volume For-Hire Vehicle (HVFHV) Trip Records**,
published monthly by the TLC as anonymised parquet files. The records
contain no rider PII; pickup and dropoff are reported at the level of
the 263 NYC taxi zones rather than as GPS coordinates.

URLs hit by the pipeline:

| Artefact                | URL                                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| Monthly HVFHV parquet   | `https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_YYYY-MM.parquet`          |
| Taxi zone lookup CSV    | `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv`                         |
| Taxi zone shapefile zip | `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip`                               |
| TLC landing page        | `https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page`                            |

No raw data is redistributed by this repository.

---

## Citation

If this repository or its results are used in academic work, please
cite the underlying methods and the dataset:

* Mnih, V. et al. (2015). Human-level control through deep reinforcement
  learning. Nature, 518: 529-533.
* Van Hasselt, H., Guez, A., Silver, D. (2016). Deep reinforcement
  learning with double Q-learning. AAAI.
* Wang, Z. et al. (2016). Dueling network architectures for deep
  reinforcement learning. ICML.
* Schaul, T., Quan, J., Antonoglou, I., Silver, D. (2016). Prioritized
  experience replay. ICLR.
* Sutton, R. S. (1988). Learning to predict by the methods of temporal
  differences. Machine Learning, 3: 9-44.
* Hessel, M. et al. (2018). Rainbow: combining improvements in deep
  reinforcement learning. AAAI.
* Huang, S., Ontanon, S. (2020). A closer look at invalid action
  masking in policy gradient algorithms. arXiv:2006.14171.
* NYC Taxi and Limousine Commission. HVFHV Trip Records, 2019.

---

## Licence

The source code in this repository is released under the MIT licence.
The HVFHV trip records are NYC TLC open data and are not redistributed
here; their terms of use are documented at the NYC TLC landing page
linked above.
