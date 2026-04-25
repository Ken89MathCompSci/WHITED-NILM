"""
Build data/WHITED/train.pkl, val.pkl, test.pkl from WHITED FLAC recordings.

Each pkl is a tuple(DataFrame) matching the AMPds format:
  - DatetimeIndex at 1-minute resolution
  - Columns: main, fridge, microwave, washing machine, kettle
  - Values: active power in watts
  - main = sum of all appliance columns

Approach: compute active power (mean of instantaneous V*I) per FLAC file,
then generate a synthetic 1-day (1440-min) schedule via a Markov chain
for each appliance. Different random seeds per split yield different days.
"""

import os, glob, pickle
import numpy as np
import pandas as pd
import soundfile as sf

WHITED_DIR = os.path.join(os.path.dirname(__file__), "WhiteD", "DATEN")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "data", "WHITED")
os.makedirs(OUT_DIR, exist_ok=True)

MK_FACTORS = {
    "MK1": {"volt": 1033.64, "curr": 61.4835},
    "MK2": {"volt": 861.15,  "curr": 60.200},
    "MK3": {"volt": 988.926, "curr": 60.9562},
}

def _mk(fname):
    for mk in ("MK1", "MK2", "MK3"):
        if mk in fname:
            return mk
    return "MK2"

def active_power(fp):
    data, _ = sf.read(fp)
    f = MK_FACTORS[_mk(os.path.basename(fp))]
    v = data[:, 0] * f["volt"]
    i = data[:, 1] * f["curr"]
    return float(np.mean(v * i))

# ── load measured power pools ──────────────────────────────────────────────
APPLIANCE_PATTERNS = {
    "fridge":           "Fridge",
    "microwave":        "Microwave",
    "washing machine":  "WashingMachine",
    "kettle":           "Kettle",
}

print("Computing active power from FLAC files …")
power_pool = {}
for label, pattern in APPLIANCE_PATTERNS.items():
    files = sorted(glob.glob(os.path.join(WHITED_DIR, f"{pattern}_*.flac")))
    powers = [active_power(f) for f in files]
    power_pool[label] = np.array(powers, dtype=np.float32)
    print(f"  {label:20s}: n={len(powers):3d}  "
          f"mean={np.mean(powers):.0f} W  "
          f"[{np.min(powers):.0f}, {np.max(powers):.0f}] W")

# ── Markov-chain scheduling parameters ────────────────────────────────────
# Each appliance: (p_on_to_off, p_off_to_on, initial_state)
# avg on  = 1 / p_on_to_off   minutes
# avg off = 1 / p_off_to_on   minutes
SCHEDULE_PARAMS = {
    #                       p(on→off)   p(off→on)   init
    "fridge":          (1/25,       1/55,       True ),  # ~31 % duty cycle
    "microwave":       (1/2,        1/358,      False),  # ~4 uses/day, ~2 min each
    "washing machine": (1/60,       1/1380,     False),  # ~1 cycle/day, ~60 min
    "kettle":          (1/3,        1/285,      False),  # ~5 uses/day, ~3 min each
}

MINUTES_PER_DAY = 1440

def make_schedule(label, rng):
    p_off, p_on, init = SCHEDULE_PARAMS[label]
    state = init
    schedule = np.zeros(MINUTES_PER_DAY, dtype=bool)
    for t in range(MINUTES_PER_DAY):
        schedule[t] = state
        if state:
            state = rng.random() > p_off
        else:
            state = rng.random() < p_on
    return schedule

def sample_power(label, n_on, rng):
    pool = power_pool[label]
    return rng.choice(pool, size=n_on, replace=True).astype(np.float32)

# ── generate one DataFrame ─────────────────────────────────────────────────
SPLIT_CONFIG = {
    "train": (42, "2013-11-21"),
    "val":   (43, "2013-12-31"),
    "test":  (44, "2012-08-23"),
}

for split, (seed, date_str) in SPLIT_CONFIG.items():
    rng = np.random.default_rng(seed)
    index = pd.date_range(date_str, periods=MINUTES_PER_DAY, freq="min", name="time")
    df = pd.DataFrame(index=index)

    for label in APPLIANCE_PATTERNS:
        sched = make_schedule(label, rng)
        col = np.zeros(MINUTES_PER_DAY, dtype=np.float32)
        n_on = sched.sum()
        if n_on:
            col[sched] = sample_power(label, int(n_on), rng)
        df[label] = col

    df["main"] = df[list(APPLIANCE_PATTERNS.keys())].sum(axis=1).astype(np.float32)
    df = df[["main"] + list(APPLIANCE_PATTERNS.keys())]  # main first

    out_path = os.path.join(OUT_DIR, f"{split}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((df,), f)

    on_mins = {label: int((df[label] > 0).sum()) for label in APPLIANCE_PATTERNS}
    print(f"\n{split}.pkl  ({date_str})  ->  {out_path}")
    print(f"  shape: {df.shape}")
    for label, mins in on_mins.items():
        print(f"  {label:20s}: {mins:4d} min on, "
              f"mean active power = {df.loc[df[label]>0, label].mean():.0f} W" if mins else
              f"  {label:20s}:    0 min on")

print("\nDone.")
