"""
Build data/WHITED/train.pkl, val.pkl, test.pkl from WHITED FLAC recordings.

Each pkl is a tuple(DataFrame) matching the AMPds format:
  - DatetimeIndex at 1-minute resolution
  - Columns: main, main_q,
             fridge, microwave, washing machine, kettle        (active power W)
             fridge_q, microwave_q, washing machine_q, kettle_q (reactive power VAR)
  - main   = sum of appliance active-power columns
  - main_q = sum of appliance reactive-power columns

Active power:   P = mean(v * i)
Reactive power: Q = Im(V1 * conj(I1)) * 2 / N^2
  V1, I1 are DFT bins at the 50 Hz fundamental (Austrian grid).
  Q > 0  -> inductive load (fridge compressor, washing-machine motor)
  Q ~= 0 -> resistive load (kettle heating element, microwave magnetron)

P and Q are sampled as correlated pairs from the same FLAC measurement so
the per-appliance power factor is preserved in the synthetic schedules.
"""

import os, glob, pickle
import numpy as np
import pandas as pd
import soundfile as sf

WHITED_DIR = os.path.join(os.path.dirname(__file__), "WhiteD", "DATEN")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "data", "WHITED_enriched")
os.makedirs(OUT_DIR, exist_ok=True)

MK_FACTORS = {
    "MK1": {"volt": 1033.64, "curr": 61.4835},
    "MK2": {"volt": 861.15,  "curr": 60.200},
    "MK3": {"volt": 988.926, "curr": 60.9562},
}

GRID_FREQ = 50.0  # Austrian grid frequency (Hz)


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


def reactive_power(fp):
    """
    Fundamental-frequency reactive power via one-sided DFT.

    Q = Im(V1 * conj(I1)) * 2 / N^2

    where V1 = rfft(v)[k0], I1 = rfft(i)[k0], k0 = round(f0 * N / sr).

    Derivation: for a real signal with rfft, the contribution of bin k (k != 0, N/2)
    to the total power is doubled (one-sided spectrum).  Active power from time-domain
    mean(v*i) equals Re(S1) and reactive power Q equals Im(S1) where
    S1 = (2/N^2) * V1 * conj(I1).

    Sign: Q > 0 for inductive loads (current lags voltage).
    """
    data, sr = sf.read(fp)
    f  = MK_FACTORS[_mk(os.path.basename(fp))]
    v  = data[:, 0] * f["volt"]
    i  = data[:, 1] * f["curr"]
    N  = len(v)
    k0 = round(GRID_FREQ * N / sr)
    V1 = np.fft.rfft(v)[k0]
    I1 = np.fft.rfft(i)[k0]
    Q1 = float(np.imag((2.0 / (N * N)) * V1 * np.conj(I1)))
    return Q1


# ── load measured power pools ──────────────────────────────────────────────
APPLIANCE_PATTERNS = {
    "fridge":           "Fridge",
    "microwave":        "Microwave",
    "washing machine":  "WashingMachine",
    "kettle":           "Kettle",
}

print("Computing active and reactive power from FLAC files ...")
power_pool = {}
q_pool     = {}

for label, pattern in APPLIANCE_PATTERNS.items():
    files  = sorted(glob.glob(os.path.join(WHITED_DIR, f"{pattern}_*.flac")))
    p_vals = [active_power(f)   for f in files]
    q_vals = [reactive_power(f) for f in files]
    power_pool[label] = np.array(p_vals, dtype=np.float32)
    q_pool[label]     = np.array(q_vals, dtype=np.float32)
    pf = np.abs(np.array(p_vals)) / (np.sqrt(np.array(p_vals)**2 + np.array(q_vals)**2) + 1e-9)
    print(f"  {label:20s}: n={len(p_vals):3d}  "
          f"P mean={np.mean(p_vals):.0f} W  "
          f"Q mean={np.mean(q_vals):.0f} VAR  "
          f"PF mean={np.mean(pf):.3f}")


# ── Markov-chain scheduling parameters ────────────────────────────────────
# Each appliance: (p_on_to_off, p_off_to_on, initial_state)
SCHEDULE_PARAMS = {
    #                       p(on->off)  p(off->on)  init
    "fridge":          (1/25,      1/55,      True ),  # ~31% duty cycle
    "microwave":       (1/5,       1/100,     False),  # ~5%  duty, ~8 uses/day
    "washing machine": (1/60,      1/540,     False),  # ~10% duty, ~1 cycle/day
    "kettle":          (1/3,       1/57,      False),  # ~5%  duty, ~15 uses/day
}

# Train uses 7 days; val/test use 1 day each
SPLIT_CONFIG = {
    "train": (42, "2013-11-21", 7),
    "val":   (43, "2013-12-31", 1),
    "test":  (44, "2012-08-23", 1),
}


def make_schedule(label, n_minutes, rng):
    p_off, p_on, init = SCHEDULE_PARAMS[label]
    state    = init
    schedule = np.zeros(n_minutes, dtype=bool)
    for t in range(n_minutes):
        schedule[t] = state
        if state:
            state = rng.random() > p_off
        else:
            state = rng.random() < p_on
    return schedule


def sample_pq(label, n_on, rng):
    """Sample correlated (P, Q) pairs from the same FLAC index."""
    pool_size = len(power_pool[label])
    idx = rng.integers(0, pool_size, size=n_on)
    return (power_pool[label][idx].astype(np.float32),
            q_pool[label][idx].astype(np.float32))


# ── generate one DataFrame per split ──────────────────────────────────────
for split, (seed, date_str, n_days) in SPLIT_CONFIG.items():
    n_minutes = n_days * 1440
    rng       = np.random.default_rng(seed)
    index     = pd.date_range(date_str, periods=n_minutes, freq="min", name="time")
    df        = pd.DataFrame(index=index)

    for label in APPLIANCE_PATTERNS:
        sched = make_schedule(label, n_minutes, rng)
        p_col = np.zeros(n_minutes, dtype=np.float32)
        q_col = np.zeros(n_minutes, dtype=np.float32)
        n_on  = sched.sum()
        if n_on:
            p_vals, q_vals = sample_pq(label, int(n_on), rng)
            p_col[sched] = p_vals
            q_col[sched] = q_vals
        df[label]        = p_col
        df[f"{label}_q"] = q_col

    app_cols   = list(APPLIANCE_PATTERNS.keys())
    app_q_cols = [f"{lbl}_q" for lbl in app_cols]

    df["main"]   = df[app_cols].sum(axis=1).astype(np.float32)
    df["main_q"] = df[app_q_cols].sum(axis=1).astype(np.float32)

    # Column order: main, main_q, P columns, Q columns
    df = df[["main", "main_q"] + app_cols + app_q_cols]

    out_path = os.path.join(OUT_DIR, f"{split}.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump((df,), fh)

    on_mins = {lbl: int((df[lbl] > 0).sum()) for lbl in app_cols}
    print(f"\n{split}.pkl  ({date_str})  ->  {out_path}")
    print(f"  shape: {df.shape}")
    for lbl, mins in on_mins.items():
        if mins:
            p_on = df.loc[df[lbl] > 0, lbl].values
            q_on = df.loc[df[lbl] > 0, f"{lbl}_q"].values
            pf   = np.mean(np.abs(p_on) / (np.sqrt(p_on**2 + q_on**2) + 1e-9))
            print(f"  {lbl:20s}: {mins:4d} min on  "
                  f"P={p_on.mean():.0f} W  Q={q_on.mean():.0f} VAR  PF={pf:.3f}")
        else:
            print(f"  {lbl:20s}:    0 min on")

print("\nDone.")
