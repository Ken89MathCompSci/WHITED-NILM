"""
Build data/WHITED_harmonic/{train,val,test}.pkl from WHITED FLAC recordings.

Extends make_whited_pkl.py with richer per-appliance features extracted from
the raw V+I waveforms at 44100 Hz.  WHITED's high-frequency recordings make
these features uniquely computable — they are unavailable in AMPds.

Additive features (superposition holds, summed into 'main_*' aggregate columns):
  P    — active power:             mean(v * i)
  Q1   — fundamental reactive Q:   Im(V1 * conj(I1)) * 2/N²   at 50 Hz
  Q3   — 3rd harmonic reactive Q:  Im(V3 * conj(I3)) * 2/N²   at 150 Hz
  Q5   — 5th harmonic reactive Q:  Im(V5 * conj(I5)) * 2/N²   at 250 Hz
  Q7   — 7th harmonic reactive Q:  Im(V7 * conj(I7)) * 2/N²   at 350 Hz

Non-additive features (per-appliance only; useful as auxiliary targets):
  thd      — current THD: sqrt(sum |I_h|^2, h=2..10) / |I_1|
  vi_area  — V-I Lissajous area over one 50 Hz cycle (normalized, shoelace)
  crest    — current crest factor = peak(|i|) / RMS(i)

Aggregate input columns:  main, main_q1, main_q3, main_q5, main_q7
Per-appliance columns:    {app}, {app}_q1..q7, {app}_thd, {app}_vi_area, {app}_crest

Why Q3/Q5/Q7 help discriminate appliances:
  Kettle      — purely resistive: Q1≈Q3≈Q5≈Q7≈0, THD≈0
  Microwave   — magnetron switching: non-zero odd harmonics, moderate THD
  Fridge      — compressor motor: inductive Q1>0, lower harmonics
  Wash. mach. — mixed heating+motor: capacitive Q1<0, complex harmonic pattern

Output: data/WHITED_harmonic/{train,val,test}.pkl
"""

import os, glob, pickle
import numpy as np
import pandas as pd
import soundfile as sf

WHITED_DIR = os.path.join(os.path.dirname(__file__), "WhiteD", "DATEN")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "data", "WHITED_enriched_Harmonic")
os.makedirs(OUT_DIR, exist_ok=True)

MK_FACTORS = {
    "MK1": {"volt": 1033.64, "curr": 61.4835},
    "MK2": {"volt": 861.15,  "curr": 60.200},
    "MK3": {"volt": 988.926, "curr": 60.9562},
}

GRID_FREQ = 50.0
HARMONICS = (1, 3, 5, 7)


def _mk(fname):
    for mk in ("MK1", "MK2", "MK3"):
        if mk in fname:
            return mk
    return "MK2"


def extract_features(fp):
    """Extract all 8 features from one FLAC file in a single FFT pass."""
    data, sr = sf.read(fp)
    f  = MK_FACTORS[_mk(os.path.basename(fp))]
    v  = data[:, 0] * f["volt"]
    i  = data[:, 1] * f["curr"]
    N  = len(v)
    k0 = round(GRID_FREQ * N / sr)

    P = float(np.mean(v * i))

    V_fft = np.fft.rfft(v)
    I_fft = np.fft.rfft(i)

    Q = {}
    for h in HARMONICS:
        k = k0 * h
        Q[h] = (float(np.imag((2.0 / (N * N)) * V_fft[k] * np.conj(I_fft[k])))
                if k < len(V_fft) else 0.0)

    I1_mag  = np.abs(I_fft[k0])
    harm_sq = sum(np.abs(I_fft[k0 * n]) ** 2
                  for n in range(2, 11) if k0 * n < len(I_fft))
    thd = float(np.sqrt(harm_sq) / (I1_mag + 1e-9))

    # V-I Lissajous area (shoelace) over one cycle at recording midpoint
    period  = int(round(sr / GRID_FREQ))
    mid     = N // 2
    v_c     = v[mid: mid + period]
    i_c     = i[mid: mid + period]
    v_n     = v_c / (np.max(np.abs(v_c)) + 1e-9)
    i_n     = i_c / (np.max(np.abs(i_c)) + 1e-9)
    vi_area = float(0.5 * np.abs(
        np.sum(v_n[:-1] * i_n[1:] - v_n[1:] * i_n[:-1])))

    rms_i = float(np.sqrt(np.mean(i ** 2)))
    crest = float(np.max(np.abs(i)) / (rms_i + 1e-9))

    return {
        'P': P,
        'Q1': Q[1], 'Q3': Q[3], 'Q5': Q[5], 'Q7': Q[7],
        'thd': thd, 'vi_area': vi_area, 'crest': crest,
    }


# ── Load measured feature pools ────────────────────────────────────────────────
APPLIANCE_PATTERNS = {
    "fridge":           "Fridge",
    "microwave":        "Microwave",
    "washing machine":  "WashingMachine",
    "kettle":           "Kettle",
}

FEATURE_KEYS = ['P', 'Q1', 'Q3', 'Q5', 'Q7', 'thd', 'vi_area', 'crest']

print("Extracting harmonic features from FLAC files ...")
feature_pools = {}

for label, pattern in APPLIANCE_PATTERNS.items():
    files = sorted(glob.glob(os.path.join(WHITED_DIR, f"{pattern}_*.flac")))
    feats = [extract_features(fp) for fp in files]
    feature_pools[label] = {
        k: np.array([f[k] for f in feats], dtype=np.float32)
        for k in FEATURE_KEYS
    }
    p  = feature_pools[label]
    pf = np.abs(p['P']) / (np.sqrt(p['P'] ** 2 + p['Q1'] ** 2) + 1e-9)
    print(f"  {label:20s}: n={len(files):3d}  "
          f"P={p['P'].mean():+7.0f} W  "
          f"Q1={p['Q1'].mean():+6.0f} VAR  "
          f"Q3={p['Q3'].mean():+5.0f}  Q5={p['Q5'].mean():+5.0f}  Q7={p['Q7'].mean():+5.0f}  "
          f"THD={p['thd'].mean():.3f}  "
          f"VI={p['vi_area'].mean():.3f}  "
          f"crest={p['crest'].mean():.2f}  "
          f"PF={pf.mean():.3f}")


# ── Markov-chain scheduling ────────────────────────────────────────────────────
SCHEDULE_PARAMS = {
    "fridge":          (1/25,  1/55,  True ),
    "microwave":       (1/5,   1/100, False),
    "washing machine": (1/60,  1/540, False),
    "kettle":          (1/3,   1/57,  False),
}

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
        state = (rng.random() > p_off) if state else (rng.random() < p_on)
    return schedule


def sample_features(label, n_on, rng):
    """Sample all features from the same FLAC index to preserve correlations."""
    pool_size = len(feature_pools[label]['P'])
    idx = rng.integers(0, pool_size, size=n_on)
    return {k: feature_pools[label][k][idx] for k in FEATURE_KEYS}


# ── Generate DataFrames ────────────────────────────────────────────────────────
for split, (seed, date_str, n_days) in SPLIT_CONFIG.items():
    n_minutes = n_days * 1440
    rng       = np.random.default_rng(seed)
    index     = pd.date_range(date_str, periods=n_minutes, freq="min", name="time")
    df        = pd.DataFrame(index=index)

    for label in APPLIANCE_PATTERNS:
        sched = make_schedule(label, n_minutes, rng)
        cols  = {k: np.zeros(n_minutes, dtype=np.float32) for k in FEATURE_KEYS}
        n_on  = int(sched.sum())
        if n_on:
            sampled = sample_features(label, n_on, rng)
            for k in FEATURE_KEYS:
                cols[k][sched] = sampled[k]

        df[label]              = cols['P']
        df[f'{label}_q1']      = cols['Q1']
        df[f'{label}_q3']      = cols['Q3']
        df[f'{label}_q5']      = cols['Q5']
        df[f'{label}_q7']      = cols['Q7']
        df[f'{label}_thd']     = cols['thd']
        df[f'{label}_vi_area'] = cols['vi_area']
        df[f'{label}_crest']   = cols['crest']

    app_cols = list(APPLIANCE_PATTERNS.keys())

    # Aggregate: only additive features (superposition holds for P and Q harmonics)
    df['main']    = df[app_cols].sum(axis=1).astype(np.float32)
    df['main_q1'] = df[[f'{a}_q1' for a in app_cols]].sum(axis=1).astype(np.float32)
    df['main_q3'] = df[[f'{a}_q3' for a in app_cols]].sum(axis=1).astype(np.float32)
    df['main_q5'] = df[[f'{a}_q5' for a in app_cols]].sum(axis=1).astype(np.float32)
    df['main_q7'] = df[[f'{a}_q7' for a in app_cols]].sum(axis=1).astype(np.float32)

    main_cols        = ['main', 'main_q1', 'main_q3', 'main_q5', 'main_q7']
    app_feature_cols = []
    for a in app_cols:
        app_feature_cols += [a, f'{a}_q1', f'{a}_q3', f'{a}_q5', f'{a}_q7',
                              f'{a}_thd', f'{a}_vi_area', f'{a}_crest']
    df = df[main_cols + app_feature_cols]

    out_path = os.path.join(OUT_DIR, f"{split}.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump((df,), fh)

    print(f"\n{split}.pkl  ({date_str})  ->  {out_path}")
    print(f"  shape: {df.shape}")
    for a in app_cols:
        on_mask = df[a] > 0
        n_on    = int(on_mask.sum())
        if n_on:
            print(f"  {a:20s}: {n_on:4d} min  "
                  f"P={df.loc[on_mask, a].mean():.0f} W  "
                  f"Q1={df.loc[on_mask, f'{a}_q1'].mean():.0f} VAR  "
                  f"Q3={df.loc[on_mask, f'{a}_q3'].mean():.0f}  "
                  f"THD={df.loc[on_mask, f'{a}_thd'].mean():.3f}  "
                  f"VI={df.loc[on_mask, f'{a}_vi_area'].mean():.3f}  "
                  f"crest={df.loc[on_mask, f'{a}_crest'].mean():.2f}")
        else:
            print(f"  {a:20s}:    0 min on")

print("\nDone.")
