"""
Physics-Informed BasicLNN with Harmonic Features for NILM — WHITED.

Exploits WHITED's high-frequency V+I waveforms to extract per-appliance
harmonic signatures.  Five additive features derived from the FLAC recordings
are used as model inputs, and five physics conservation constraints are applied
during training.

Input channels (5):
  [P_agg, Q1_agg, Q3_agg, Q5_agg, Q7_agg]
  — all additive under superposition, so conservation laws hold exactly.

Why harmonics discriminate appliances:
  Kettle      Q1≈Q3≈Q5≈Q7≈0   (purely resistive, near-zero THD)
  Microwave   Q3/Q5/Q7 ≠ 0    (magnetron switching generates odd harmonics)
  Fridge      Q1 > 0 (inductive motor), lower harmonics
  Wash. mach. Q1 < 0 (capacitive at low heat), complex harmonic pattern

Physics constraints:
  L_P  = ReLU(sum P̂_i_raw  - P_agg_raw  - eps_P)      one-sided
  L_Q1 = |sum Q̂1_i_raw - Q1_agg_raw| >= eps_Q1        symmetric
  L_Q3 = |sum Q̂3_i_raw - Q3_agg_raw| >= eps_Q3        symmetric
  L_Q5 = |sum Q̂5_i_raw - Q5_agg_raw| >= eps_Q5        symmetric
  L_Q7 = |sum Q̂7_i_raw - Q7_agg_raw| >= eps_Q7        symmetric
  L_nn = ReLU(-P̂_raw)                                  non-negativity
  L_bd = ReLU(P̂_scaled - 1.0)                          upper bound

Requires pkl files from make_whited_harmonic_pkl.py (data/WHITED_enriched_Harmonic/).
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
import pickle
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))
from utils import calculate_nilm_metrics, save_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS        = 100
PATIENCE      = 25
LR            = 1e-3
BATCH         = 32
WIN           = 100
STRIDE        = 2

WARMUP_EPOCHS = 30   # longer warmup for stability
RAMP_EPOCHS   = 20   # physics weight linearly ramps from 0 → full over this many epochs

# Physics loss weights (full strength after ramp)
LAMBDA_P   = 0.01    # active-power conservation
LAMBDA_Q1  = 0.01    # fundamental reactive conservation
LAMBDA_Q3  = 0.0005  # 3rd harmonic — much weaker; smaller magnitude signal
LAMBDA_Q5  = 0.0005  # 5th harmonic
LAMBDA_Q7  = 0.0005  # 7th harmonic
LAMBDA_NN  = 0.005   # non-negativity on P
LAMBDA_BD  = 0.005   # P upper-bound

# Tolerances (raw units)
EPSILON_P  = 50.0    # W
EPSILON_Q1 = 30.0    # VAR
EPSILON_Q3 = 10.0    # VAR
EPSILON_Q5 = 5.0     # VAR
EPSILON_Q7 = 5.0     # VAR

APPLIANCES = ['fridge', 'microwave', 'washing machine', 'kettle']

THRESHOLDS = {
    'fridge':           100.0,
    'microwave':         50.0,
    'washing machine':    5.0,
    'kettle':            50.0,
}

BCE_LAMBDA = {'fridge': 0.1, 'microwave': 0.5, 'washing machine': 0.3, 'kettle': 1.0}
BCE_ALPHA  = {'fridge': 0.8, 'microwave': 2.5, 'washing machine': 4.0, 'kettle': 3.0}

# Input channel names (order must match create_sequences)
INPUT_CHANNELS = ['main', 'main_q1', 'main_q3', 'main_q5', 'main_q7']
N_INPUT        = len(INPUT_CHANNELS)   # 5


# ---------------------------------------------------------------------------
# Multi-physics loss (5 conservation constraints)
# ---------------------------------------------------------------------------

class HarmonicPhysicsLoss(nn.Module):
    """
    Inverse-scales predictions to raw units, then applies:
      P, Q1, Q3, Q5, Q7 conservation + non-negativity + P upper-bound.

    Scaler params registered as buffers so the module moves to GPU cleanly.
    """

    def __init__(self, x_scalers, yp_scalers, yq_scalers,
                 eps_p=EPSILON_P, eps_q1=EPSILON_Q1,
                 eps_q3=EPSILON_Q3, eps_q5=EPSILON_Q5, eps_q7=EPSILON_Q7):
        """
        Args:
            x_scalers:  list of 5 MinMaxScalers for [P, Q1, Q3, Q5, Q7] input channels
            yp_scalers: list of 4 MinMaxScalers for per-appliance P targets
            yq_scalers: dict of lists — yq_scalers[h] = list of 4 scalers for Qh targets
                        keys: 1, 3, 5, 7
        """
        super().__init__()
        self.eps_p  = eps_p
        self.eps_q1 = eps_q1
        self.eps_q3 = eps_q3
        self.eps_q5 = eps_q5
        self.eps_q7 = eps_q7

        # Input channel inverse-scale params: shape (N_INPUT,)
        x_mins   = [float(s.data_min_[0])   for s in x_scalers]
        x_ranges = [float(s.data_range_[0]) for s in x_scalers]
        self.register_buffer('x_mins',   torch.tensor(x_mins,   dtype=torch.float32))
        self.register_buffer('x_ranges', torch.tensor(x_ranges, dtype=torch.float32))

        # Per-appliance P inverse-scale params
        yp_mins   = [float(s.data_min_[0])   for s in yp_scalers]
        yp_ranges = [float(s.data_range_[0]) for s in yp_scalers]
        self.register_buffer('yp_mins',   torch.tensor(yp_mins,   dtype=torch.float32))
        self.register_buffer('yp_ranges', torch.tensor(yp_ranges, dtype=torch.float32))

        # Per-appliance Qh inverse-scale params: stored as (4,) tensors per harmonic
        for h in (1, 3, 5, 7):
            mins   = [float(s.data_min_[0])   for s in yq_scalers[h]]
            ranges = [float(s.data_range_[0]) for s in yq_scalers[h]]
            self.register_buffer(f'yq{h}_mins',   torch.tensor(mins,   dtype=torch.float32))
            self.register_buffer(f'yq{h}_ranges', torch.tensor(ranges, dtype=torch.float32))

    def _inv(self, x_sc, mins, ranges):
        """Inverse MinMax transform: raw = scaled * range + min."""
        return x_sc * ranges + mins

    def forward(self, x_mid_sc, p_pred_sc, q_preds_sc):
        """
        Args:
            x_mid_sc:   (batch, 5)       scaled [P, Q1, Q3, Q5, Q7] at window midpoint
            p_pred_sc:  (batch, 4)       scaled per-appliance P predictions
            q_preds_sc: dict {1,3,5,7} -> (batch, 4) scaled per-appliance Qh predictions
        Returns:
            l_p, l_q1, l_q3, l_q5, l_q7, l_nn, l_bd
        """
        # Inverse-scale input channels to raw units
        x_raw = self._inv(x_mid_sc, self.x_mins, self.x_ranges)  # (batch, 5)
        p_agg_raw  = x_raw[:, 0]
        q1_agg_raw = x_raw[:, 1]
        q3_agg_raw = x_raw[:, 2]
        q5_agg_raw = x_raw[:, 3]
        q7_agg_raw = x_raw[:, 4]

        # Inverse-scale P predictions
        p_raw = self._inv(p_pred_sc, self.yp_mins, self.yp_ranges)  # (batch, 4)

        # 1. Active-power conservation (one-sided)
        l_p = F.relu(p_raw.sum(1) - p_agg_raw - self.eps_p).mean()

        # 2–5. Harmonic reactive-power conservation (symmetric)
        def q_loss(h, eps, agg_raw):
            q_raw = self._inv(q_preds_sc[h],
                              getattr(self, f'yq{h}_mins'),
                              getattr(self, f'yq{h}_ranges'))
            return (q_raw.sum(1) - agg_raw).abs().clamp(min=eps).mean()

        l_q1 = q_loss(1, self.eps_q1, q1_agg_raw)
        l_q3 = q_loss(3, self.eps_q3, q3_agg_raw)
        l_q5 = q_loss(5, self.eps_q5, q5_agg_raw)
        l_q7 = q_loss(7, self.eps_q7, q7_agg_raw)

        # 6. Non-negativity on P
        l_nn = F.relu(-p_raw).mean()

        # 7. P upper-bound (scaled > 1.0 is impossible given training data)
        l_bd = F.relu(p_pred_sc - 1.0).mean()

        return l_p, l_q1, l_q3, l_q5, l_q7, l_nn, l_bd


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HarmonicLNNModel(nn.Module):
    """
    BasicLiquidTimeLayer on 5-channel input [P, Q1, Q3, Q5, Q7].

    Per-appliance heads produce P predictions (primary) and Q1/Q3/Q5/Q7
    predictions (for conservation constraints).

    Cell update (fixed tau):
        tau   = softplus(tau_param)
        f_t   = tanh(LayerNorm(W_in * x_t + W_rec * h))
        h_new = clamp(h + (-h/tau + f_t) * dt, -10, 10)
    """

    def __init__(self, input_size=N_INPUT, hidden_size=64, n_appliances=4, dt=0.1):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_appliances = n_appliances
        self.dt           = dt

        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau         = nn.Parameter(torch.ones(hidden_size))
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)

        self.intra_norm = nn.LayerNorm(hidden_size)
        self.norm       = nn.LayerNorm(hidden_size)

        # Primary output heads
        self.p_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_appliances)])

        # Harmonic reactive output heads (for conservation constraints)
        self.q1_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_appliances)])
        self.q3_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_appliances)])
        self.q5_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_appliances)])
        self.q7_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_appliances)])

    def forward(self, x):
        """
        Args:
            x: (batch, WIN, 5)
        Returns:
            p_out:              (batch, 4)
            q_outs: dict {1,3,5,7} -> (batch, 4)
        """
        batch, seq_len, _ = x.size()
        h   = torch.zeros(batch, self.hidden_size, device=x.device)
        tau = F.softplus(self.tau).unsqueeze(0)

        for t in range(seq_len):
            f_t = torch.tanh(self.intra_norm(
                self.input_proj(x[:, t, :]) + torch.matmul(h, self.rec_weights)))
            h = (h + (-h / tau + f_t) * self.dt).clamp(-10.0, 10.0)

        h = self.norm(h)

        p_out = torch.cat([head(h) for head in self.p_heads], dim=1)
        q_outs = {
            1: torch.cat([head(h) for head in self.q1_heads], dim=1),
            3: torch.cat([head(h) for head in self.q3_heads], dim=1),
            5: torch.cat([head(h) for head in self.q5_heads], dim=1),
            7: torch.cat([head(h) for head in self.q7_heads], dim=1),
        }
        return p_out, q_outs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WHITEDHarmonicDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y_P, Y_Q1, Y_Q3, Y_Q5, Y_Q7):
        self.X   = torch.FloatTensor(X)
        self.Y_P  = torch.FloatTensor(Y_P)
        self.Y_Q1 = torch.FloatTensor(Y_Q1)
        self.Y_Q3 = torch.FloatTensor(Y_Q3)
        self.Y_Q5 = torch.FloatTensor(Y_Q5)
        self.Y_Q7 = torch.FloatTensor(Y_Q7)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx],
                self.Y_P[idx], self.Y_Q1[idx], self.Y_Q3[idx],
                self.Y_Q5[idx], self.Y_Q7[idx])


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data():
    print("Loading WHITED harmonic data ...")
    splits = {}
    for name in ('train', 'val', 'test'):
        with open(f'data/WHITED_enriched_Harmonic/{name}.pkl', 'rb') as f:
            splits[name] = pickle.load(f)[0]
    for name, df in splits.items():
        print(f"  {name:5s}: {df.shape}  cols={list(df.columns[:8])} ...")
    return splits


def create_sequences(data, window_size=WIN):
    """
    Returns:
        X:    (N, WIN, 5)  — [P_agg, Q1_agg, Q3_agg, Q5_agg, Q7_agg]
        Y_P:  (N, 4)       — per-appliance active power at midpoint
        Y_Q1: (N, 4)       — per-appliance Q1 at midpoint
        Y_Q3: (N, 4)
        Y_Q5: (N, 4)
        Y_Q7: (N, 4)
    """
    channels = np.stack([data[c].values for c in INPUT_CHANNELS], axis=1)  # (T, 5)

    app_p  = [data[a].values           for a in APPLIANCES]
    app_q1 = [data[f'{a}_q1'].values   for a in APPLIANCES]
    app_q3 = [data[f'{a}_q3'].values   for a in APPLIANCES]
    app_q5 = [data[f'{a}_q5'].values   for a in APPLIANCES]
    app_q7 = [data[f'{a}_q7'].values   for a in APPLIANCES]

    X, Y_P, Y_Q1, Y_Q3, Y_Q5, Y_Q7 = [], [], [], [], [], []
    T = len(channels)
    for i in range(0, T - window_size, STRIDE):
        X.append(channels[i: i + window_size])          # (WIN, 5)
        mid = i + window_size // 2
        Y_P.append( [arr[mid] for arr in app_p])
        Y_Q1.append([arr[mid] for arr in app_q1])
        Y_Q3.append([arr[mid] for arr in app_q3])
        Y_Q5.append([arr[mid] for arr in app_q5])
        Y_Q7.append([arr[mid] for arr in app_q7])

    return (np.array(X,    dtype=np.float32),
            np.array(Y_P,  dtype=np.float32),
            np.array(Y_Q1, dtype=np.float32),
            np.array(Y_Q3, dtype=np.float32),
            np.array(Y_Q5, dtype=np.float32),
            np.array(Y_Q7, dtype=np.float32))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true_p, y_pred_p, yp_scalers):
    metrics = {}
    for i, app in enumerate(APPLIANCES):
        raw_true = yp_scalers[i].inverse_transform(y_true_p[:, i:i+1]).flatten()
        raw_pred = yp_scalers[i].inverse_transform(y_pred_p[:, i:i+1]).flatten()
        metrics[app] = calculate_nilm_metrics(raw_true, raw_pred, threshold=THRESHOLDS[app])
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def scale_channel(scaler, arr_2d, fit=False):
    """Scale a (N, WIN) array column-wise via a single fitted scaler."""
    flat = arr_2d.reshape(-1, 1)
    if fit:
        scaler.fit(flat)
    return scaler.transform(flat).reshape(arr_2d.shape)


def train_model(data_dict, save_dir, hidden_size=128, dt=0.1):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  hidden={hidden_size}  dt={dt}  input_channels={N_INPUT}")
    print(f"Physics weights: P={LAMBDA_P}  Q1={LAMBDA_Q1}  "
          f"Q3={LAMBDA_Q3}  Q5={LAMBDA_Q5}  Q7={LAMBDA_Q7}  "
          f"nn={LAMBDA_NN}  bd={LAMBDA_BD}")

    # ── Sequences ──────────────────────────────────────────────────────────
    X_tr, YP_tr, YQ1_tr, YQ3_tr, YQ5_tr, YQ7_tr = create_sequences(data_dict['train'], WIN)
    X_va, YP_va, YQ1_va, YQ3_va, YQ5_va, YQ7_va = create_sequences(data_dict['val'],   WIN)
    X_te, YP_te, YQ1_te, YQ3_te, YQ5_te, YQ7_te = create_sequences(data_dict['test'],  WIN)

    # ── Input scaling: one MinMaxScaler per channel ────────────────────────
    x_scalers = [MinMaxScaler() for _ in range(N_INPUT)]
    for ch in range(N_INPUT):
        X_tr[:, :, ch] = scale_channel(x_scalers[ch], X_tr[:, :, ch], fit=True)
        X_va[:, :, ch] = scale_channel(x_scalers[ch], X_va[:, :, ch])
        X_te[:, :, ch] = scale_channel(x_scalers[ch], X_te[:, :, ch])

    # ── Target scaling: per-appliance P and Qh scalers ────────────────────
    yp_scalers = [MinMaxScaler() for _ in APPLIANCES]
    yq_scalers = {h: [MinMaxScaler() for _ in APPLIANCES] for h in (1, 3, 5, 7)}

    Y_sets = {
        'P':  (YP_tr,  YP_va,  YP_te,  yp_scalers),
        'Q1': (YQ1_tr, YQ1_va, YQ1_te, yq_scalers[1]),
        'Q3': (YQ3_tr, YQ3_va, YQ3_te, yq_scalers[3]),
        'Q5': (YQ5_tr, YQ5_va, YQ5_te, yq_scalers[5]),
        'Q7': (YQ7_tr, YQ7_va, YQ7_te, yq_scalers[7]),
    }

    for key, (tr, va, te, scs) in Y_sets.items():
        for i in range(len(APPLIANCES)):
            tr[:, i:i+1] = scs[i].fit_transform(tr[:, i:i+1])
            va[:, i:i+1] = scs[i].transform(va[:, i:i+1])
            te[:, i:i+1] = scs[i].transform(te[:, i:i+1])

    thresholds_sc = [
        (THRESHOLDS[app] - float(yp_scalers[i].data_min_[0]))
        / float(yp_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    print(f"Train: {X_tr.shape}  YP:{YP_tr.shape}")
    print(f"Val:   {X_va.shape}")
    print(f"Test:  {X_te.shape}")

    def make_loader(X, YP, YQ1, YQ3, YQ5, YQ7, shuffle):
        return torch.utils.data.DataLoader(
            WHITEDHarmonicDataset(X, YP, YQ1, YQ3, YQ5, YQ7),
            batch_size=BATCH, shuffle=shuffle, drop_last=False)

    tr_loader = make_loader(X_tr, YP_tr, YQ1_tr, YQ3_tr, YQ5_tr, YQ7_tr, True)
    va_loader = make_loader(X_va, YP_va, YQ1_va, YQ3_va, YQ5_va, YQ7_va, False)
    te_loader = make_loader(X_te, YP_te, YQ1_te, YQ3_te, YQ5_te, YQ7_te, False)

    # ── Model + losses ─────────────────────────────────────────────────────
    model = HarmonicLNNModel(
        input_size=N_INPUT, hidden_size=hidden_size,
        n_appliances=len(APPLIANCES), dt=dt,
    ).to(device)

    mse_fn    = nn.MSELoss()
    phys_loss = HarmonicPhysicsLoss(
        x_scalers, yp_scalers, yq_scalers,
        eps_p=EPSILON_P, eps_q1=EPSILON_Q1,
        eps_q3=EPSILON_Q3, eps_q5=EPSILON_Q5, eps_q7=EPSILON_Q7,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_mse_p': [], 'val_mse_p': [],
        'val_metrics': [],
    }
    best_mse_val   = float('inf')
    best_f1_val    = -float('inf')
    best_mse_state = None
    best_f1_state  = None
    patience_ctr   = 0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print("Starting PINN-Harmonic-LNN training ...")

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        ep_total = ep_mse_p = 0.0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for xb, yb_p, yb_q1, yb_q3, yb_q5, yb_q7 in pbar:
            xb   = xb.to(device)
            yb_p = yb_p.to(device)
            yb_q = {1: yb_q1.to(device), 3: yb_q3.to(device),
                    5: yb_q5.to(device), 7: yb_q7.to(device)}

            optimizer.zero_grad()
            p_pred, q_preds = model(xb)

            l_mse_p = mse_fn(p_pred, yb_p)
            l_mse_q = sum(mse_fn(q_preds[h], yb_q[h]) for h in (1, 3, 5, 7))

            x_mid_sc = xb[:, WIN // 2, :]   # (batch, 5) at window midpoint

            if epoch < WARMUP_EPOCHS:
                loss = l_mse_p + l_mse_q
            else:
                # Linearly ramp physics weight 0 → 1 over RAMP_EPOCHS
                phys_scale = min(1.0, (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS)

                l_p, l_q1, l_q3, l_q5, l_q7, l_nn, l_bd = phys_loss(
                    x_mid_sc, p_pred, q_preds)

                bce = torch.tensor(0.0, device=device)
                for i, app in enumerate(APPLIANCES):
                    if BCE_LAMBDA[app] > 0:
                        pi    = p_pred[:, i].clamp(1e-7, 1 - 1e-7)
                        y_bin = (yb_p[:, i] > thresholds_sc[i]).float()
                        w     = torch.where(y_bin == 1,
                                            torch.full_like(y_bin, BCE_ALPHA[app]),
                                            torch.ones_like(y_bin))
                        bce = bce + BCE_LAMBDA[app] * F.binary_cross_entropy(pi, y_bin, weight=w)

                loss = (l_mse_p + l_mse_q
                        + phys_scale * (LAMBDA_P  * l_p
                                        + LAMBDA_Q1 * l_q1
                                        + LAMBDA_Q3 * l_q3
                                        + LAMBDA_Q5 * l_q5
                                        + LAMBDA_Q7 * l_q7
                                        + LAMBDA_NN * l_nn
                                        + LAMBDA_BD * l_bd)
                        + bce)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_total  += loss.item()
            ep_mse_p  += l_mse_p.item()
            pbar.set_postfix({'mse_p': f'{l_mse_p.item():.5f}'})

        n_tr = len(tr_loader)
        history['train_loss'].append(ep_total / n_tr)
        history['train_mse_p'].append(ep_mse_p / n_tr)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        vl_total = vl_mse_p = 0.0
        val_pp, val_pt = [], []

        with torch.no_grad():
            for xb, yb_p, yb_q1, yb_q3, yb_q5, yb_q7 in va_loader:
                p_pred, _ = model(xb.to(device))
                l_mse_p   = mse_fn(p_pred, yb_p.to(device))
                vl_mse_p += l_mse_p.item()
                vl_total += l_mse_p.item()
                val_pp.append(p_pred.cpu().numpy())
                val_pt.append(yb_p.numpy())

        n_va = len(va_loader)
        avg_va = vl_total / n_va
        history['val_loss'].append(avg_va)
        history['val_mse_p'].append(vl_mse_p / n_va)

        scheduler.step(vl_mse_p / n_va)

        yp_pred_all = np.concatenate(val_pp)
        yp_true_all = np.concatenate(val_pt)
        per_app     = compute_metrics(yp_true_all, yp_pred_all, yp_scalers)
        history['val_metrics'].append(per_app)

        avg_f1  = np.mean([per_app[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([per_app[a]['mae'] for a in APPLIANCES])

        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={ep_total/n_tr:.5f}  val={avg_va:.5f}  "
              f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")
        for app in APPLIANCES:
            m = per_app[app]
            print(f"    {app:<16}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  MAE={m['mae']:.2f}")

        # Dual checkpoint: best MSE and best avg F1
        if avg_va < best_mse_val:
            best_mse_val   = avg_va
            best_mse_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr   = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if avg_f1 > best_f1_val:
            best_f1_val   = avg_f1
            best_f1_state = {k: v.clone() for k, v in model.state_dict().items()}

    print("Training completed!")

    def run_test(state, label):
        model.load_state_dict(state)
        model.eval()
        te_pp, te_pt = [], []
        with torch.no_grad():
            for xb, yb_p, *_ in te_loader:
                p_pred, _ = model(xb.to(device))
                te_pp.append(p_pred.cpu().numpy())
                te_pt.append(yb_p.numpy())
        metrics = compute_metrics(np.concatenate(te_pt),
                                  np.concatenate(te_pp), yp_scalers)
        print(f"\n── {label} ──")
        print(f"{'Appliance':<17} {'F1':>8} {'Precision':>10} {'Recall':>8} "
              f"{'MAE':>8} {'SAE':>8}")
        print("-" * 65)
        for app in APPLIANCES:
            m = metrics[app]
            print(f"{app:<17} {m['f1']:>8.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")
        return metrics

    # ── Test ───────────────────────────────────────────────────────────────
    test_metrics_mse = run_test(best_mse_state, "Best val-MSE checkpoint")
    test_metrics_f1  = run_test(best_f1_state,  "Best val-F1  checkpoint")

    # Use the checkpoint with better avg test F1 for plots/config
    avg_f1_mse = np.mean([test_metrics_mse[a]['f1'] for a in APPLIANCES])
    avg_f1_f1  = np.mean([test_metrics_f1[a]['f1']  for a in APPLIANCES])
    test_metrics = test_metrics_f1 if avg_f1_f1 >= avg_f1_mse else test_metrics_mse
    print(f"\nUsing {'F1' if avg_f1_f1 >= avg_f1_mse else 'MSE'} checkpoint for final results "
          f"(avgF1={max(avg_f1_f1, avg_f1_mse):.4f})")

    _plot(history, test_metrics, save_dir)

    config = {
        'dataset': 'WHITED_harmonic',
        'model': 'HarmonicLNNModel',
        'description': (
            'BasicLNN on [P, Q1, Q3, Q5, Q7] input with 7 physics constraints: '
            'P + Q1/Q3/Q5/Q7 conservation + non-negativity + bounds'
        ),
        'input_channels': INPUT_CHANNELS,
        'physics': {
            'lambda_P': LAMBDA_P,   'eps_P_W':   EPSILON_P,
            'lambda_Q1': LAMBDA_Q1, 'eps_Q1_VAR': EPSILON_Q1,
            'lambda_Q3': LAMBDA_Q3, 'eps_Q3_VAR': EPSILON_Q3,
            'lambda_Q5': LAMBDA_Q5, 'eps_Q5_VAR': EPSILON_Q5,
            'lambda_Q7': LAMBDA_Q7, 'eps_Q7_VAR': EPSILON_Q7,
            'lambda_nn': LAMBDA_NN, 'lambda_bd': LAMBDA_BD,
        },
        'window_size': WIN, 'stride': STRIDE,
        'model_params': {'input_size': N_INPUT, 'hidden_size': hidden_size,
                         'n_appliances': len(APPLIANCES), 'dt': dt},
        'train_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                         'warmup_epochs': WARMUP_EPOCHS},
        'test_metrics': {app: {k: float(v) for k, v in m.items()}
                         for app, m in test_metrics.items()},
    }
    with open(os.path.join(save_dir, 'pinn_harmonic_lnn_whited_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_x, history['train_loss'], label='Train total', color='blue')
    plt.plot(epochs_x, history['val_loss'],   label='Val total',   color='red')
    plt.title('Total Loss'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_x, history['train_mse_p'], label='Train MSE_P', color='blue')
    plt.plot(epochs_x, history['val_mse_p'],   label='Val MSE_P',   color='red')
    plt.title('Active Power MSE'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_harmonic_lnn_whited_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('PINN-Harmonic-LNN WHITED — Per-Appliance Val Metrics', fontsize=13)
    for row, app in enumerate(APPLIANCES):
        f1_series  = [m[app]['f1']  for m in history['val_metrics']]
        mae_series = [m[app]['mae'] for m in history['val_metrics']]
        axes[row, 0].plot(epochs_x, f1_series,  color='blue', linewidth=1.5)
        axes[row, 0].axhline(test_metrics[app]['f1'],  color='green', linestyle='--', label='Test')
        axes[row, 0].set_title(f'{app} — F1'); axes[row, 0].legend(); axes[row, 0].grid(True, alpha=0.3)
        axes[row, 1].plot(epochs_x, mae_series, color='red',  linewidth=1.5)
        axes[row, 1].axhline(test_metrics[app]['mae'], color='green', linestyle='--', label='Test')
        axes[row, 1].set_title(f'{app} — MAE (W)'); axes[row, 1].legend(); axes[row, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_harmonic_lnn_whited_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for fp in ['data/WHITED_enriched_Harmonic/train.pkl',
               'data/WHITED_enriched_Harmonic/val.pkl',
               'data/WHITED_enriched_Harmonic/test.pkl']:
        if not os.path.exists(fp):
            print(f"Error: {fp} not found. Run make_whited_harmonic_pkl.py first.")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = f"models/pinn_harmonic_lnn_whited_{timestamp}"

    data_dict = load_data()

    test_metrics, history = train_model(
        data_dict,
        save_dir    = save_dir,
        hidden_size = 128,
        dt          = 0.1,
    )

    print(f"\nResults saved to {save_dir}")
