"""
Physics-Informed Basic LNN with Adaptive Epsilon for NILM — WHITED.

Extends test_pinn_basic_lnn_whited_specific_splits.py with a data-driven
physics tolerance epsilon instead of a hand-tuned constant.

Problem with fixed epsilon
--------------------------
In residential datasets the monitored appliances rarely account for 100% of
the aggregate.  Background loads (always-on devices, unlabelled loads) create
a persistent residual:

    residual(t) = max(0,  P_agg(t) - sum_i P_i_true(t))

If epsilon is set too small the model is penalised for physically valid
predictions.  If set too large the physics constraint becomes inactive.

Adaptive epsilon
----------------
Computed once from the training split before training begins:

    mu    = mean( max(0, P_agg - sum_i P_i_true) )   over all training minutes
    sigma = std(  max(0, P_agg - sum_i P_i_true) )

    epsilon = mu + K_SIGMA * sigma

K_SIGMA (default 1.0) controls tightness:
  small K  ->  tighter constraint, risks penalising valid predictions
  large K  ->  looser constraint, approaches removing L_phys entirely

For the WHITED synthetic dataset, main = exact sum of appliances so mu ≈ 0.
epsilon is floored at EPSILON_FLOOR to keep the constraint numerically stable.
For real industrial datasets with large background loads, mu >> 0 and the
adaptive approach becomes critical.
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

EPOCHS        = 80
PATIENCE      = 20
LR            = 1e-3
BATCH         = 32
WIN           = 100
STRIDE        = 5

LAMBDA_PHYS   = 0.01
WARMUP_EPOCHS = 20
RAMP_EPOCHS   = 15     # BCE ramps 0 → 1 over this many epochs after warmup

# Adaptive epsilon
K_SIGMA       = 1.0    # epsilon = mu_residual + K_SIGMA * sigma_residual
# Floor must cover simultaneous appliance prediction noise, not just background
# loads.  For WHITED (exact sum, no background) mu=sigma=0, so floor dominates.
# 50W gives ~12% slack on fridge power (~408W), matching the original fixed value.
EPSILON_FLOOR = 50.0   # minimum epsilon (W)

APPLIANCES = ['fridge', 'microwave', 'washing machine', 'kettle']

THRESHOLDS = {
    'fridge':           50.0,
    'microwave':        50.0,
    'washing machine':   5.0,
    'kettle':           50.0,
}

BCE_LAMBDA = {'fridge': 0.3, 'microwave': 0.3, 'washing machine': 0.3, 'kettle': 0.3}
BCE_ALPHA  = {'fridge': 1.5, 'microwave': 1.5, 'washing machine': 4.0, 'kettle': 1.5}


# ---------------------------------------------------------------------------
# Adaptive epsilon computation
# ---------------------------------------------------------------------------

def compute_epsilon(train_df, k=K_SIGMA, floor=EPSILON_FLOOR):
    """
    Estimate physics tolerance from training data.

    residual(t) = max(0, P_agg(t) - sum_i P_i_true(t))

    This is the per-minute background power not accounted for by the
    monitored appliances.  For WHITED (synthetic, exact sum) this will be ~0.
    For real industrial datasets with unmonitored loads it can be substantial.

    Returns:
        epsilon  — the adaptive tolerance to use in PhysicsConsistencyLoss
        mu       — mean residual (W)
        sigma    — std of residual (W)
    """
    P_agg  = train_df['main'].values.astype(np.float64)
    P_sum  = sum(train_df[app].values.astype(np.float64) for app in APPLIANCES)
    resid  = np.maximum(0.0, P_agg - P_sum)
    mu     = float(resid.mean())
    sigma  = float(resid.std())
    eps    = max(mu + k * sigma, floor)
    if mu < 1.0:
        print(f"  [Note] mu_residual≈0: all monitored appliances account for "
              f"~100% of aggregate. Epsilon floored at {floor:.1f} W to allow "
              f"simultaneous appliance prediction noise.")
    return eps, mu, sigma


# ---------------------------------------------------------------------------
# Physics Consistency Loss
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """
    Soft one-sided penalty: ReLU(sum P̂_i_raw - P_agg_raw - epsilon)

    epsilon is passed in at construction time (adaptive or fixed).
    All arithmetic stays in raw Watts via differentiable inverse MinMaxScaler.
    """

    def __init__(self, x_scaler, y_scalers, appliances, epsilon_w):
        super().__init__()
        self.epsilon = epsilon_w

        self.register_buffer('x_min',   torch.tensor(float(x_scaler.data_min_[0])))
        self.register_buffer('x_range', torch.tensor(float(x_scaler.data_range_[0])))

        y_mins   = [float(y_scalers[i].data_min_[0])   for i in range(len(appliances))]
        y_ranges = [float(y_scalers[i].data_range_[0]) for i in range(len(appliances))]
        self.register_buffer('y_mins',   torch.tensor(y_mins,   dtype=torch.float32))
        self.register_buffer('y_ranges', torch.tensor(y_ranges, dtype=torch.float32))

    def forward(self, x_mid_scaled, pred_scaled):
        x_raw  = x_mid_scaled * self.x_range + self.x_min     # (batch,)
        p_raw  = pred_scaled  * self.y_ranges + self.y_mins   # (batch, n_apps)
        p_sum  = p_raw.sum(dim=1)
        return F.relu(p_sum - x_raw - self.epsilon).mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PhysicsInformedBasicLiquidNetworkModel(nn.Module):
    """Basic LNN (fixed tau, no gate) -> per-appliance linear heads."""

    def __init__(self, input_size, hidden_size, n_appliances, dt=0.1):
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

        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_appliances)
        ])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h   = torch.zeros(batch_size, self.hidden_size, device=x.device)
        tau = F.softplus(self.tau).unsqueeze(0)

        for t in range(seq_len):
            x_t = x[:, t, :]
            f_t = torch.tanh(self.intra_norm(
                self.input_proj(x_t) + torch.matmul(h, self.rec_weights)))
            h   = (h + (-h / tau + f_t) * self.dt).clamp(-10.0, 10.0)

        return torch.cat([head(self.norm(h)) for head in self.heads], dim=1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiApplianceDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data():
    print("Loading WHITED data ...")
    splits = {}
    for name in ('train', 'val', 'test'):
        with open(f'data/WHITED_enriched/{name}.pkl', 'rb') as f:
            splits[name] = pickle.load(f)[0]
    for name, df in splits.items():
        print(f"  {name:5s}: {df.shape}  "
              f"{df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Columns: {list(splits['train'].columns)}")
    return splits


def create_sequences(data, window_size=WIN):
    mains    = data['main'].values
    app_vals = {app: data[app].values for app in APPLIANCES}
    X, Y = [], []
    for i in range(0, len(mains) - window_size, STRIDE):
        X.append(mains[i:i + window_size])
        mid = i + window_size // 2
        Y.append([app_vals[app][mid] for app in APPLIANCES])
    return (
        np.array(X, dtype=np.float32).reshape(-1, window_size, 1),
        np.array(Y, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_per_appliance_metrics(y_true, y_pred, y_scalers):
    metrics = {}
    for i, app in enumerate(APPLIANCES):
        raw_true = y_scalers[i].inverse_transform(y_true[:, i:i+1]).flatten()
        raw_pred = y_scalers[i].inverse_transform(y_pred[:, i:i+1]).flatten()
        metrics[app] = calculate_nilm_metrics(raw_true, raw_pred,
                                              threshold=THRESHOLDS[app])
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pinn_model(data_dict, save_dir,
                     hidden_size=64, dt=0.1,
                     lambda_phys=LAMBDA_PHYS,
                     k_sigma=K_SIGMA):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Compute adaptive epsilon from training data ─────────────────────────
    epsilon_w, mu_resid, sigma_resid = compute_epsilon(
        data_dict['train'], k=k_sigma, floor=EPSILON_FLOOR)

    print(f"Device: {device}  hidden={hidden_size}  dt={dt}")
    print(f"Residual analysis on training split:")
    print(f"  mu_residual    = {mu_resid:.2f} W")
    print(f"  sigma_residual = {sigma_resid:.2f} W")
    print(f"  K_SIGMA        = {k_sigma}")
    print(f"  epsilon (adaptive) = {epsilon_w:.2f} W  "
          f"  [floor={EPSILON_FLOOR} W]")
    print(f"  lambda_phys    = {lambda_phys}")

    # ── Sequences ──────────────────────────────────────────────────────────
    X_tr, Y_tr = create_sequences(data_dict['train'], WIN)
    X_va, Y_va = create_sequences(data_dict['val'],   WIN)
    X_te, Y_te = create_sequences(data_dict['test'],  WIN)

    # ── Scaling ────────────────────────────────────────────────────────────
    x_scaler = MinMaxScaler()
    X_tr = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, i:i+1] = ys.fit_transform(Y_tr[:, i:i+1])
        Y_va[:, i:i+1] = ys.transform(Y_va[:, i:i+1])
        Y_te[:, i:i+1] = ys.transform(Y_te[:, i:i+1])
        y_scalers.append(ys)

    thresholds_scaled = [
        (THRESHOLDS[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    print(f"Train: {X_tr.shape} -> {Y_tr.shape}")
    print(f"Val:   {X_va.shape} -> {Y_va.shape}")
    print(f"Test:  {X_te.shape} -> {Y_te.shape}")

    tr_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_tr, Y_tr), batch_size=BATCH,
        shuffle=True, drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_va, Y_va), batch_size=BATCH,
        shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_te, Y_te), batch_size=BATCH,
        shuffle=False, drop_last=False)

    # ── Model + losses ──────────────────────────────────────────────────────
    model = PhysicsInformedBasicLiquidNetworkModel(
        input_size=1, hidden_size=hidden_size,
        n_appliances=len(APPLIANCES), dt=dt,
    ).to(device)

    mse_criterion  = nn.MSELoss()
    phys_criterion = PhysicsConsistencyLoss(
        x_scaler, y_scalers, APPLIANCES, epsilon_w=epsilon_w
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history = {
        'train_loss': [], 'train_mse': [], 'train_phys': [],
        'val_loss':   [], 'val_mse':   [], 'val_phys':   [],
        'val_metrics': [],
    }
    best_val_loss = float('inf')
    best_state    = None
    counter       = 0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print("Starting PINN-AdaptiveEpsilon-BasicLNN training ...")

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        ep_mse = ep_phys = ep_total = 0.0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            pred      = model(xb)
            mse_loss  = mse_criterion(pred, yb)
            x_mid     = xb[:, WIN // 2, 0]
            phys_loss = phys_criterion(x_mid, pred)

            if epoch < WARMUP_EPOCHS:
                loss = mse_loss
            else:
                # Linearly ramp BCE and physics 0 → 1 over RAMP_EPOCHS
                ramp = min(1.0, (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS)

                bce_loss = torch.tensor(0.0, device=device)
                for i, app in enumerate(APPLIANCES):
                    if BCE_LAMBDA[app] > 0:
                        pred_i = pred[:, i].clamp(1e-7, 1 - 1e-7)
                        y_bin  = (yb[:, i] > thresholds_scaled[i]).float()
                        w      = torch.where(y_bin == 1,
                                             torch.full_like(y_bin, BCE_ALPHA[app]),
                                             torch.ones_like(y_bin))
                        bce_loss = bce_loss + BCE_LAMBDA[app] * F.binary_cross_entropy(
                            pred_i, y_bin, weight=w)
                loss = mse_loss + ramp * (lambda_phys * phys_loss + bce_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_mse   += mse_loss.item()
            ep_phys  += phys_loss.item()
            ep_total += loss.item()
            pbar.set_postfix({
                'mse':  f'{mse_loss.item():.5f}',
                'phys': f'{phys_loss.item():.5f}',
            })

        n_tr = len(tr_loader)
        history['train_mse'].append(ep_mse   / n_tr)
        history['train_phys'].append(ep_phys  / n_tr)
        history['train_loss'].append(ep_total / n_tr)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        vl_mse = vl_phys = vl_total = 0.0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred      = model(xb)
                mse_loss  = mse_criterion(pred, yb)
                x_mid     = xb[:, WIN // 2, 0]
                phys_loss = phys_criterion(x_mid, pred)
                loss      = mse_loss + lambda_phys * phys_loss
                vl_mse   += mse_loss.item()
                vl_phys  += phys_loss.item()
                vl_total += loss.item()
                val_preds.append(pred.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        n_va = len(va_loader)
        history['val_mse'].append(vl_mse   / n_va)
        history['val_phys'].append(vl_phys  / n_va)
        history['val_loss'].append(vl_total / n_va)

        scheduler.step(vl_mse / n_va)

        per_app = compute_per_appliance_metrics(
            np.concatenate(val_trues), np.concatenate(val_preds), y_scalers)
        history['val_metrics'].append(per_app)

        avg_f1  = np.mean([per_app[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([per_app[a]['mae'] for a in APPLIANCES])

        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={ep_total/n_tr:.5f} "
              f"(mse={ep_mse/n_tr:.5f} phys={ep_phys/n_tr:.5f})  "
              f"val={vl_total/n_va:.5f} "
              f"(mse={vl_mse/n_va:.5f} phys={vl_phys/n_va:.5f})  "
              f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")
        for app in APPLIANCES:
            m = per_app[app]
            print(f"    {app:<16}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}")

        if vl_mse / n_va < best_val_loss:
            best_val_loss = vl_mse / n_va
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter       = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print("Training completed!")

    # ── Test ───────────────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    te_preds, te_trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            te_preds.append(model(xb.to(device)).cpu().numpy())
            te_trues.append(yb.numpy())

    test_metrics = compute_per_appliance_metrics(
        np.concatenate(te_trues), np.concatenate(te_preds), y_scalers)

    print(f"\n{'Appliance':<17} {'F1':>8} {'Precision':>10} {'Recall':>8} "
          f"{'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<17} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    _plot(history, test_metrics, save_dir)

    config = {
        'dataset': 'WHITED_enriched',
        'model': 'PhysicsInformedBasicLiquidNetworkModel',
        'description': 'BasicLNN + adaptive epsilon physics constraint',
        'epsilon': {
            'method':        'adaptive: mu + K_SIGMA * sigma',
            'k_sigma':       k_sigma,
            'mu_residual_W': mu_resid,
            'sigma_residual_W': sigma_resid,
            'epsilon_W':     epsilon_w,
            'floor_W':       EPSILON_FLOOR,
        },
        'window_size': WIN, 'stride': STRIDE,
        'model_params': {
            'input_size': 1, 'hidden_size': hidden_size,
            'n_appliances': len(APPLIANCES), 'dt': dt,
        },
        'train_params': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'lambda_phys': lambda_phys,
            'warmup_epochs': WARMUP_EPOCHS, 'ramp_epochs': RAMP_EPOCHS,
        },
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'pinn_adaptive_epsilon_lnn_whited_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_x, history['train_loss'], label='Train total', color='blue')
    plt.plot(epochs_x, history['val_loss'],   label='Val total',   color='red')
    plt.title('Total Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_x, history['train_mse'], label='Train MSE', color='blue')
    plt.plot(epochs_x, history['val_mse'],   label='Val MSE',   color='red')
    plt.title('MSE Loss'); plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_x, history['train_phys'], label='Train Phys', color='blue')
    plt.plot(epochs_x, history['val_phys'],   label='Val Phys',   color='red')
    plt.title('Physics Loss (adaptive ε)'); plt.xlabel('Epoch'); plt.ylabel('L_phys')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_adaptive_epsilon_lnn_whited_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('PINN-AdaptiveEpsilon-LNN WHITED — Per-Appliance Val Metrics',
                 fontsize=13)
    for row, app in enumerate(APPLIANCES):
        f1_series  = [m[app]['f1']  for m in history['val_metrics']]
        mae_series = [m[app]['mae'] for m in history['val_metrics']]
        axes[row, 0].plot(epochs_x, f1_series, color='blue', linewidth=1.5)
        axes[row, 0].axhline(test_metrics[app]['f1'], color='green',
                             linestyle='--', label='Test F1')
        axes[row, 0].set_title(f'{app} — F1')
        axes[row, 0].legend(); axes[row, 0].grid(True, alpha=0.3)
        axes[row, 1].plot(epochs_x, mae_series, color='red', linewidth=1.5)
        axes[row, 1].axhline(test_metrics[app]['mae'], color='green',
                             linestyle='--', label='Test MAE')
        axes[row, 1].set_title(f'{app} — MAE (W)')
        axes[row, 1].legend(); axes[row, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
                             'pinn_adaptive_epsilon_lnn_whited_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for fp in ['data/WHITED_enriched/train.pkl',
               'data/WHITED_enriched/val.pkl',
               'data/WHITED_enriched/test.pkl']:
        if not os.path.exists(fp):
            print(f"Error: {fp} not found!")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = f"models/pinn_adaptive_epsilon_lnn_whited_{timestamp}"

    data_dict = load_data()

    test_metrics, history = train_pinn_model(
        data_dict,
        save_dir    = save_dir,
        hidden_size = 64,
        dt          = 0.1,
        lambda_phys = LAMBDA_PHYS,
        k_sigma     = K_SIGMA,
    )

    print(f"\nResults saved to {save_dir}")
