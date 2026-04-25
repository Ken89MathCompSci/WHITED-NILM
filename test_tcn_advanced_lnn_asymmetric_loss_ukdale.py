import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
import pickle
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))

from models import TCNAdvancedLiquidNetworkModel
from utils import calculate_nilm_metrics, save_model


class UKDALEDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Asymmetric Weighted BCE Loss
# ---------------------------------------------------------------------------

class AsymmetricLoss(nn.Module):
    """
    L_total = L_MSE + bce_lambda * L_BCE_asymmetric

    L_BCE_asymmetric = -1/N * sum[
        alpha * y_i     * log(p_i)       +   <- positive (ON)  term
        beta  * (1-y_i) * log(1 - p_i)       <- negative (OFF) term
    ]

    where p_i = sigmoid(output_i)

    Setting beta > alpha penalises false positives more heavily —
    the model is discouraged from predicting ON when the appliance is OFF.
    Useful for dish washer which has long OFF periods and rare ON events,
    where the model tends to over-predict ON.

    Setting alpha > beta penalises false negatives more heavily —
    useful for microwave / washer dryer where missing ON events is costly.

    Args:
        alpha:      weight on the ON  (positive) term. Default 0.5.
        beta:       weight on the OFF (negative) term. Default 2.0.
        bce_lambda: scale of asymmetric BCE relative to MSE. Default 0.1.
    """
    def __init__(self, alpha=0.5, beta=2.0, bce_lambda=0.1):
        super(AsymmetricLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_lambda = bce_lambda
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets, threshold_scaled):
        loss_mse = self.mse(outputs, targets)

        # Binary labels
        y = (targets >= threshold_scaled).float()

        # Sigmoid maps scaled power to (0,1)
        p = torch.clamp(torch.sigmoid(outputs), min=1e-7, max=1.0 - 1e-7)

        loss_bce = -(
            self.alpha * y       * torch.log(p) +
            self.beta  * (1 - y) * torch.log(1 - p)
        ).mean()

        return loss_mse + self.bce_lambda * loss_bce


# ---------------------------------------------------------------------------
# Per-appliance asymmetric loss parameters
# ---------------------------------------------------------------------------
# beta > alpha  -> penalise false positives  (dish washer: rarely ON)
# alpha > beta  -> penalise false negatives  (microwave, washer dryer)
# alpha == beta -> symmetric BCE             (fridge: balanced)

APPLIANCE_LOSS_PARAMS = {
    'dish washer':  {'alpha': 0.25, 'beta': 4.0},   # strongly discourage false ON
    'fridge':       {'alpha': 0.75, 'beta': 0.75},  # balanced
    'microwave':    {'alpha': 2.0,  'beta': 0.5},   # discourage missing ON spikes
    'washer dryer': {'alpha': 1.5,  'beta': 0.5},   # discourage missing long cycles
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_ukdale_specific_splits():
    print("Loading UKDALE data with specific splits...")

    with open('data/ukdale/train_small.pkl', 'rb') as f:
        train_data = pickle.load(f)[0]
    with open('data/ukdale/val_small.pkl', 'rb') as f:
        val_data = pickle.load(f)[0]
    with open('data/ukdale/test_small.pkl', 'rb') as f:
        test_data = pickle.load(f)[0]

    print(f"Train date range: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Val   date range: {val_data.index.min()} to {val_data.index.max()}")
    print(f"Test  date range: {test_data.index.min()} to {test_data.index.max()}")
    print(f"Available columns: {list(train_data.columns)}")

    return {'train': train_data, 'val': val_data, 'test': test_data}


def create_sequences(data, window_size=100):
    mains = data['main'].values
    X, stride = [], 5
    for i in range(0, len(mains) - window_size + 1, stride):
        X.append(mains[i:i + window_size])
    return np.array(X).reshape(-1, window_size, 1)


def get_threshold_for_appliance(appliance_name):
    return 0.5 if appliance_name == 'washer dryer' else 10.0


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_on_appliance(data_dict, appliance_name, window_size=100,
                       hidden_size=64, num_layers=2, dt=0.1,
                       num_channels=None, kernel_size=3, dropout=0.2,
                       bce_lambda=0.1,
                       epochs=80, lr=0.001, patience=20,
                       save_dir='models/tcn_advanced_lnn_asymmetric_loss_ukdale'):
    if num_channels is None:
        num_channels = [32, 64, 128]

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']

    print(f"Creating sequences for {appliance_name}...")
    X_train = create_sequences(train_data, window_size)
    X_val   = create_sequences(val_data,   window_size)
    X_test  = create_sequences(test_data,  window_size)

    y_train = train_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_train)]
    y_val   = val_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_val)]
    y_test  = test_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_test)]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val   = x_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test  = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    y_train = y_scaler.fit_transform(y_train)
    y_val   = y_scaler.transform(y_val)
    y_test  = y_scaler.transform(y_test)

    print(f"Training sequences:   {X_train.shape} -> {y_train.shape}")
    print(f"Validation sequences: {X_val.shape} -> {y_val.shape}")
    print(f"Test sequences:       {X_test.shape} -> {y_test.shape}")

    raw_threshold    = get_threshold_for_appliance(appliance_name)
    threshold_scaled = float(y_scaler.transform([[raw_threshold]])[0][0])

    loss_p = APPLIANCE_LOSS_PARAMS[appliance_name]
    print(f"Asymmetric BCE -> alpha={loss_p['alpha']}  beta={loss_p['beta']}  "
          f"bce_lambda={bce_lambda}  "
          f"({'FP-penalised' if loss_p['beta'] > loss_p['alpha'] else 'FN-penalised'})")

    num_on  = (y_train >= threshold_scaled).sum()
    num_off = (y_train <  threshold_scaled).sum()
    print(f"Threshold (raw): {raw_threshold}W  |  Threshold (scaled): {threshold_scaled:.4f}"
          f"  |  ON: {num_on}  |  OFF: {num_off}")

    train_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = TCNAdvancedLiquidNetworkModel(
        input_size=1, hidden_size=hidden_size, output_size=1,
        dt=dt, num_channels=num_channels, kernel_size=kernel_size,
        dropout=dropout, num_layers=num_layers
    ).to(device)

    criterion = AsymmetricLoss(alpha=loss_p['alpha'], beta=loss_p['beta'],
                               bce_lambda=bce_lambda)
    mse_only  = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    counter = 0
    best_model_path = os.path.join(
        save_dir,
        f"tcn_advanced_lnn_asymmetric_loss_ukdale_{appliance_name.replace(' ', '_')}_best.pth")

    print(f"Starting TCN-Advanced-LNN (Asymmetric Loss) training for {appliance_name}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, threshold_scaled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        all_targets, all_outputs = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += mse_only(outputs, targets).item()
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        raw_tgts = y_scaler.inverse_transform(
            np.concatenate(all_targets).reshape(-1, 1)).flatten()
        raw_outs = y_scaler.inverse_transform(
            np.concatenate(all_outputs).reshape(-1, 1)).flatten()

        metrics = calculate_nilm_metrics(raw_tgts, raw_outs, threshold=raw_threshold)
        history['val_metrics'].append(metrics)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
              f"Val MSE: {avg_val_loss:.6f}, Val MAE: {metrics['mae']:.2f}, "
              f"Val SAE: {metrics['sae']:.2f}, Val F1: {metrics['f1']:.4f}, "
              f"Val Precision: {metrics['precision']:.4f}, Val Recall: {metrics['recall']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            save_model(model,
                       {'input_size': 1, 'output_size': 1,
                        'hidden_size': hidden_size, 'num_layers': num_layers, 'dt': dt,
                        'num_channels': num_channels, 'kernel_size': kernel_size,
                        'dropout': dropout},
                       {'lr': lr, 'epochs': epochs, 'patience': patience,
                        'window_size': window_size, 'appliance': appliance_name,
                        'alpha': loss_p['alpha'], 'beta': loss_p['beta'],
                        'bce_lambda': bce_lambda},
                       metrics, best_model_path)
            print(f"Model saved to {best_model_path}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break

    print("Training completed!")

    print(f"Loading best val loss model from {best_model_path} for test evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_loss = 0.0
    all_test_targets, all_test_outputs = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += mse_only(outputs, targets).item()
            all_test_targets.append(targets.cpu().numpy())
            all_test_outputs.append(outputs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    all_test_targets = y_scaler.inverse_transform(
        np.concatenate(all_test_targets).reshape(-1, 1)).flatten()
    all_test_outputs = y_scaler.inverse_transform(
        np.concatenate(all_test_outputs).reshape(-1, 1)).flatten()
    test_metrics = calculate_nilm_metrics(all_test_targets, all_test_outputs,
                                          threshold=raw_threshold)

    val_mae_series       = [m['mae']       for m in history['val_metrics']]
    val_sae_series       = [m['sae']       for m in history['val_metrics']]
    val_f1_series        = [m['f1']        for m in history['val_metrics']]
    val_precision_series = [m['precision'] for m in history['val_metrics']]
    val_recall_series    = [m['recall']    for m in history['val_metrics']]

    aggregates = {
        'train_loss_mean':      float(np.mean(history['train_loss'])),
        'train_loss_var':       float(np.var(history['train_loss'])),
        'val_loss_mean':        float(np.mean(history['val_loss'])),
        'val_loss_var':         float(np.var(history['val_loss'])),
        'val_mae_mean':         float(np.mean(val_mae_series)),
        'val_mae_var':          float(np.var(val_mae_series)),
        'val_sae_mean':         float(np.mean(val_sae_series)),
        'val_sae_var':          float(np.var(val_sae_series)),
        'val_f1_mean':          float(np.mean(val_f1_series)),
        'val_f1_var':           float(np.var(val_f1_series)),
        'val_precision_mean':   float(np.mean(val_precision_series)),
        'val_precision_var':    float(np.var(val_precision_series)),
        'val_recall_mean':      float(np.mean(val_recall_series)),
        'val_recall_var':       float(np.var(val_recall_series)),
        'test_mae':             float(test_metrics['mae']),
        'test_sae':             float(test_metrics['sae']),
        'test_f1':              float(test_metrics['f1']),
        'test_precision':       float(test_metrics['precision']),
        'test_recall':          float(test_metrics['recall']),
        'test_loss':            float(avg_test_loss)
    }

    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test Metrics: {test_metrics}")
    print("Aggregates:")
    print(json.dumps(aggregates, indent=2))

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'],   label='Val MSE',    color='red')
    plt.title(f'Loss - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(val_mae_series, label='Val MAE', color='red')
    plt.axhline(test_metrics['mae'], label='Test MAE', color='green', linestyle='--')
    plt.title(f'MAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MAE (W)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(val_sae_series, label='Val SAE', color='red')
    plt.axhline(test_metrics['sae'], label='Test SAE', color='green', linestyle='--')
    plt.title(f'SAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('SAE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(val_f1_series,        label='Val F1',        color='red')
    plt.plot(val_precision_series, label='Val Precision', color='blue')
    plt.plot(val_recall_series,    label='Val Recall',    color='orange')
    plt.axhline(test_metrics['f1'],        color='red',    linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['precision'], color='blue',   linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['recall'],    color='orange', linestyle='--', alpha=0.5)
    plt.title(f'F1 / Precision / Recall - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('Score')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
        f"tcn_advanced_lnn_asymmetric_loss_ukdale_{appliance_name.replace(' ', '_')}_metrics.png"),
        dpi=300, bbox_inches='tight')
    plt.close()

    config = {
        'appliance': appliance_name,
        'dataset': 'UKDALE',
        'loss': 'MSE + Asymmetric BCE',
        'loss_params': {
            'alpha': loss_p['alpha'], 'beta': loss_p['beta'], 'bce_lambda': bce_lambda,
            'mode': 'FP-penalised' if loss_p['beta'] > loss_p['alpha'] else 'FN-penalised'
        },
        'window_size': window_size,
        'model_params': {
            'input_size': 1, 'output_size': 1,
            'hidden_size': hidden_size, 'num_layers': num_layers, 'dt': dt,
            'num_channels': num_channels, 'kernel_size': kernel_size, 'dropout': dropout
        },
        'train_params': {'lr': lr, 'epochs': epochs, 'patience': patience},
        'final_metrics': {
            'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'val_loss':   history['val_loss'][-1]   if history['val_loss']   else None,
            'test_loss':  avg_test_loss,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'aggregates': aggregates
        }
    }
    with open(os.path.join(save_dir,
            f'tcn_advanced_lnn_asymmetric_loss_ukdale_{appliance_name.replace(" ", "_")}_history.json'),
            'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics


def test_on_all_appliances(window_size=100, hidden_size=64, num_layers=2, dt=0.1,
                           num_channels=None, kernel_size=3, dropout=0.2,
                           bce_lambda=0.1, epochs=80, lr=0.001, patience=20):
    if num_channels is None:
        num_channels = [32, 64, 128]

    data_dict = load_ukdale_specific_splits()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"models/tcn_advanced_lnn_asymmetric_loss_ukdale_test_{timestamp}"

    all_results = {}
    appliances = ['dish washer', 'fridge', 'microwave', 'washer dryer']

    for appliance_name in appliances:
        lp = APPLIANCE_LOSS_PARAMS[appliance_name]
        mode = 'FP-penalised' if lp['beta'] > lp['alpha'] else 'FN-penalised'
        print(f"\n{'='*60}")
        print(f"Testing TCN-Advanced-LNN (Asymmetric Loss) on {appliance_name}")
        print(f"  alpha={lp['alpha']}  beta={lp['beta']}  [{mode}]")
        print(f"{'='*60}\n")

        appliance_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))
        os.makedirs(appliance_dir, exist_ok=True)

        try:
            model, history, test_metrics = train_on_appliance(
                data_dict,
                appliance_name=appliance_name,
                window_size=window_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dt=dt,
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                bce_lambda=bce_lambda,
                epochs=epochs,
                lr=lr,
                patience=patience,
                save_dir=appliance_dir
            )
            if model is not None:
                all_results[appliance_name] = {
                    'model_path': os.path.join(
                        appliance_dir,
                        f"tcn_advanced_lnn_asymmetric_loss_ukdale_"
                        f"{appliance_name.replace(' ', '_')}_best.pth"),
                    'final_metrics': {k: float(v) for k, v in test_metrics.items()}
                }
                print(f"Successfully tested on {appliance_name}")
        except Exception as e:
            print(f"Error on {appliance_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    summary = {
        'timestamp': timestamp,
        'dataset': 'UKDALE',
        'loss': 'MSE + Asymmetric BCE',
        'loss_params': {app: APPLIANCE_LOSS_PARAMS[app] for app in appliances},
        'bce_lambda': bce_lambda,
        'dataset_splits': {
            'training':   {'house': 1, 'date': '2014-11-09'},
            'validation': {'house': 1, 'date': '2014-12-07'},
            'testing':    {'house': 5, 'date': '2014-08-24'}
        },
        'window_size': window_size,
        'model_params': {
            'hidden_size': hidden_size, 'num_layers': num_layers, 'dt': dt,
            'num_channels': num_channels, 'kernel_size': kernel_size, 'dropout': dropout
        },
        'train_params': {'epochs': epochs, 'lr': lr, 'patience': patience},
        'results': all_results
    }
    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nTCN-Advanced-LNN (Asymmetric Loss) UKDALE testing completed. "
          f"Results saved to {base_save_dir}")
    return all_results


if __name__ == "__main__":
    print("Testing TCN-Advanced-LNN with Asymmetric Loss on UKDALE dataset...")

    for f in ['data/ukdale/train_small.pkl', 'data/ukdale/val_small.pkl',
              'data/ukdale/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            sys.exit(1)

    results = test_on_all_appliances(
        window_size=100,
        hidden_size=64,
        num_layers=2,
        dt=0.1,
        num_channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.2,
        bce_lambda=0.1,
        epochs=80,
        lr=0.001,
        patience=20
    )

    print(f"\nSummary of TCN-Advanced-LNN (Asymmetric Loss) on UKDALE dataset:")
    print(f"{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} {'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    for appliance, result in results.items():
        m = result['final_metrics']
        print(f"{appliance:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.2f}")
