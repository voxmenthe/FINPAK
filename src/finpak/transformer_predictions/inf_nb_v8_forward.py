# %% [markdown]
# V7 Forward-Looking Inference Notebook (Python)
#
# This notebook mirrors `inf_nb_v7.py` but focuses on forward-looking
# predictions for a user-selected checkpoint, tickers, and date range.

# %%
import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from finpak.data.fetchers.yahoo import download_multiple_tickers
from preprocessing_v7 import create_stock_features
from timeseries_decoder_v6 import TimeSeriesDecoder
from configs import all_configs

# %%
# Device helpers

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
print("Using device:", DEVICE)

# %%
# User settings
# - Set CHECKPOINT_PATH to a specific checkpoint file.
# - Set TICKERS / START_DATE / END_DATE for your data window.
# - Set CONFIG_NAME to match training config if not using checkpoint metadata.

CONFIG_NAME = "vMP007a_stability"
USE_CHECKPOINT_CONFIG = True

START_DATE = "1990-01-01"
END_DATE = "2025-12-26"
TICKERS = [
    "AAPL",
    "AMZN",
    "MSFT",
    "SPY",
    "TSLA",
    "NVDA",
    "QQQ",
]

USE_LOCAL_CSV = False # True
LOCAL_CSV_PATH = "TRAIN_VAL_DATA/val_df_v11.csv"

N_FUTURE_STEPS = 16
USE_MULTI_HORIZON = True
PLOT_WINDOW_SIZE = 64
PREDICTION_OFFSETS = [2, 4, 7, 14, 21]
SHOW_ACTUAL_FOR_OFFSETS = True
PREDICTION_TARGET_PERIODS = [1, 3, 8] # None  # e.g. [1, 3, 8] to force specific horizons; None uses multi-horizon
HORIZON_CALIBRATION = "std"  # "std" or None
CALIBRATION_LOOKBACK = 756  # trading days to estimate horizon scaling (≈3y)

# Optional: override default prediction parameters
PREDICTION_KWARGS = {
    "use_sampling": False,
    "temperature": 0.01,
    "stability_threshold": 0.1,
    "dampening_factor": 0.95,
    "ewma_alpha": 0.7,
    "return_scaling_power": 1.0,
}

# %%
# Config selection

CONFIG = all_configs[CONFIG_NAME]
print("Config name:", CONFIG_NAME)
print("Data params keys:", list(CONFIG["data_params"].keys()))

# %%
# Data loading

def load_price_df(tickers, start_date, end_date, local_csv_path=None):
    if local_csv_path and os.path.exists(local_csv_path):
        df = pd.read_csv(local_csv_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        available = [t for t in tickers if t in df.columns]
        missing = sorted(set(tickers) - set(available))
        if missing:
            print("Missing tickers in local CSV:", missing)
        df = df[available]
    else:
        df = download_multiple_tickers(tickers, start_date, end_date)
        df = df.loc[:, "Adj Close"]

    df = df.ffill().dropna(axis=0, how="any")
    return df


price_df = load_price_df(
    TICKERS,
    START_DATE,
    END_DATE,
    local_csv_path=LOCAL_CSV_PATH if USE_LOCAL_CSV else None,
)

print("Loaded price_df shape:", price_df.shape)
price_df.tail()

# %%
# Checkpoints

ROOT = Path(__file__).resolve().parents[3]  # FINPAK repo root
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_PATH = "checkpoints/vMP007a_stability_e1142_valloss_0.0000357_tc12_vc0.pt"

checkpoint_path = Path(CHECKPOINT_PATH)
if not checkpoint_path.is_absolute():
    candidate = ROOT / checkpoint_path
    if candidate.exists():
        checkpoint_path = candidate
    else:
        candidate = CHECKPOINT_DIR / checkpoint_path
        if candidate.exists():
            checkpoint_path = candidate

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}. "
        f"Set CHECKPOINT_PATH to an absolute path or a path relative to {ROOT}."
    )

print("Checkpoint path:", checkpoint_path)

# %%
# Load checkpoint and build model

def build_model_from_config(config):
    data_params = config["data_params"]
    model_params = config["model_params"]

    return_periods = data_params["return_periods"]
    sma_periods = data_params["sma_periods"]
    target_periods = data_params["target_periods"]
    use_volatility = data_params.get("use_volatility", False)
    use_momentum = data_params.get("use_momentum", False)
    momentum_periods = data_params.get("momentum_periods", [])

    n_continuous = len(return_periods) + len(sma_periods)
    if use_volatility:
        n_continuous += len(return_periods)
    if use_momentum:
        n_continuous += len(momentum_periods)

    price_change_bins = data_params.get("price_change_bins")
    target_bins = data_params.get("target_bins")

    n_categorical = 1 if price_change_bins else 0
    if price_change_bins:
        n_bins = price_change_bins.get("n_bins", 10)
    elif target_bins:
        n_bins = target_bins.get("n_bins", 10)
    else:
        n_bins = 10

    n_continuous_outputs = len(target_periods)
    n_categorical_outputs = len(target_periods) if target_bins else None

    model = TimeSeriesDecoder(
        d_continuous=n_continuous,
        n_categorical=n_categorical,
        n_bins=n_bins,
        d_model=model_params["d_model"],
        n_heads=model_params["n_heads"],
        n_layers=model_params["n_layers"],
        d_ff=model_params["d_ff"],
        dropout=0.0,
        n_continuous_outputs=n_continuous_outputs,
        n_categorical_outputs=n_categorical_outputs,
        use_multi_scale=model_params.get("use_multi_scale", False),
        use_relative_pos=model_params.get("use_relative_pos", False),
        use_hope_pos=model_params.get("use_hope_pos", False),
        temporal_scales=model_params.get("temporal_scales", [1, 2, 4]),
        base=model_params.get("base", 10000),
    ).to(DEVICE)

    return model


checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
print("Checkpoint keys:", list(checkpoint.keys()))

if USE_CHECKPOINT_CONFIG and "metadata" in checkpoint and hasattr(checkpoint["metadata"], "config"):
    CONFIG = checkpoint["metadata"].config
    print("Using config from checkpoint metadata")

model = build_model_from_config(CONFIG)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

if "metadata" in checkpoint:
    metadata = checkpoint["metadata"]
    print("Checkpoint epoch:", metadata.epoch)
    print("Checkpoint val_loss:", metadata.val_loss)

# %%
# Feature utilities

def normalize_continuous_features(features, eps=1e-6):
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True, unbiased=False)
    std = torch.where(std < eps, torch.ones_like(std), std)
    return (features - mean) / std


def build_features(prices, config):
    features = create_stock_features(prices, config["data_params"])
    continuous = normalize_continuous_features(features.continuous_features)
    categorical = features.categorical_features
    return continuous, categorical, features.valid_start_idx

# %%
# Horizon calibration

def compute_horizon_scalers(prices, target_periods, lookback=756, eps=1e-8):
    prices_np = prices.detach().cpu().numpy() if torch.is_tensor(prices) else np.asarray(prices)
    if len(prices_np) < 3:
        return {period: 1.0 for period in target_periods}

    def _returns(series, period):
        if len(series) <= period:
            return np.array([])
        num = series[period:] - series[:-period]
        den = series[:-period] + eps
        return num / den

    one_day = _returns(prices_np, 1)
    if lookback is not None and len(one_day) > lookback:
        one_day = one_day[-lookback:]
    std_1d = np.std(one_day) if len(one_day) else 0.0
    if std_1d < eps:
        return {period: 1.0 for period in target_periods}

    scalers = {}
    for period in target_periods:
        period_returns = _returns(prices_np, period)
        if lookback is not None and len(period_returns) > lookback:
            period_returns = period_returns[-lookback:]
        std_p = np.std(period_returns) if len(period_returns) else 0.0
        scalers[period] = (std_1d / std_p) if std_p > eps else 1.0
    return scalers

# %%
# Autoregressive prediction

def make_autoregressive_prediction(
    model,
    prices,
    config,
    start_index,
    n_steps=30,
    use_multi_horizon=True,
    horizon_weights=None,
    use_sampling=False,
    temperature=0.01,
    stability_threshold=0.1,
    dampening_factor=0.95,
    use_ewma_smoothing=True,
    ewma_alpha=0.7,
    return_scaling_power=1.0,
    prediction_target_period=None,
    horizon_scalers=None,
    device=DEVICE,
):
    data_params = config["data_params"]
    sequence_length = data_params["sequence_length"]
    target_periods = data_params["target_periods"]

    if use_multi_horizon and horizon_weights is None:
        horizon_weights = [1.0 / len(target_periods)] * len(target_periods)

    current_prices = prices[: start_index + 1].clone()
    if len(current_prices) < sequence_length:
        raise ValueError("Not enough history for the requested sequence_length")

    predictions = []
    predicted_prices = []

    with torch.no_grad():
        for _ in range(n_steps):
            continuous, categorical, valid_start_idx = build_features(current_prices, config)

            min_valid_end = valid_start_idx + sequence_length - 1
            if len(current_prices) - 1 < min_valid_end:
                raise ValueError(
                    f"Not enough valid history after warmup. "
                    f"Need end index >= {min_valid_end}, got {len(current_prices) - 1}"
                )

            seq_end = len(current_prices) - 1
            seq_start = seq_end - sequence_length + 1

            continuous_seq = continuous[seq_start : seq_end + 1].unsqueeze(0).to(device)
            if categorical is not None:
                categorical_seq = categorical[seq_start : seq_end + 1].unsqueeze(0).to(device)
            else:
                categorical_seq = None

            output = model(continuous_seq, categorical_seq)
            if isinstance(output, tuple):
                pred_continuous = output[0]
            else:
                pred_continuous = output

            if use_sampling:
                dist = torch.distributions.Normal(pred_continuous, temperature)
                pred_continuous = dist.sample()

            predictions.append(pred_continuous[0].detach().cpu())

            last_price = current_prices[-1].item()

            if prediction_target_period is not None:
                if prediction_target_period in target_periods:
                    target_idx = target_periods.index(prediction_target_period)
                else:
                    target_idx = int(np.argmin([abs(p - prediction_target_period) for p in target_periods]))
                    prediction_target_period = target_periods[target_idx]

                period = target_periods[target_idx]
                period_return = pred_continuous[0][target_idx].item()
                abs_return = abs(period_return)
                sign = 1 if period_return >= 0 else -1
                daily_return = sign * ((1 + abs_return) ** (return_scaling_power / period) - 1)
                if horizon_scalers is not None:
                    daily_return *= horizon_scalers.get(period, 1.0)

                next_return = daily_return
            elif use_multi_horizon:
                horizon_returns = []
                for i, period in enumerate(target_periods):
                    period_return = pred_continuous[0][i].item()
                    abs_return = abs(period_return)
                    sign = 1 if period_return >= 0 else -1
                    daily_return = sign * ((1 + abs_return) ** (return_scaling_power / period) - 1)
                    if horizon_scalers is not None:
                        daily_return *= horizon_scalers.get(period, 1.0)
                    horizon_dampening = dampening_factor ** (period - 1)
                    horizon_returns.append(daily_return * horizon_dampening)

                next_return = sum(w * r for w, r in zip(horizon_weights, horizon_returns))

                if abs(next_return) > stability_threshold:
                    next_return = stability_threshold * (1 if next_return >= 0 else -1)

                if use_ewma_smoothing and len(current_prices) > 1:
                    prev_return = current_prices[-1] / current_prices[-2] - 1
                    next_return = ewma_alpha * next_return + (1 - ewma_alpha) * prev_return
            else:
                next_return = pred_continuous[0][0].item()

            next_price = float(last_price) * (1 + float(next_return))
            current_prices = torch.cat(
                [current_prices, torch.tensor([next_price], dtype=torch.float32)]
            )
            predicted_prices.append(next_price)

    return torch.stack(predictions), torch.tensor(predicted_prices)

# %%
# Visualization

def make_future_dates(historical_dates, n_steps):
    inferred = pd.infer_freq(historical_dates)
    freq = inferred if inferred is not None else "B"
    start = historical_dates[-1]
    return pd.date_range(start=start, periods=n_steps + 1, freq=freq)[1:]


def plot_forward_prediction(
    historical_prices,
    historical_dates,
    start_index,
    prediction_sets,
    offsets_used,
    prediction_labels=None,
    window_size=180,
    show_actual_for_offsets=True,
    title="Forward-Looking Prediction",
):
    plt.figure(figsize=(14, 7))

    hist_start = max(0, start_index - window_size)
    hist_end = start_index + 1

    x_hist = historical_dates[hist_start:hist_end]
    y_hist = historical_prices[hist_start:hist_end].cpu().numpy()
    if len(y_hist) > 0:
        hist_std = float(np.std(y_hist))
        y_min = float(np.min(y_hist) - hist_std)
        y_max = float(np.max(y_hist) + hist_std)
    else:
        hist_std = 0.0
        y_min = None
        y_max = None

    if not prediction_sets:
        print("No predictions to plot.")
        return

    pred_series = []
    for idx, (pred_prices, offset) in enumerate(zip(prediction_sets, offsets_used)):
        if len(pred_prices) == 0:
            continue
        shifted_start = start_index - offset
        if shifted_start < 0:
            continue
        x_pred = make_future_dates(historical_dates[: shifted_start + 1], len(pred_prices))
        label = prediction_labels[idx] if prediction_labels else f"Predicted t-{offset}"
        pred_series.append((x_pred, pred_prices, offset, label))

    if not pred_series:
        print("No predictions to plot.")
        return

    plt.plot(x_hist, y_hist, label="Historical", color="steelblue")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(pred_series), 1)))
    for (x_pred, pred_prices, offset, label), color in zip(pred_series, colors):
        plt.plot(
            x_pred,
            pred_prices,
            label=label,
            color=color,
            alpha=0.9,
        )

    if show_actual_for_offsets:
        valid_offsets = [offset for offset in offsets_used if offset > 0]
        if valid_offsets:
            longest_offset = max(valid_offsets)
            actual_start = max(0, start_index - longest_offset)
            x_actual = historical_dates[actual_start : start_index + 1]
            y_actual = historical_prices[actual_start : start_index + 1].cpu().numpy()
            plt.plot(
                x_actual,
                y_actual,
                label=f"Actual t-{longest_offset}→t-0",
                color="black",
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
                zorder=5,
            )

    if len(pred_series) > 1:
        common_dates = None
        pred_aligned = []
        for x_pred, pred_prices, _, _ in pred_series:
            series = pd.Series(pred_prices, index=pd.DatetimeIndex(x_pred))
            pred_aligned.append(series)
            common_dates = series.index if common_dates is None else common_dates.intersection(series.index)
        if common_dates is not None and len(common_dates) > 0:
            stacked = np.vstack([series.reindex(common_dates).to_numpy() for series in pred_aligned])
            pred_min = np.nanmin(stacked, axis=0)
            pred_max = np.nanmax(stacked, axis=0)
            plt.fill_between(
                common_dates,
                pred_min,
                pred_max,
                color="gray",
                alpha=0.07,
                linewidth=0,
            )

    plt.axvline(historical_dates[start_index], color="gray", linestyle=":", alpha=0.6)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%y%m%d"))
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()

# %%
# Forward-looking predictions per ticker

for ticker in price_df.columns:
    series = price_df[ticker].dropna()
    if series.empty:
        print(f"Skipping {ticker}: no data")
        continue

    prices = torch.tensor(series.to_numpy(), dtype=torch.float32)
    dates = series.index

    start_index = len(prices) - 1
    if start_index < 0:
        print(f"Skipping {ticker}: insufficient history")
        continue

    prediction_sets = []
    offsets_used = []
    prediction_labels = []
    horizon_scalers = None
    if HORIZON_CALIBRATION == "std":
        horizon_scalers = compute_horizon_scalers(
            prices=prices,
            target_periods=CONFIG["data_params"]["target_periods"],
            lookback=CALIBRATION_LOOKBACK,
        )
    if PREDICTION_TARGET_PERIODS is None:
        target_periods_to_use = [None]
    elif isinstance(PREDICTION_TARGET_PERIODS, (list, tuple)):
        target_periods_to_use = list(PREDICTION_TARGET_PERIODS)
    else:
        target_periods_to_use = [PREDICTION_TARGET_PERIODS]
    for offset in PREDICTION_OFFSETS:
        if N_FUTURE_STEPS <= offset:
            print(f"Skipping {ticker} offset t-{offset}: N_FUTURE_STEPS too small")
            continue
        shifted_start = start_index - offset
        if shifted_start < 0:
            continue
        for target_period in target_periods_to_use:
            try:
                _, pred_prices = make_autoregressive_prediction(
                    model=model,
                    prices=prices,
                    config=CONFIG,
                    start_index=shifted_start,
                    n_steps=N_FUTURE_STEPS,
                    use_multi_horizon=USE_MULTI_HORIZON,
                    prediction_target_period=target_period,
                    horizon_scalers=horizon_scalers,
                    **PREDICTION_KWARGS,
                )
            except ValueError as exc:
                print(f"Skipping {ticker} offset t-{offset}: {exc}")
                continue
            prediction_sets.append(pred_prices.cpu().numpy())
            offsets_used.append(offset)
            if target_period is None:
                label = f"Predicted t-{offset} (multi)"
            else:
                label = f"Predicted t-{offset} ({target_period}d)"
            prediction_labels.append(label)

    if not prediction_sets:
        print(f"Skipping {ticker}: no valid predictions")
        continue

    plot_forward_prediction(
        historical_prices=prices,
        historical_dates=dates,
        start_index=start_index,
        prediction_sets=prediction_sets,
        offsets_used=offsets_used,
        prediction_labels=prediction_labels,
        window_size=PLOT_WINDOW_SIZE,
        show_actual_for_offsets=SHOW_ACTUAL_FOR_OFFSETS,
        title=f"V7 Forward Prediction: {ticker}",
    )

# %%
# Optional: inspect last run outputs
prediction_sets[:1] if 'prediction_sets' in locals() else None

# %%
