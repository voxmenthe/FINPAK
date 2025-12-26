# %% [markdown]
# V7 Inference Notebook (Python)
#
# This notebook mirrors the v7 training pipeline from `main_v7.py` and provides
# a compact flow for model loading, inference, and visualization.

# %%
import os
import json
import glob
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from finpak.data.fetchers.yahoo import download_multiple_tickers
from preprocessing_v7 import combine_price_series, create_stock_features
from timeseries_decoder_v6 import TimeSeriesDecoder
from configs import all_configs
from ticker_configs import val_tickers_v11

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
# Config selection
# - Set CONFIG_NAME to the config used for training.
# - If USE_CHECKPOINT_CONFIG is True and the checkpoint contains metadata,
#   the config will be overridden to match the checkpoint.

CONFIG_NAME = "test"
USE_CHECKPOINT_CONFIG = True

CONFIG = all_configs[CONFIG_NAME]
print("Config name:", CONFIG_NAME)
print("Data params keys:", list(CONFIG["data_params"].keys()))

# %%
# Optional: inspect available configs
# sorted(all_configs.keys())[:50]

# %%
# Data loading

START_DATE = "1990-01-01"
END_DATE = "2025-12-24"
TICKERS = val_tickers_v11

USE_LOCAL_CSV = True
LOCAL_CSV_PATH = "TRAIN_VAL_DATA/val_df_v11.csv"


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
# Build combined price series

price_series_list = []
date_series_list = []
for ticker in price_df.columns:
    prices = price_df[ticker]
    price_tensor = torch.tensor(prices.to_numpy(), dtype=torch.float32)
    price_series_list.append(price_tensor)
    date_series_list.append(prices.index)

def combine_price_series_with_dates(price_series, date_series, debug=False):
    if not price_series:
        raise ValueError("Empty price series list")

    cleaned_prices = []
    cleaned_dates = []
    for i, series in enumerate(price_series):
        valid_mask = (series != 0) & torch.isfinite(series)
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) == 0:
            if debug:
                print(f"\nWarning: Series {i} has no valid values, skipping")
            continue
        first_valid_idx = valid_indices[0].item()
        cleaned_prices.append(series[first_valid_idx:])
        cleaned_dates.append(date_series[i][first_valid_idx:])

    if not cleaned_prices:
        raise ValueError("No valid price series after cleaning")

    eps = 1e-8
    normalized_series = [cleaned_prices[0]]
    combined_dates = [cleaned_dates[0]]

    for i in range(1, len(cleaned_prices)):
        current_series = cleaned_prices[i]
        prev_series_end = normalized_series[-1][-1]
        scaling_factor = prev_series_end / (current_series[0] + eps)
        if not torch.isfinite(scaling_factor):
            if debug:
                print(f"\nWarning: Invalid scaling factor detected for series {i}")
            scaling_factor = 1.0
        normalized_series.append(current_series * scaling_factor)
        combined_dates.append(cleaned_dates[i])

    combined_series = torch.cat(normalized_series)
    combined_dates = pd.DatetimeIndex(np.concatenate([d.to_numpy() for d in combined_dates]))
    return combined_series, combined_dates


combined_prices, combined_dates = combine_price_series_with_dates(
    price_series_list,
    date_series_list,
)
print("Combined prices length:", len(combined_prices))
print("Combined dates length:", len(combined_dates))

# %%
# Checkpoints

ROOT = Path(__file__).resolve().parents[3]  # FINPAK repo root
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_PATH = ""  # set explicitly to override auto-selection


def find_latest_checkpoint(checkpoint_dir, pattern="*.pt"):
    paths = sorted(Path(checkpoint_dir).glob(pattern), key=lambda p: p.stat().st_mtime)
    return str(paths[-1]) if paths else None


if not CHECKPOINT_PATH:
    CHECKPOINT_PATH = find_latest_checkpoint(CHECKPOINT_DIR)

print("Checkpoint path:", CHECKPOINT_PATH)
# %%
CHECKPOINT_PATH = ROOT / "checkpoints" / "vMP007a_stability_e1142_valloss_0.0000357_tc12_vc0.pt"
print("Checkpoint path:", CHECKPOINT_PATH)

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


checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
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
        for step in range(n_steps):
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

            # Convert predicted returns into a single next-day return
            last_price = current_prices[-1].item()

            if use_multi_horizon:
                horizon_returns = []
                for i, period in enumerate(target_periods):
                    period_return = pred_continuous[0][i].item()
                    abs_return = abs(period_return)
                    sign = 1 if period_return >= 0 else -1
                    daily_return = sign * ((1 + abs_return) ** (return_scaling_power / period) - 1)
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

def plot_prediction(
    historical_prices,
    historical_dates,
    start_index,
    predicted_prices,
    window_size=120,
    title="Price Prediction",
):
    plt.figure(figsize=(14, 7))

    hist_start = max(0, start_index - window_size)
    hist_end = start_index + 1

    x_hist = historical_dates[hist_start:hist_end]
    y_hist = historical_prices[hist_start:hist_end].cpu().numpy()

    x_pred = historical_dates[start_index + 1 : start_index + 1 + len(predicted_prices)]
    y_pred = predicted_prices.cpu().numpy()

    plt.plot(x_hist, y_hist, label="Historical", color="steelblue")
    plt.plot(x_pred, y_pred, label="Predicted", color="firebrick")
    plt.axvline(historical_dates[start_index], color="gray", linestyle=":", alpha=0.6)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%y%m%d"))
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()

# %%
# Backtest-style predictions from multiple start points

def get_backtest_start_indices(
    series_length,
    n_steps,
    n_windows=6,
    stride=60,
    anchor="end",
):
    if anchor == "end":
        last_start = series_length - n_steps - 1
        starts = [last_start - i * stride for i in range(n_windows)]
    else:
        starts = [i * stride for i in range(n_windows)]
    return [s for s in starts if s >= 0]


def run_backtest_predictions(
    model,
    prices,
    config,
    start_indices,
    n_steps,
    start_offsets=(0, 1, 2),
    use_multi_horizon=True,
    temperature=0.01,
    use_sampling=False,
    stability_threshold=0.1,
    dampening_factor=0.95,
    ewma_alpha=0.7,
    return_scaling_power=1.0,
):
    results = []
    for start_index in start_indices:
        if start_index + n_steps >= len(prices):
            continue

        prediction_sets = []
        offsets_used = []
        for offset in start_offsets:
            shifted_start = start_index - offset
            if shifted_start < 0:
                continue
            if shifted_start + n_steps >= len(prices):
                continue
            _, pred_prices = make_autoregressive_prediction(
                model=model,
                prices=prices,
                config=config,
                start_index=shifted_start,
                n_steps=n_steps,
                use_multi_horizon=use_multi_horizon,
                temperature=temperature,
                use_sampling=use_sampling,
                stability_threshold=stability_threshold,
                dampening_factor=dampening_factor,
                ewma_alpha=ewma_alpha,
                return_scaling_power=return_scaling_power,
            )
            prediction_sets.append(pred_prices.cpu().numpy())
            offsets_used.append(offset)

        actual_future = prices[start_index + 1 : start_index + 1 + n_steps].cpu().numpy()
        results.append(
            {
                "start_index": start_index,
                "predictions": prediction_sets,
                "offsets": offsets_used,
                "actual_future": actual_future,
            }
        )
    return results


def plot_backtest_predictions(
    historical_prices,
    historical_dates,
    results,
    window_size=120,
    title="V7 Backtest Predictions",
):
    if not results:
        print("No valid backtest windows to plot.")
        return

    n_plots = len(results)
    n_cols = 2 if n_plots > 1 else 1
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for ax, result in zip(axes, results):
        start_index = result["start_index"]
        prediction_sets = result["predictions"]
        offsets_used = result["offsets"]
        actual_future = result["actual_future"]
        hist_start = max(0, start_index - window_size)
        hist_end = start_index + 1

        x_hist = historical_dates[hist_start:hist_end]
        y_hist = historical_prices[hist_start:hist_end].cpu().numpy()

        if not prediction_sets:
            continue

        aligned_lengths = [
            max(0, len(pred) - offset) for pred, offset in zip(prediction_sets, offsets_used)
        ]
        common_len = min([len(actual_future)] + aligned_lengths)
        if common_len <= 0:
            continue

        x_future = historical_dates[start_index + 1 : start_index + 1 + common_len]
        actual_future = actual_future[:common_len]

        ax.plot(x_hist, y_hist, label="Historical", color="steelblue")
        ax.plot(x_future, actual_future, label="Actual", color="black", linestyle="--", alpha=0.7)

        colors = ["firebrick", "darkorange", "seagreen"]
        aligned_predictions = []
        for pred_prices, offset, color in zip(prediction_sets, offsets_used, colors):
            aligned_pred = pred_prices[offset : offset + common_len]
            aligned_predictions.append(aligned_pred)
            ax.plot(
                x_future,
                aligned_pred,
                label=f"Predicted t-{offset}",
                color=color,
                alpha=0.9,
            )

        if aligned_predictions:
            pred_stack = np.vstack(aligned_predictions)
            pred_min = pred_stack.min(axis=0)
            pred_max = pred_stack.max(axis=0)
            ax.fill_between(
                x_future,
                pred_min,
                pred_max,
                color="gray",
                alpha=0.07,
                linewidth=0,
            )
        ax.axvline(historical_dates[start_index], color="gray", linestyle=":", alpha=0.6)
        ax.set_title(f"Start t={start_index}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y%m%d"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(35)
            tick.set_horizontalalignment("right")
        ax.legend(fontsize=9)

    for ax in axes[len(results):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    plt.show()


N_FUTURE_STEPS = 30
USE_MULTI_HORIZON = True
BACKTEST_WINDOWS = 6
BACKTEST_STRIDE = 60

START_INDICES = get_backtest_start_indices(
    series_length=len(combined_prices),
    n_steps=N_FUTURE_STEPS,
    n_windows=BACKTEST_WINDOWS,
    stride=BACKTEST_STRIDE,
    anchor="end",
)

backtest_results = run_backtest_predictions(
    model=model,
    prices=combined_prices,
    config=CONFIG,
    start_indices=START_INDICES,
    n_steps=N_FUTURE_STEPS,
    use_multi_horizon=USE_MULTI_HORIZON,
    temperature=0.01,
    use_sampling=False,
    stability_threshold=0.1,
    dampening_factor=0.95,
    ewma_alpha=0.7,
    return_scaling_power=1.0,
)

plot_backtest_predictions(
    historical_prices=combined_prices,
    historical_dates=combined_dates,
    results=backtest_results,
    title="V7 Backtest Predictions (Multiple Start Points)",
)

# %%
# Optional: inspect prediction tensor
backtest_results[:1]
