import re, os, torch, argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
from data_loading_v8 import create_subset_dataloaders
from timeseries_decoder_v8 import TimeSeriesDecoder
from train_v8 import train_model
import pandas as pd
from finpak.data.fetchers.yahoo import download_multiple_tickers
from ticker_cycler import TickerCycler
from configs import all_configs
from ticker_configs import train_tickers_v11, val_tickers_v11


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@dataclass
class CheckpointLoadReport:
    loaded_keys: List[str]
    skipped_keys: List[str]
    expanded_keys: List[str]


@dataclass
class RegistryCheckResult:
    compatible: bool
    reason: Optional[str]
    added_features: List[str]
    added_targets: List[str]


def _init_tensor(tensor: torch.Tensor, init_mode: str) -> torch.Tensor:
    if init_mode == "zeros":
        tensor.zero_()
    elif init_mode == "normal":
        torch.nn.init.normal_(tensor, mean=0.0, std=0.02)
    elif init_mode == "kaiming":
        torch.nn.init.kaiming_uniform_(tensor, a=0.01, nonlinearity="leaky_relu")
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")
    return tensor


def _expand_linear_out(weight: torch.Tensor, bias: torch.Tensor, out_features: int, init_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if weight.size(0) >= out_features:
        return weight[:out_features].clone(), bias[:out_features].clone()
    new_weight = weight.new_zeros((out_features, weight.size(1)))
    new_bias = bias.new_zeros((out_features,))
    new_weight[:weight.size(0)] = weight
    new_bias[:bias.size(0)] = bias
    _init_tensor(new_weight[weight.size(0):], init_mode)
    _init_tensor(new_bias[bias.size(0):], "zeros")
    return new_weight, new_bias


def _expand_linear_in(weight: torch.Tensor, out_features: int, in_features: int, init_mode: str) -> torch.Tensor:
    if weight.size(0) != out_features:
        raise ValueError(f"Unexpected out_features mismatch: {weight.size(0)} vs {out_features}")
    if weight.size(1) >= in_features:
        return weight[:, :in_features].clone()
    new_weight = weight.new_zeros((out_features, in_features))
    new_weight[:, :weight.size(1)] = weight
    _init_tensor(new_weight[:, weight.size(1):], init_mode)
    return new_weight


def _maybe_expand_input_projection(
    model: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    init_mode: str
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    expanded = []
    w_key = "input_embedding.continuous_projection.weight"
    b_key = "input_embedding.continuous_projection.bias"
    if w_key not in state or b_key not in state:
        return state, expanded

    target_w = model.state_dict()[w_key]
    target_b = model.state_dict()[b_key]
    src_w = state[w_key]
    src_b = state[b_key]
    if src_w.shape == target_w.shape and src_b.shape == target_b.shape:
        return state, expanded

    # Expand only input feature dimension; output dimension should match model's d_continuous_proj.
    if src_w.size(0) != target_w.size(0):
        return state, expanded

    expanded_w = _expand_linear_in(src_w, target_w.size(0), target_w.size(1), init_mode)
    expanded_b = src_b[:target_b.size(0)].clone()
    state[w_key] = expanded_w
    state[b_key] = expanded_b
    expanded.extend([w_key, b_key])
    return state, expanded


def _maybe_expand_output_projection(
    model: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    init_mode: str
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    expanded = []
    w_key = "continuous_projection.weight"
    b_key = "continuous_projection.bias"
    if w_key not in state or b_key not in state:
        return state, expanded

    target_w = model.state_dict()[w_key]
    target_b = model.state_dict()[b_key]
    src_w = state[w_key]
    src_b = state[b_key]
    if src_w.shape == target_w.shape and src_b.shape == target_b.shape:
        return state, expanded

    if src_w.size(1) != target_w.size(1):
        return state, expanded

    expanded_w, expanded_b = _expand_linear_out(src_w, src_b, target_w.size(0), init_mode)
    state[w_key] = expanded_w
    state[b_key] = expanded_b
    expanded.extend([w_key, b_key])
    return state, expanded


def _maybe_expand_categorical_projections(
    model: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    init_mode: str
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    expanded = []
    target_state = model.state_dict()
    for key, tensor in list(state.items()):
        if not key.startswith("categorical_projections."):
            continue
        if key not in target_state:
            continue
        target_tensor = target_state[key]
        if tensor.shape == target_tensor.shape:
            continue
        # Expand only output dimension (n_bins) or number of heads; keep d_model stable.
        if tensor.dim() == 2 and tensor.size(1) == target_tensor.size(1):
            expanded_tensor = tensor.new_zeros(target_tensor.shape)
            rows = min(tensor.size(0), target_tensor.size(0))
            expanded_tensor[:rows] = tensor[:rows]
            _init_tensor(expanded_tensor[rows:], init_mode)
            state[key] = expanded_tensor
            expanded.append(key)
    return state, expanded


def load_checkpoint_with_expansion(
    model: torch.nn.Module,
    checkpoint_state: Dict[str, torch.Tensor],
    strict: bool = False,
    init_mode: str = "zeros"
) -> CheckpointLoadReport:
    state = dict(checkpoint_state)
    expanded_keys = []

    state, expanded = _maybe_expand_input_projection(model, state, init_mode)
    expanded_keys.extend(expanded)
    state, expanded = _maybe_expand_output_projection(model, state, init_mode)
    expanded_keys.extend(expanded)
    state, expanded = _maybe_expand_categorical_projections(model, state, init_mode)
    expanded_keys.extend(expanded)

    model_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []
    loaded_keys = []
    for key, tensor in state.items():
        if key not in model_state:
            skipped_keys.append(key)
            continue
        if tensor.shape != model_state[key].shape:
            skipped_keys.append(key)
            continue
        filtered_state[key] = tensor
        loaded_keys.append(key)

    model.load_state_dict(filtered_state, strict=strict)
    return CheckpointLoadReport(
        loaded_keys=loaded_keys,
        skipped_keys=skipped_keys,
        expanded_keys=expanded_keys
    )


def _infer_checkpoint_io_dimensions(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    io_dims = {}
    in_w = state.get("input_embedding.continuous_projection.weight")
    out_w = state.get("continuous_projection.weight")
    if in_w is not None:
        io_dims["d_continuous"] = in_w.size(1)
        io_dims["d_continuous_proj"] = in_w.size(0)
    if out_w is not None:
        io_dims["n_continuous_outputs"] = out_w.size(0)
        io_dims["d_model"] = out_w.size(1)
    return io_dims


def _registry_match(old: List[str], new: List[str]) -> RegistryCheckResult:
    if old == new:
        return RegistryCheckResult(True, None, [], [])
    if len(old) <= len(new) and old == new[:len(old)]:
        added = new[len(old):]
        return RegistryCheckResult(True, None, added, [])
    if len(new) <= len(old) and new == old[:len(new)]:
        removed = old[len(new):]
        return RegistryCheckResult(False, f"Registry shrinking not allowed. Removed: {removed}", [], [])
    return RegistryCheckResult(False, "Registry order mismatch detected", [], [])


def check_feature_target_registry(
    previous: Optional[Dict[str, Optional[List[str]]]],
    current: Dict[str, Optional[List[str]]]
) -> RegistryCheckResult:
    if not previous:
        return RegistryCheckResult(True, None, [], [])

    added_features = []
    added_targets = []

    for key in ("continuous_features", "categorical_features"):
        old = previous.get(key) or []
        new = current.get(key) or []
        result = _registry_match(old, new)
        if not result.compatible:
            return result
        added_features.extend(result.added_features)

    for key in ("continuous_targets", "categorical_targets"):
        old = previous.get(key) or []
        new = current.get(key) or []
        result = _registry_match(old, new)
        if not result.compatible:
            return result
        added_targets.extend(result.added_features)

    return RegistryCheckResult(True, None, added_features, added_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train transformer model with specified config')
    parser.add_argument('--config', type=str, required=True, help='Configuration version (e.g., vMP004a)')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from (optional)')
    parser.add_argument('--unsafe-load-checkpoint', action='store_true',
                        help='Use torch.load(weights_only=False). Only use for trusted checkpoints.')
    parser.add_argument('--expand-checkpoint-io', action='store_true',
                        help='Expand input/output layers to fit new feature/target counts.')
    parser.add_argument('--strict-checkpoint', action='store_true',
                        help='Enforce strict state_dict loading after expansion.')
    parser.add_argument('--checkpoint-io-init', type=str, default='zeros',
                        choices=['zeros', 'normal', 'kaiming'],
                        help='Initialization for newly expanded IO weights.')
    parser.add_argument('--enforce-registry', action='store_true',
                        help='Fail if feature/target registry order mismatches checkpoint.')
    parser.add_argument('--io-warmup-epochs', type=int, default=None,
                        help='Freeze backbone for N epochs so IO layers adapt first.')
    args = parser.parse_args()

    # Use this device throughout your code
    device = get_device()

    print(f"Using device: {device}")

    if args.config not in all_configs:
        raise ValueError(f"Config '{args.config}' not found. Available configs: {list(all_configs.keys())}")
    
    CONFIG = all_configs[args.config]
    print(CONFIG)
    
    # Use checkpoint path from command line argument
    checkpoint_path = args.checkpoint
    if checkpoint_path:
        print(f"Will attempt to load checkpoint from: {checkpoint_path}")

    checkpoint_dir = 'checkpoints'

    train_tickers = train_tickers_v11
    val_tickers = val_tickers_v11

    if CONFIG['data_params'].get('reverse_tickers', False):
        train_tickers = train_tickers[::-1]
        val_tickers = val_tickers[::-1]

    train_df_fname = 'TRAIN_VAL_DATA/train_df_v11.csv'
    val_df_fname = 'TRAIN_VAL_DATA/val_df_v11.csv'
    os.makedirs(os.path.dirname(train_df_fname), exist_ok=True)
    FORCE_RELOAD = False

    start_date = '1986-01-01'
    end_date = '2025-12-26'

    # Extract only parameters needed for data loading and model initialization
    batch_size = CONFIG['train_params']['batch_size']
    sequence_length = CONFIG['data_params']['sequence_length']
    return_periods = CONFIG['data_params']['return_periods']
    sma_periods = CONFIG['data_params']['sma_periods']
    target_periods = CONFIG['data_params']['target_periods']
    use_volatility = CONFIG['data_params'].get('use_volatility', False)
    use_momentum = CONFIG['data_params'].get('use_momentum', False)
    momentum_periods = CONFIG['data_params'].get('momentum_periods', [9, 28, 47])

    # Calculate number of continuous and categorical features
    n_continuous = len(return_periods) + len(sma_periods)
    if use_volatility:
        n_continuous += len(return_periods)
    if use_momentum:
        n_continuous += len(momentum_periods)
    
    # Categorical features from binned returns (currently only 1-day returns are binned)
    price_change_bins = CONFIG['data_params'].get('price_change_bins')
    target_bins = CONFIG['data_params'].get('target_bins')
    n_categorical = 1 if price_change_bins else 0
    if price_change_bins:
        n_bins = price_change_bins.get('n_bins', 10)
    elif target_bins:
        n_bins = target_bins.get('n_bins', 10)
    else:
        n_bins = 10

    # Get loss weights for continuous and categorical predictions
    loss_weights = {
        'continuous': CONFIG['train_params'].get('continuous_loss_weight', 1.0),
        'categorical': CONFIG['train_params'].get('categorical_loss_weight', 1.0)
    }
    CONFIG['train_params'].setdefault('loss_weights', loss_weights)

    # Calculate number of outputs (one per target period)
    n_continuous_outputs = len(target_periods)
    n_categorical_outputs = len(target_periods) if target_bins else None

    # Validation cycling parameters
    validation_subset_size = CONFIG['train_params'].get('validation_subset_size', len(val_tickers))
    validation_overlap = CONFIG['train_params'].get('validation_overlap', 0)

    DEBUG = False
    MODEL_PARAMS = CONFIG['model_params']
    
    # Create model
    model = TimeSeriesDecoder(
        d_continuous=n_continuous,
        n_categorical=n_categorical,
        n_bins=n_bins,
        d_model=MODEL_PARAMS['d_model'],
        n_heads=MODEL_PARAMS['n_heads'],
        n_layers=MODEL_PARAMS['n_layers'],
        d_ff=MODEL_PARAMS['d_ff'],
        dropout=MODEL_PARAMS['dropout'],
        n_continuous_outputs=n_continuous_outputs,
        n_categorical_outputs=n_categorical_outputs,
        use_multi_scale=MODEL_PARAMS.get('use_multi_scale', False),
        use_relative_pos=MODEL_PARAMS.get('use_relative_pos', False),
        use_hope_pos=MODEL_PARAMS.get('use_hope_pos', False),
        temporal_scales=MODEL_PARAMS.get('temporal_scales', [1, 2, 4]),
        base=10000
    ).to(device)

    CONFIG['train_params']['io_dimensions'] = {
        'd_continuous': n_continuous,
        'n_continuous_outputs': n_continuous_outputs,
        'n_categorical_outputs': 0 if n_categorical_outputs is None else n_categorical_outputs,
        'n_categorical_features': n_categorical,
        'n_bins': n_bins
    }
    CONFIG['train_params']['feature_registry'] = registry
    if args.io_warmup_epochs is not None:
        CONFIG['train_params']['io_warmup_epochs'] = args.io_warmup_epochs
    else:
        CONFIG['train_params'].setdefault('io_warmup_epochs', 0)

    # Load or download data
    if os.path.exists(train_df_fname) and os.path.exists(val_df_fname) and not FORCE_RELOAD:
        print("Loading existing data files...")
        train_df = pd.read_csv(train_df_fname, index_col=0)
        val_df = pd.read_csv(val_df_fname, index_col=0)
        
        # Convert index to datetime
        train_df.index = pd.to_datetime(train_df.index)
        val_df.index = pd.to_datetime(val_df.index)
        
        # Forward fill any missing values within each ticker's series
        train_df = train_df.ffill()
        val_df = val_df.ffill()
        
        # Drop any remaining NaN values
        train_df = train_df.dropna(axis=0, how='any')
        val_df = val_df.dropna(axis=0, how='any')
        
        if DEBUG:
            print("\nData loading statistics:")
            print("Training data:")
            print(f"Shape: {train_df.shape}")
            print(f"Date range: {train_df.index[0]} to {train_df.index[-1]}")
            print(f"NaN counts:\n{train_df.isna().sum()}")
            print("\nValidation data:")
            print(f"Shape: {val_df.shape}")
            print(f"Date range: {val_df.index[0]} to {val_df.index[-1]}")
            print(f"NaN counts:\n{val_df.isna().sum()}")
        
        # Debug: Print available tickers
        print("\nAvailable tickers in training data:", sorted(train_df.columns.tolist()))
        missing_train = sorted(set(train_tickers) - set(train_df.columns))
        print("\nMissing tickers in training data:", missing_train)
        print("\nAvailable tickers in validation data:", sorted(val_df.columns.tolist()))
        missing_val = sorted(set(val_tickers) - set(val_df.columns))
        print("\nMissing tickers in validation data:", missing_val)

        # Remove missing tickers from lists
        if missing_train:
            print(f"\nRemoving {len(missing_train)} missing tickers from training set")
            train_tickers = [t for t in train_tickers if t not in missing_train]
        
        if missing_val:
            print(f"\nRemoving {len(missing_val)} missing tickers from validation set")
            val_tickers = [t for t in val_tickers if t not in missing_val]

    else:
        # Download and process training data
        train_df = download_multiple_tickers(train_tickers, start_date, end_date)
        train_df = train_df.loc[:, 'Adj Close']
        train_df = train_df.dropna(axis=1, how='all')
        train_df.index = pd.to_datetime(train_df.index)
        train_df = train_df.ffill()
        train_df = train_df.dropna(axis=0, how='any')
        train_df.to_csv(train_df_fname)
        
        # Download and process validation data
        val_df = download_multiple_tickers(val_tickers, start_date, end_date)
        val_df = val_df.loc[:, 'Adj Close']
        val_df = val_df.dropna(axis=1, how='all')
        val_df.index = pd.to_datetime(val_df.index)
        val_df = val_df.ffill()
        val_df = val_df.dropna(axis=0, how='any')
        val_df.to_csv(val_df_fname)

        missing_train = sorted(set(train_tickers) - set(train_df.columns))
        missing_val = sorted(set(val_tickers) - set(val_df.columns))
        if missing_train:
            print(f"\nRemoving {len(missing_train)} missing tickers from training set")
            train_tickers = [t for t in train_tickers if t not in missing_train]
        if missing_val:
            print(f"\nRemoving {len(missing_val)} missing tickers from validation set")
            val_tickers = [t for t in val_tickers if t not in missing_val]

    # Create cyclers with filtered ticker lists
    validation_cycler = TickerCycler(
        tickers=val_tickers,
        subset_size=validation_subset_size,
        overlap_size=validation_overlap,
        reverse_tickers=CONFIG['data_params'].get('reverse_tickers', False),
        use_anchor=CONFIG['data_params'].get('use_anchor', False)
    )

    train_cycler = TickerCycler(
        tickers=train_tickers,
        subset_size=CONFIG['train_params'].get('train_subset_size', len(train_tickers)),
        overlap_size=CONFIG['train_params'].get('train_overlap', 0),
        reverse_tickers=CONFIG['data_params'].get('reverse_tickers', False),
        use_anchor=CONFIG['data_params'].get('use_anchor', False)
    )

    # Get initial subsets from cyclers
    initial_train_tickers = train_cycler.get_current_subset()
    initial_val_tickers = validation_cycler.get_current_subset()

    # Create initial dataloaders
    train_loader, val_loader, registry = create_subset_dataloaders(
        train_df=train_df,
        val_df=val_df,
        train_tickers=initial_train_tickers,
        val_tickers=initial_val_tickers,
        config=CONFIG,
        debug=DEBUG,
        return_registry=True
    )

    # Load checkpoint if specified
    start_epoch = 0
    trained_subsets = []  # Track which subsets we've trained on

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            if args.unsafe_load_checkpoint:
                print("WARNING: Loading checkpoint with weights_only=False (unsafe).")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            else:
                # Allowlist numpy types used by older checkpoints when weights_only=True (PyTorch 2.6+ default).
                safe_globals = []
                if hasattr(np, "_core") and hasattr(np._core, "multiarray"):
                    safe_globals.append(np._core.multiarray.scalar)
                if hasattr(np, "dtype"):
                    safe_globals.append(np.dtype)
                if hasattr(np, "dtypes") and hasattr(np.dtypes, "Float64DType"):
                    safe_globals.append(np.dtypes.Float64DType)
                try:
                    if safe_globals:
                        with torch.serialization.safe_globals(safe_globals):
                            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                    else:
                        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                except Exception as e:
                    print(f"Safe checkpoint load failed: {e}")
                    print("WARNING: Falling back to weights_only=False (unsafe).")
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            previous_registry = None
            if 'metadata' in checkpoint and hasattr(checkpoint['metadata'], 'feature_registry'):
                previous_registry = checkpoint['metadata'].feature_registry

            registry_check = check_feature_target_registry(previous_registry, registry)
            if not registry_check.compatible:
                message = f"Feature/target registry mismatch: {registry_check.reason}"
                if args.enforce_registry:
                    raise ValueError(message)
                print(f"WARNING: {message}")
            else:
                if registry_check.added_features or registry_check.added_targets:
                    print("Registry expansion detected:")
                    if registry_check.added_features:
                        print(f"  Added features: {registry_check.added_features}")
                    if registry_check.added_targets:
                        print(f"  Added targets: {registry_check.added_targets}")

            if args.expand_checkpoint_io:
                report = load_checkpoint_with_expansion(
                    model=model,
                    checkpoint_state=checkpoint['model_state_dict'],
                    strict=args.strict_checkpoint,
                    init_mode=args.checkpoint_io_init
                )
                print(f"Checkpoint load: {len(report.loaded_keys)} tensors loaded, "
                      f"{len(report.skipped_keys)} skipped, {len(report.expanded_keys)} expanded.")
                if report.expanded_keys:
                    print("Expanded keys:")
                    for key in report.expanded_keys:
                        print(f"  - {key}")
                if report.skipped_keys:
                    print("Skipped keys:")
                    for key in report.skipped_keys:
                        print(f"  - {key}")
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            checkpoint_io_dims = None
            if 'metadata' in checkpoint and hasattr(checkpoint['metadata'], 'io_dimensions'):
                checkpoint_io_dims = checkpoint['metadata'].io_dimensions
            if checkpoint_io_dims is None:
                checkpoint_io_dims = _infer_checkpoint_io_dimensions(checkpoint['model_state_dict'])
            CONFIG['train_params']['previous_io_dimensions'] = checkpoint_io_dims
            
            # Try to load metadata from new format first
            try:
                metadata = checkpoint['metadata']
                start_epoch = metadata.epoch + 1
                print(f"Resuming training from epoch {start_epoch}")
                
                # Load trained subsets history
                if hasattr(metadata, 'trained_subsets'):
                    trained_subsets = metadata.trained_subsets
                    print(f"Loaded training history for {len(trained_subsets)} subsets")
                    
                    # Set train_cycler to next subset after last trained
                    if trained_subsets:
                        while train_cycler.get_current_subset() != trained_subsets[-1]:
                            if not train_cycler.has_more_subsets():
                                print("Warning: Could not find last trained subset, starting from beginning")
                                train_cycler.reset()
                                break
                            train_cycler.next_subset()
                        # Move to next subset
                        if train_cycler.has_more_subsets():
                            train_cycler.next_subset()
                else:
                    # Try to infer from checkpoint name
                    match = re.search(r'_tc(\d+)_', checkpoint_path)
                    if match:
                        train_cycle = int(match.group(1))
                        print(f"Inferred train cycle {train_cycle} from checkpoint name")
                        # Set train_cycler to next subset
                        while train_cycler.current_subset_idx < train_cycle:
                            if not train_cycler.has_more_subsets():
                                print("Warning: Could not reach inferred train cycle, starting from beginning")
                                train_cycler.reset()
                                break
                            train_cycler.next_subset()
                        if train_cycler.has_more_subsets():
                            train_cycler.next_subset()
                    else:
                        print("No train subset history found, starting from first subset")
                
                if DEBUG:
                    print("\nLoaded checkpoint metadata:")
                    print(f"  - Original config: {metadata.config}")
                    print(f"  - Training loss: {metadata.train_loss}")
                    print(f"  - Validation loss: {metadata.val_loss}")
                    print(f"  - Model parameters: {metadata.model_params}")
                    print(f"  - Saved at: {metadata.timestamp}")
                    print(f"  - Current train subset: {train_cycler.get_current_subset()}")
                    print(f"  - Current validation subset: {validation_cycler.get_current_subset()}")
            
            # Fall back to old format if metadata not found
            except (KeyError, AttributeError) as e:
                print(f"Loading checkpoint metadata (old format): {e}")
                start_epoch = checkpoint.get('epoch', 0) + 1
                print(f"Resuming training from epoch {start_epoch}")
                
                # Try to infer from checkpoint name
                match = re.search(r'_tc(\d+)_', checkpoint_path)
                if match:
                    train_cycle = int(match.group(1))
                    print(f"Inferred train cycle {train_cycle} from checkpoint name")
                    # Set train_cycler to next subset
                    while train_cycler.current_subset_idx < train_cycle:
                        if not train_cycler.has_more_subsets():
                            print("Warning: Could not reach inferred train cycle, starting from beginning")
                            train_cycler.reset()
                            break
                        train_cycler.next_subset()
                    if train_cycler.has_more_subsets():
                        train_cycler.next_subset()
        
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from epoch 0")
            start_epoch = 0
            train_cycler.reset()

    # Print model size and trainable parameters
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model with simplified interface
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        start_epoch=start_epoch,
        config=CONFIG,
        checkpoint_dir=checkpoint_dir,
        validation_cycler=validation_cycler,
        train_cycler=train_cycler,
        train_df=train_df,
        val_df=val_df,
        debug=DEBUG,
        trained_subsets=trained_subsets
    )
