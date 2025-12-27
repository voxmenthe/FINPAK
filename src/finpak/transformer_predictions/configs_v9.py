# flake8: noqa E501
import torch


all_configs = {
    "v9_stability_base": {
        "model_params": {
            "d_model": 384,
            "n_heads": 16,
            "n_layers": 24,
            "d_ff": 912,
            "dropout": 0.05,
            "use_multi_scale": False,
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]
        },
        "data_params": {
            "sequence_length": 32,
            "return_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "sma_periods": [3, 7, 17, 25, 36, 46],
            "target_periods": [1, 2],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [4, 9, 15, 21, 37, 46],
            "reverse_tickers": False,
            "categorical_features": [
                {"name": "nd_high_low_asym", "type": "nd_high_low_asym", "n": 9, "k": 27},
                {"name": "roc_gt_shift", "type": "roc_gt_shift", "p": 8, "nperiod": 27, "k": 4},
                {"name": "roc_lt_shift", "type": "roc_lt_shift", "p": 8, "nperiod": 27, "k": 4}
            ],
            "continuous_features_extra": [
                {"name": "dist_from_Nd_high_over_Kd_range", "n": 9, "k": 27, "eps": 1e-8}
            ]
        },
        "train_params": {
            "seed": 6123,
            "epochs": 2000,
            "print_every": 10,
            "batch_size": 64,
            "patience": 20,
            "max_checkpoints": 7,
            "min_delta": 1e-10,
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 3e-5,
                "max_lr": 3e-4,
                "min_lr": 1e-6,
                "warmup_epochs": 8,
                "weight_decay": 0.05
            },
            "validation_subset_size": 2,
            "validation_overlap": 1,
            "train_subset_size": 15,
            "train_overlap": 3,
            "prefix": "v9_stability_base",
            "architecture_version": "v9",
            "run_id": "0"
        }
    },
    "v9_stability_followon": {
        "model_params": {
            "d_model": 384,
            "n_heads": 16,
            "n_layers": 24,
            "d_ff": 912,
            "dropout": 0.05,
            "use_multi_scale": False,
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]
        },
        "data_params": {
            "sequence_length": 48,
            "return_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "sma_periods": [3, 7, 17, 25, 36, 46],
            "target_periods": [1, 3],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [4, 9, 15, 21, 37, 46],
            "reverse_tickers": False,
            "categorical_features": [
                {"name": "nd_high_low_asym", "type": "nd_high_low_asym", "n": 17, "k": 27},
                {"name": "roc_gt_shift", "type": "roc_gt_shift", "p": 16, "nperiod": 38, "k": 4},
                {"name": "roc_lt_shift", "type": "roc_lt_shift", "p": 16, "nperiod": 38, "k": 4}
            ],
            "continuous_features_extra": [
                {"name": "dist_from_Nd_high_over_Kd_range", "n": 17, "k": 38, "eps": 1e-8}
            ]
        },
        "train_params": {
            "seed": 6123,
            "epochs": 2000,
            "print_every": 10,
            "batch_size": 64,
            "patience": 20,
            "max_checkpoints": 7,
            "min_delta": 1e-10,
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 3e-5,
                "max_lr": 3e-4,
                "min_lr": 1e-6,
                "warmup_epochs": 8,
                "weight_decay": 0.05
            },
            "validation_subset_size": 2,
            "validation_overlap": 1,
            "train_subset_size": 15,
            "train_overlap": 3,
            "prefix": "v9_stability_followon",
            "architecture_version": "v9",
            "run_id": "0"
        }
    }
}
