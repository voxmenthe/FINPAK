import torch


all_configs = {

    "test_fourier": {
        "model_params": {
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 8,
            "d_ff": 32,
            "dropout": 0.02,
        },
        "data_params": {
            "sequence_length": 47,
            "return_periods": [1, 4],
            "sma_periods": [20],
            "target_periods": [1, 4],
        },
        "fourier_params": {
            "n_bins": 16,
            "n_freqs": 16,
            # Define bin edges
            "min_return": -0.1,
            "max_return": 0.1,
            # fill in with min and max return and number of bins + 1
            "bin_edges": torch.linspace(-0.1, 0.1, 16 + 1)  # Edges from -0.1 to 0.1
        },
        "train_params": {
            "epochs": 20,
            "learning_rate": 1e-3,
            "batch_size": 128,
            "patience": 10,
            "max_checkpoints": 3,
            "prefix": "test_fourier",
        },
    },

    "v0": {
        "model_params": {
            "d_model": 512,
            "n_heads": 4,
            "n_layers": 56,
            "d_ff": 2048,
            "dropout": 0.12,
        },
    },

    "v000": {
        "model_params": {
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 8,
            "d_ff": 128,
            "dropout": 0.02
        },
        "data_params": {
            "sequence_length": 47,
            "return_periods": [1, 2, 3, 4],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4],
        },
        "train_params": {
            "epochs": 250,
            "learning_rate": 3e-5,
            "batch_size": 128,
            "patience": 8,
            "max_checkpoints": 3,
            "prefix": "mpv000",
        },
    },

    "v1": {
        "model_params": {
            "d_model": 1024,
            "n_heads": 8,
            "n_layers": 88,
            "d_ff": 2048,
            "dropout": 0.32,
        },
    },

    "v1a": {
        "model_params": {
            "d_model": 1024,
            "n_heads": 8,
            "n_layers": 88,
            "d_ff": 3072,
            "dropout": 0.23,
        },
    },

    "v1b": {
        "model_params": {
            "d_model": 2048,
            "n_heads": 16,
            "n_layers": 88,
            "d_ff": 3072,
            "dropout": 0.28,
        },
    },

    "v2": {
        "model_params": {
            "d_model": 2048,
            "n_heads": 32,
            "n_layers": 96,
            "d_ff": 8192,
            "dropout": 0.12,
        },
    }
}