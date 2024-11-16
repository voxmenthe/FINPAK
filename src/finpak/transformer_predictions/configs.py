# flake8: noqa E501
import torch


all_configs = {

    "vMS0001": {
        "model_params": {
            "d_model": 256,
            "n_heads": 32,
            "n_layers": 16,
            "d_ff": 512,
            "dropout": 0.02,
            "use_multi_scale": True,
            "use_relative_pos": False,
            "temporal_scales": [1, 2, 4]
        },
        "data_params": {
            "sequence_length": 47,
            "return_periods": [1, 2, 3, 4],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [9, 28, 47]
        },
        "train_params": {
            "epochs": 320,
            "learning_rate": 1e-4,
            "warmup_steps": 1000,
            "decay_step_multiplier": None,
            "batch_size": 128,
            "patience": 32,
            "max_checkpoints": 5,
            "prefix": "mpvMS0001",
        },
    },

    "vMS0002": {
        "model_params": {
            "d_model": 256,
            "n_heads": 16, 
            "n_layers": 16,
            "d_ff": 256,
            "dropout": 0.02,
            "use_multi_scale": False,
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6] # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 34,
            "return_periods": [1, 2, 3, 4, 6],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4, 6],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12]
        },
        "train_params": {
            "epochs": 640,
            "learning_rate": 3e-5, # try 4 or 5e-5
            "warmup_steps": 2000,
            "decay_step_multiplier": 12,
            "batch_size": 128,
            "patience": 88,
            "max_checkpoints": 5,
            "prefix": "mpvMS0002", # no relative positional encoding
        },
    },

    "vMS0003": {
        "model_params": {
            "d_model": 256,
            "n_heads": 32, 
            "n_layers": 16,
            "d_ff": 512,
            "dropout": 0.02,
            "use_multi_scale": False, # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6] # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 34, # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 6],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4, 6],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12]
        },
        "train_params": {
            "epochs": 1800,
            "learning_rate": 5e-3, # previous 1e-3
            "initial_learning_rate": 1e-6,
            "weight_decay": 0.07, # previous 0.02
            "warmup_steps": 80000,
            "decay_step_multiplier": 0.7,
            "enable_lr_adaptation": True,  # Enable adaptive learning rate
            "lr_acceleration_factor": 1.2,  # Increase learning rate by 20% when improving
            "lr_deceleration_factor": 0.8,  # Decrease learning rate by 20% when stagnating
            "lr_adaptation_epochs": 5,  # Wait 5 epochs before adapting learning rate
            "min_lr": 1e-6,  # Minimum learning rate
            "max_lr": 1e-3,  # Maximum learning rate
            "batch_size": 128,
            "patience": 32,
            "max_checkpoints": 5,
            "prefix": "mpvMS0003", # no relative positional encoding
        },
    },

    "vMS0003a": {
        "model_params": {
            "d_model": 256,
            "n_heads": 32,
            "n_layers": 16,
            "d_ff": 512,
            "dropout": 0.02,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 34,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 6],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4, 6],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12]
        },
        "train_params": {
            "seed": 42,  
            "epochs": 1800,
            "batch_size": 128,
            "patience": 32,
            "max_checkpoints": 5,
            "min_delta": 0.0,  # Minimum change in loss to be considered an improvement
            "learning_rate": 5e-3,
            "initial_learning_rate": 1e-7,
            "weight_decay": 0.07,
            "warmup_steps": 80000,
            "decay_step_multiplier": 0.7,
            "enable_lr_adaptation": True,
            "lr_acceleration_factor": 2.0,
            "lr_deceleration_factor": 0.8,
            "lr_adaptation_epochs": 3,
            "min_lr": 1e-7,
            "max_lr": 1e-2,
            "prefix": "mpvMS0003a",
        }
    },

    "vMS0003b": {
        "model_params": {
            "d_model": 256,
            "n_heads": 32,
            "n_layers": 16,
            "d_ff": 512,
            "dropout": 0.05,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 34,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 6],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4, 6],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12]
        },
        "train_params": {
            "seed": 6657,  
            "epochs": 1800,
            "batch_size": 64,
            "patience": 32,
            "max_checkpoints": 5,
            "min_delta": 0.0,  # Minimum change in loss to be considered an improvement
            "learning_rate": 4e-4,
            "initial_learning_rate": 1e-8,
            "weight_decay": 0.2,
            "warmup_steps": 180000,
            "decay_step_multiplier": 0.7,
            "enable_lr_adaptation": True,
            "lr_acceleration_factor": 1.1,
            "lr_deceleration_factor": 0.8,
            "lr_adaptation_epochs": 3,
            "min_lr": 1e-8,
            "max_lr": 1e-2,
            "prefix": "mpvMS0003b",
        }
    },

    "vMS0003c": {
        "model_params": {
            "d_model": 256,
            "n_heads": 32,
            "n_layers": 16,
            "d_ff": 512,
            "dropout": 0.01,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 34,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 6],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4, 6],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12]
        },
        "train_params": {
            "seed": 6657,
            "epochs": 1800,
            "print_every": 10,
            "batch_size": 32,
            "patience": 32,
            "max_checkpoints": 5,
            "min_delta": 0.0,  # Minimum change in loss to be considered an improvement
            "learning_rate": 4e-4,
            "initial_learning_rate": 1e-9,
            "weight_decay": 0.3,
            "warmup_steps": 300_000,
            "decay_step_multiplier": 0.7,
            "enable_lr_adaptation": True,
            "lr_acceleration_factor": 0.85,
            "lr_deceleration_factor": 0.8,
            "lr_adaptation_epochs": 6,
            "min_lr": 1e-9,
            "max_lr": 1e-2,
            "prefix": "mpvMS0003c",
        }
    },

    "vMS0003d": {
        "model_params": {
            "d_model": 256,
            "n_heads": 32,
            "n_layers": 16,
            "d_ff": 512,
            "dropout": 0.01,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 34,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 6],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4, 6],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12]
        },
        "train_params": {
            "seed": 6657,
            "epochs": 1800,
            "print_every": 10,
            "batch_size": 32,
            "patience": 32,
            "max_checkpoints": 5,
            "min_delta": 0.0,  # Minimum change in loss to be considered an improvement
            "learning_rate": 6e-4,
            "initial_learning_rate": 1e-9,
            "weight_decay": 0.45,
            "warmup_steps": 500_000,
            "decay_step_multiplier": 0.7,
            "enable_lr_adaptation": True,
            "lr_acceleration_factor": 0.75,
            "lr_deceleration_factor": 0.8,
            "lr_adaptation_epochs": 6,
            "min_lr": 1e-9,
            "max_lr": 1e-2,
            "prefix": "mpvMS0003d",
        }
    },

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
            "warmup_steps": 1000,
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
            "warmup_steps": 1000,
            "batch_size": 128,
            "patience": 8,
            "max_checkpoints": 3,
            "prefix": "mpv000",
        },
    },

    "v001": {
        "model_params": {
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 24,
            "d_ff": 1024,
            "dropout": 0.07
        },
        "data_params": {
            "sequence_length": 47,
            "return_periods": [1, 2, 3, 4, 5, 6],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4, 5, 6],
        },
        "train_params": {
            "epochs": 320,
            "learning_rate": 1e-3,
            "warmup_steps": 3000,
            "batch_size": 128,
            "patience": 12,
            "max_checkpoints": 5,
            "prefix": "mpv001",
        },
    },

    "v002": {
        "model_params": {
            "d_model": 256,
            "n_heads": 32,
            "n_layers": 16,
            "d_ff": 512,
            "dropout": 0.02
        },
        "data_params": {
            "sequence_length": 47,
            "return_periods": [1, 2, 3, 4],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4],
        },
        "train_params": {
            "epochs": 320,
            "learning_rate": 1e-4,
            "warmup_steps": 1000,
            "batch_size": 128,
            "patience": 32,
            "max_checkpoints": 5,
            "prefix": "mpv002",
        },
    },

    "v003": {
        "model_params": {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 256,
            "dropout": 0.03
        },
        "data_params": {
            "sequence_length": 47,
            "return_periods": [1, 2, 3, 4],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4],
        },
        "train_params": {
            "epochs": 320,
            "learning_rate": 7e-5,
            "warmup_steps": 1000,
            "decay_step_multiplier": None,
            "batch_size": 128,
            "patience": 32,
            "max_checkpoints": 5,
            "prefix": "mpv003",
        },
    },

    "v004": {
        "model_params": {
            "d_model": 320,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 320,
            "dropout": 0.05
        },
        "data_params": {
            "sequence_length": 47,
            "return_periods": [1, 2, 3, 4, 5],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4, 5],
        },
        "train_params": {
            "epochs": 320,
            "learning_rate": 2e-5,
            "warmup_steps": 1000,
            "decay_step_multiplier": None,
            "batch_size": 128,
            "patience": 32,
            "max_checkpoints": 5,
            "prefix": "mpv004",
        },
    },

    "v005": {
        "model_params": {
            "d_model": 384,
            "n_heads": 8,
            "n_layers": 24,
            "d_ff": 384,
            "dropout": 0.06
        },
        "data_params": {
            "sequence_length": 47,
            "return_periods": [1, 2, 3, 4, 5],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4, 5],
        },
        "train_params": {
            "epochs": 12000,
            "learning_rate": 3e-5,
            "warmup_steps": 2000,
            "decay_step_multiplier": 10,
            "batch_size": 128,
            "patience": 220,
            "max_checkpoints": 5,
            "prefix": "mpv005",
        },
    },

    "v005a": {
        "model_params": {
            "d_model": 384,
            "n_heads": 8,
            "n_layers": 24,
            "d_ff": 384,
            "dropout": 0.06
        },
        "data_params": {
            "sequence_length": 47,
            "return_periods": [1, 2, 3, 4, 5],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4, 5],
        },
        "train_params": {
            "epochs": 12000,
            "learning_rate": 3e-3,
            "warmup_steps": 5000,
            "decay_step_multiplier": 5,
            "batch_size": 128,
            "patience": 50,
            "max_checkpoints": 3,
            "prefix": "mpv005a",
        },
    },

    "v006": {
        "model_params": {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 256,
            "dropout": 0.03
        },
        "data_params": {
            "sequence_length": 47,
            "return_periods": [1, 2, 3, 4],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4],
        },
        "train_params": {
            "epochs": 12000,
            "learning_rate": 4e-4,
            "warmup_steps": 4000,
            "decay_step_multiplier": 8,
            "batch_size": 128,
            "patience": 50,
            "max_checkpoints": 3,
            "prefix": "mpv006",
        },
    },

    "v007": {
        "model_params": {
            "d_model": 512,
            "n_heads": 32,
            "n_layers": 32,
            "d_ff": 512,
            "dropout": 0.05
        },
        "data_params": {
            "sequence_length": 32,
            "return_periods": [1, 2, 3, 4],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4],
        },
        "train_params": {
            "epochs": 12000,
            "learning_rate": 1e-4,
            "warmup_steps": 4000,
            "decay_step_multiplier": 8,
            "batch_size": 128,
            "patience": 50,
            "max_checkpoints": 3,
            "prefix": "mpv007",
        },
    },

    "v008": {
        "model_params": {
            "d_model": 256,
            "n_heads": 4,
            "n_layers": 24,
            "d_ff": 256,
            "dropout": 0.03
        },
        "data_params": {
            "sequence_length": 56,
            "return_periods": [1, 2, 3, 4],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4],
        },
        "train_params": {
            "epochs": 12000,
            "learning_rate": 3e-4,
            "warmup_steps": 4000,
            "decay_step_multiplier": 15,
            "batch_size": 128,
            "patience": 88,
            "max_checkpoints": 3,
            "prefix": "mpv008",
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

train_tickers_v1 = [
    'AAPL', 'AAL', 'AMZN', 'AVGO', 'ADBE', 'AXP', 
    'BA', 'BIIB', 'CLX', 'CMG', 'CRM', 'DIS', 'DE',
    'EBAY', 'ED', 'FDX',
    'GM', 'GD', 'GDX', 'GOOGL', 'GS', 'HD',
    'IBM', 'INTC','ISRG', 
    'JNJ', 'JPM', 
    'KRE', 'KO',
    'LEN', 'LLY','LMT', 'LULU', 'LVS',
    'MA', 'META', 'MGM','MS', 'MSFT', 'NVDA',
    'NOW', 'ORCL',
    'PG', 
    'OXY', 'PANW',
    'LUV', 'PYPL', 
    'SBUX', 'SCHW', 'SMH',
    'TEVA', 'TGT','TOL', 'TSLA',
    'UAL', 'UNH', 'UPS',
    'WBA', 'WMT',
]

val_tickers_v1 = ['UAL', 'SNOW', 'CRWD', 'IBKR', 'AMD', 'COIN'] # 'FTNT', 'CRWD', 'CAVA', 'AMD', 'SNOW', 'UAL', 'DKNG',  # Validation tickers

train_tickers_v2 = [
    'AAPL', 'AAL', 'AMD', 'AMZN', 'AVGO', 'ADBE', 'AXP', 
    'BA', 'BIIB', 'CLX', 'CMG', 'COIN', 'CRM', 'DIS', 'DE',
    'EBAY', 'ED', 'F','FDX',
    'GM', 'GD', 'GDX', 'GOOGL', 'GS', 'HD',
    'IBM', 'INTC','ISRG', 
    'JNJ', 'JPM', 
    'KRE', 'KO',
    'LEN', 'LLY','LMT', 'LULU', 'LVS',
    'MA', 'META', 'MGM','MS', 'MSFT', 'MU','NVDA',
    'NOW', 'ORCL',
    'PG', 
    'OXY', 'PANW',
    'LUV', 'PYPL', 
    'SBUX', 'SCHW', 'SMH',
    'TEVA', 'TGT','TOL', 'TSLA',
    'UAL', 'UNH', 'UPS',
    'WBA', 'WMT', 'X', 'XOM'
]

val_tickers_v2 = ['UAL', 'SNOW', 'CRWD', 'IBKR', 'FTNT', 'CRWD', 'CAVA', 'DKNG']  # Validation tickers  


train_tickers_v3 = [
    'AAPL', 'AAL', 'AMD', 'AMZN', 'AVGO', 'ADBE', 'AXP', 
    'BA', 'BIIB', 'CLX', 'CMG', 'COIN', 'CRM', 'DIS', 'DE',
    'EBAY', 'ED', 'F','FDX',
    'GM', 'GD', 'GDX', 'GOOGL', 'GS', 
    'H', 'HD', 'HEES', 'HON',
    'IBM', 'INTC','ISRG', 
    'JNJ', 'JPM', 
    'KRE', 'KO',
    'LEN', 'LLY','LMT', 'LULU', 'LUV', 'LVS',
    'MA', 'META', 'MGM','MS', 'MSFT', 'MSTR', 'MU',
    'NVDA', 'NOW', 'ORCL', 'OXY', 'PANW', 'PG', 'PYPL', 
    'SBUX', 'SCHW', 'SMH',
    'TEVA', 'TGT','TOL', 'TSLA',
    'UAL', 'UNH', 'UPS',
    'WBA', 'WMT', 'X', 'XOM'
]

val_tickers_v3 = ['UAL', 'SNOW', 'PLTR', 'SHOP', 'CRWD', 'IBKR', 'FTNT', 'CRWD', 'CAVA', 'IBIT', 'DKNG']  # Validation tickers  


train_tickers_v4 = [
    'AAPL', 'AAL', 'AMD', 'AMZN', 'AVGO', 'ADBE', 'AXP', 
    'BA', 'BIIB', 'CLX', 'CMG', 'COIN', 'CRM', 
    'DAL','DIS', 'DE',
    'EBAY', 'ED', 'F','FDX',
    'GM', 'GD', 'GDX', 'GOOGL', 'GS', 
    'H', 'HD', 'HEES', 'HON',
    'IBM', 'INTC','ISRG', 
    'JNJ', 'JPM', 
    'KRE', 'KO',
    'LEN', 'LLY','LMT', 'LULU', 'LUV', 'LVS',
    'MA', 'MCD', 'META', 'MGM','MS', 'MSFT', 'MSTR', 'MU',
    'NOW', 'NVDA', 'NVO', 
    'ORCL', 'OXY', 'PANW', 
    'PG', 'PYPL', 'QCOM',
    'SBUX', 'SCHW', 'SMH',
    'TEVA', 'TGT','TOL', 'TSLA',
    'UBER','UAL', 'UNH', 'UPS', 'V'
    'WBA', 'WMT', 'X', 'XHB','XOM'
]

val_tickers_v4 = [
    'SNOW', 'PANW', 'PLTR', 'LYFT','SHOP', 
    'CRWD', 'IBKR', 'FTNT', 'CRWD', 'CAVA', 
    'IBIT', 'AMGN', 'DKNG'] # 'SNAP', 'SOFI'
