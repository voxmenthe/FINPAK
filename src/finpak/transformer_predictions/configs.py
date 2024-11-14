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
            "n_heads": 16, 
            "n_layers": 16,
            "d_ff": 512,
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
            "learning_rate": 1e-3, # try 4 or 5e-5
            "weight_decay": 0.02,
            "warmup_steps": 35000,
            "decay_step_multiplier": 0.7,
            "batch_size": 128,
            "patience": 32,
            "max_checkpoints": 5,
            "prefix": "mpvMS0003", # no relative positional encoding
        },
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
    'UAL', 'UNH', 'UPS', 'V'
    'WBA', 'WMT', 'X', 'XOM'
]

val_tickers_v4 = ['SNOW', 'PLTR', 'SHOP', 'CRWD', 'IBKR', 'FTNT', 'CRWD', 'CAVA', 'IBIT', 'AMGN', 'DKNG']  # Validation tickers  