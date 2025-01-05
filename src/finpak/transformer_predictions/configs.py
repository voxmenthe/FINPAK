# flake8: noqa E501
import torch


all_configs = {

    "test": {
        "model_params": {
            "d_model": 512,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 1024,
            "dropout": 0.47,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 56,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "sma_periods": [3, 7, 17, 25, 36, 46],
            "target_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [4, 9, 15, 21, 37, 46],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 24, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 6, # 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "test",
            "architecture_version": "v4",
            "run_id": "0"
        },
        "augmentation_params": {  # New section
            "enabled": True,
            "t_dist_df": 6,       # Degrees of freedom for Student-t distribution
            "scale_factor": 0.05,  # Fraction of std dev to use for noise
            "subset_fraction": 0.3  # Fraction of subset to augment
        }
    },


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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 1e-4,
                "warmup_steps": 1000,
                "decay_step_multiplier": None,
            },
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 3e-5,
                "warmup_steps": 2000,
                "decay_step_multiplier": 12,
            },
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
            "scheduler": {
                "type": "cyclical",  
                "base_lr": 5e-3,
                "max_lr": 1e-2,     
                "min_lr": 1e-6,     
                "warmup_epochs": 10,  
                "cycle_params": {
                    "cycle_length": 50,     
                    "decay_factor": 0.85,   
                    "cycles": 10            
                }
            },
            "weight_decay": 0.07,
            "batch_size": 128,
            "patience": 88,
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 5e-3,
                "initial_lr": 1e-7,
                "warmup_steps": 80000,
                "decay_step_multiplier": 0.7,
            },
            "weight_decay": 0.07,
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 4e-4,
                "initial_lr": 1e-8,
                "warmup_steps": 180000,
                "decay_step_multiplier": 0.7,
            },
            "weight_decay": 0.2,
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
            "print_every": 10,
            "batch_size": 32,
            "patience": 32,
            "max_checkpoints": 5,
            "min_delta": 0.0,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 4e-4,
                "initial_lr": 1e-9,
                "warmup_steps": 300_000,
                "decay_step_multiplier": 0.7,
            },
            "weight_decay": 0.3,
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
            "use_volatility": False,
            "use_momentum": False,
            "momentum_periods": [6, 12]
        },
        "train_params": {
            "seed": 6657,
            "print_every": 10,
            "batch_size": 32,
            "patience": 32,
            "max_checkpoints": 5,
            "min_delta": 0.0,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 6e-4,
                "initial_lr": 1e-9,
                "warmup_steps": 500_000,
                "decay_step_multiplier": 0.7,
            },
            "weight_decay": 0.45,
            "prefix": "mpvMS0003d",
        }
    },


    "vMS0004a": {
        "model_params": {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 512,
            "dropout": 0.01,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 34,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 10],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4, 10],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12],
            "reverse_tickers": True
        },
        "train_params": {
            "seed": 6123,
            "print_every": 10,
            "batch_size": 64,
            "patience": 8,
            "max_checkpoints": 5,
            "min_delta": 0.0,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 6e-4,
                "initial_lr": 1e-9,
                "warmup_steps": 500_000,
                "min_epochs_before_stopping": 30,
                "decay_step_multiplier": 0.7,
            },
            "weight_decay": 0.15,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 13,  # Number of training tickers to use in each subset
            "train_overlap": 3,       # Number of training tickers to overlap between subsets
            "prefix": "mpvMS0004a",
            "architecture_version": "v3",
            "run_id": "8"
        }
    },

    "vMS0004b": {
        "model_params": {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 512,
            "dropout": 0.01,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 34,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 10],
            "sma_periods": [20],
            "target_periods": [1, 2, 3, 4, 10],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12],
            "reverse_tickers": True
        },
        "train_params": {
            "seed": 6123,
            "epochs": 1000,
            "print_every": 10,
            "batch_size": 64,
            "patience": 8,
            "max_checkpoints": 5,
            "min_delta": 0.0,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 1e-3,
                "max_lr": 1e-2,
                "min_lr": 1e-6,
                "warmup_epochs": 10,
                "cycle_params": {
                    "cycle_length": 20,     # 20 epochs per cycle
                    "decay_factor": 0.85,   # Decay peaks by 15% each cycle
                    "cycles": 10            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.15,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 13,  # Number of training tickers to use in each subset
            "train_overlap": 3,       # Number of training tickers to overlap between subsets
            "prefix": "mpvMS0004b",
            "architecture_version": "v3",
            "run_id": "0"
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 1e-3,
                "warmup_steps": 1000,
            },
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 3e-5,
                "warmup_steps": 1000,
            },
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 1e-3,
                "warmup_steps": 3000,
            },
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 1e-4,
                "warmup_steps": 1000,
            },
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 7e-5,
                "warmup_steps": 1000,
                "decay_step_multiplier": None,
            },
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 2e-5,
                "warmup_steps": 1000,
                "decay_step_multiplier": None,
            },
            "batch_size": 128,
            "patience": 32,
            "max_checkpoints": 5,
            "validation_subset_size": 4,
            "validation_overlap": 2,
            "train_subset_size": 20,  # Number of training tickers to use in each subset
            "train_overlap": 8,       # Number of training tickers to overlap between subsets
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 3e-5,
                "warmup_steps": 2000,
                "decay_step_multiplier": 10,
            },
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 3e-3,
                "warmup_steps": 5000,
                "decay_step_multiplier": 5,
            },
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 4e-4,
                "warmup_steps": 4000,
                "decay_step_multiplier": 8,
            },
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 1e-4,
                "warmup_steps": 4000,
                "decay_step_multiplier": 8,
            },
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
            "scheduler": {
                "type": "warmup_decay",
                "base_lr": 3e-4,
                "warmup_steps": 4000,
                "decay_step_multiplier": 15,
            },
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
    },

    "vMP001a": {
        "model_params": {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 512,
            "dropout": 0.02,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 50,  # try much longer and shorter sequence lengths
            "return_periods": [1, 4, 8, 16, 32],
            "sma_periods": [5, 20, 50],
            "target_periods": [1, 4, 8, 16, 32],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12, 24],
            "reverse_tickers": True
        },
        "train_params": {
            "seed": 6123,
            "epochs": 1000,
            "print_every": 10,
            "batch_size": 64,
            "patience": 8,
            "max_checkpoints": 5,
            "min_delta": 0.0,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 1e-3,
                "max_lr": 1e-2,
                "min_lr": 1e-6,
                "warmup_epochs": 10,
                "cycle_params": {
                    "cycle_length": 20,     # 20 epochs per cycle
                    "decay_factor": 0.95,   # Decay peaks by 15% each cycle
                    "cycles": 20            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 15,  # Number of training tickers to use in each subset
            "train_overlap": 3,       # Number of training tickers to overlap between subsets
            "prefix": "vMP001a",
            "architecture_version": "v3",
            "run_id": "0"
        }
    },

    "vMP002a": {
        "model_params": {
            "d_model": 320,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 24,
            "d_ff": 640,
            "dropout": 0.03,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 50,  # try much longer and shorter sequence lengths
            "return_periods": [1, 4, 8, 13, 26, 36],
            "sma_periods": [5, 17, 36, 50],
            "target_periods": [1, 4, 8, 13, 26, 36],
            "use_volatility": False,
            "use_momentum": False,
            "momentum_periods": [6, 12, 16, 28, 34],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 1000,
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 5,
            "min_delta": 0.0,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 1e-3,
                "max_lr": 1e-2,
                "min_lr": 1e-6,
                "warmup_epochs": 15, # 10
                "cycle_params": {
                    "cycle_length": 30,     # 20,     # 20 epochs per cycle
                    "decay_factor": 0.85,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 60            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 4,
            "validation_overlap": 2,
            "train_subset_size": 12, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 6, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP002a",
            "architecture_version": "v3",
            "run_id": "1"
        }
    },


    "vMP003a": {
        "model_params": {
            "d_model": 360,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 720,
            "dropout": 0.35,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 48,  # try much longer and shorter sequence lengths
            "return_periods": [1, 4, 8, 13, 26, 36],
            "sma_periods": [5, 17, 36, 50],
            "target_periods": [1, 4, 8, 13, 26, 36],
            "use_volatility": False,
            "use_momentum": False,
            "momentum_periods": [6, 12, 16, 28, 34],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 14, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP003a",
            "architecture_version": "v3",
            "run_id": "1"
        }
    },

    "vMP003b": {
        "model_params": {
            "d_model": 360,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 720,
            "dropout": 0.35,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 48,  # try much longer and shorter sequence lengths
            "return_periods": [1, 4, 8, 13, 26, 36],
            "sma_periods": [5, 17, 36, 50],
            "target_periods": [1, 4, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12, 16, 28, 34],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 14, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP003b",
            "architecture_version": "v3",
            "run_id": "1"
        }
    },

    "vMP003c": {
        "model_params": {
            "d_model": 392,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 784,
            "dropout": 0.35,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 48,  # try much longer and shorter sequence lengths
            "return_periods": [1, 4, 8, 13, 26, 36],
            "sma_periods": [5, 17, 36, 50],
            "target_periods": [1, 4, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12, 16, 28, 34],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 12,
                "cycle_params": {
                    "cycle_length": 24,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 7, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 2, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP003c",
            "architecture_version": "v3",
            "run_id": "0"
        }
    },

    "vMP003d": {
        "model_params": {
            "d_model": 424,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 848,
            "dropout": 0.47,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 56,  # try much longer and shorter sequence lengths
            "return_periods": [1, 4, 8, 13, 26, 36],
            "sma_periods": [5, 17, 36, 50],
            "target_periods": [1, 4, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12, 16, 28, 34],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 16, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP003d",
            "architecture_version": "v3",
            "run_id": "3"
        }
    },

    "vMP003e": {
        "model_params": {
            "d_model": 424,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 848,
            "dropout": 0.47,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": True,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 56,  # try much longer and shorter sequence lengths
            "return_periods": [1, 4, 8, 13, 26, 36],
            "sma_periods": [5, 17, 36, 50],
            "target_periods": [1, 4, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12, 16, 28, 34],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 16, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP003e",
            "architecture_version": "v3",
            "run_id": "0"
        }
    },

    "vMP003h": {
        "model_params": {
            "d_model": 424,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 848,
            "dropout": 0.47,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "use_hope_pos": True,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 56,  # try much longer and shorter sequence lengths
            "return_periods": [1, 4, 8, 13, 26, 36],
            "sma_periods": [5, 17, 36, 50],
            "target_periods": [1, 4, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12, 16, 28, 34],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 16, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP003h",
            "architecture_version": "v3",
            "run_id": "0"
        }
    },

    "vMP003hcat": {
        "model_params": {
            "d_model": 424,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 848,
            "dropout": 0.47,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "use_hope_pos": True,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 56,  # try much longer and shorter sequence lengths
            "return_periods": [1, 4, 8, 13, 26, 36],
            "sma_periods": [5, 17, 36, 50],
            "target_periods": [1, 4, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12, 16, 28, 34],
            'price_change_bins': {
                'n_bins': 10,
                'min_val': -0.1,  # Optional: -10% return
                'max_val': 0.1    # Optional: +10% return
            },
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12, 16, 28, 34],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 16, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP003hcat",
            "architecture_version": "v3",
            "run_id": "0"
        }
    },

    "vMP003hcatout": {
        "model_params": {
            "d_model": 424,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 848,
            "dropout": 0.47,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "use_hope_pos": True,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 56,  # try much longer and shorter sequence lengths
            "return_periods": [1, 4, 8, 13, 26, 36],
            "sma_periods": [5, 17, 36, 50],
            "target_periods": [1, 4, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12, 16, 28, 34],
            'price_change_bins': {
                'n_bins': 10,
                'min_val': -0.1,  # Optional: -10% return
                'max_val': 0.1    # Optional: +10% return
            },
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12, 16, 28, 34],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 16, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP003hcatout",
            "architecture_version": "v3",
            "run_id": "0"
        }
    },


    "vMP004a": {
        "model_params": {
            "d_model": 424,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 848,
            "dropout": 0.47,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 64,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 8, 13, 26, 36],
            "sma_periods": [5, 17, 36, 50],
            "target_periods": [1, 2, 3, 4, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [3, 6, 9, 12, 16, 28, 34],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 2,
            "validation_overlap": 1,
            "train_subset_size": 14, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP004a",
            "architecture_version": "v3",
            "run_id": "0"
        }
    },

    "vMP005a": {
        "model_params": {
            "d_model": 456,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 912,
            "dropout": 0.49,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 64,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 8, 13, 26, 36],
            "sma_periods": [3, 7, 12, 19, 25, 36, 50],
            "target_periods": [1, 2, 3, 4, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [3, 6, 9, 12, 16, 28, 34],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 2,
            "validation_overlap": 1,
            "train_subset_size": 10, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 7, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP005a",
            "architecture_version": "v3",
            "run_id": "2"
        }
    },

    "vMP006a": {
        "model_params": {
            "d_model": 384,  # try much longer and shorter sequence lengths,
            "n_heads": 16,
            "n_layers": 24,
            "d_ff": 912,
            "dropout": 0.49,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 32,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 4, 8, 16, 32],
            "sma_periods": [3, 7, 17, 25, 36, 50],
            "target_periods": [1, 2, 4, 8, 16, 32],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [4, 9, 15, 21, 37, 46],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 2,
            "validation_overlap": 1,
            "train_subset_size": 15, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 3, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP006a",
            "architecture_version": "v3",
            "run_id": "0"
        }
    },

    "vMP007a": {
        "model_params": {
            "d_model": 384,  # try much longer and shorter sequence lengths,
            "n_heads": 16,
            "n_layers": 24,
            "d_ff": 912,
            "dropout": 0.11,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 32,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "sma_periods": [3, 7, 17, 25, 36, 46],
            "target_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [4, 9, 15, 21, 37, 46],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 2,
            "validation_overlap": 1,
            "train_subset_size": 15, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 3, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP007a",
            "architecture_version": "v3",
            "run_id": "0"
        }
    },

    "vMP008a": {
        "model_params": {
            "d_model": 512,  # try much longer and shorter sequence lengths,
            "n_heads": 16,
            "n_layers": 24,
            "d_ff": 1024,
            "dropout": 0.16,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 32,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "sma_periods": [3, 7, 17, 25, 36, 46],
            "target_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [4, 9, 15, 21, 37, 46],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 2,
            "validation_overlap": 1,
            "train_subset_size": 15, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 3, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP008a",
            "architecture_version": "v3",
            "run_id": "0"
        }
    },

    "vMP009a": {
        "model_params": {
            "d_model": 512,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 1024,
            "dropout": 0.47,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 56,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "sma_periods": [3, 7, 17, 25, 36, 46],
            "target_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [4, 9, 15, 21, 37, 46],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 16, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP009a",
            "architecture_version": "v3",
            "run_id": "0"
        }
    },


    "vMP009h": {
        "model_params": {
            "d_model": 512,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 1024,
            "dropout": 0.47,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 56,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "sma_periods": [3, 7, 17, 25, 36, 46],
            "target_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [4, 9, 15, 21, 37, 46],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },
            "min_epochs_per_subset": 4,
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 16, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP009h",
            "architecture_version": "v4",
            "run_id": "0"
        },
        "augmentation_params": {  # New section
            "enabled": True,
            "t_dist_df": 6,       # Degrees of freedom for Student-t distribution
            "scale_factor": 0.05,  # Fraction of std dev to use for noise
            "subset_fraction": 0.32  # Fraction of subset to augment
        }
    },

    "vMP009h2": {
        "model_params": {
            "d_model": 512,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 16,
            "d_ff": 1024,
            "dropout": 0.47,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 56,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "sma_periods": [3, 7, 17, 25, 36, 46],
            "target_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [4, 9, 15, 21, 37, 46],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 24, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 6, # 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP009h2",
            "architecture_version": "v4",
            "run_id": "0"
        }
    },

    "vMP0010h": {
        "model_params": {
            "d_model": 1024,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 24,
            "d_ff": 2048,
            "dropout": 0.16,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 87,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "sma_periods": [3, 7, 17, 25, 36, 46, 64],
            "target_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [4, 9, 15, 21, 29, 41, 56],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 23000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 8e-4, #7e-4,
                "min_lr": 1e-6, #7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 3000            # Run for 10 cycles then maintain min_lr
                }
            },
            "min_epochs_per_subset": 4,
            "weight_decay": 0.13,
            "validation_subset_size": 5,
            "validation_overlap": 3,
            "train_subset_size": 17, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 7, # 4, # 5,       # Number of training tickers to overlap between subsets
            "rewind_quantile_divisions": 8,  # or 4 for quartiles, etc.
            "rewind_min_extra_epochs": 7,     # min epochs needed to rewind to inferior checkpoint
            "prefix": "vMP0010h",
            "architecture_version": "v4",
            "run_id": "0"
        },
        "augmentation_params": {  # New section
            "enabled": True,
            "t_dist_df": 6,       # Degrees of freedom for Student-t distribution
            "scale_factor": 0.09, # 0.08,  # Fraction of std dev to use for noise
            "subset_fraction": 0.26  # 0.27 # Fraction of subset to augment
        }
    },

    "vMP0011h": {
        "model_params": {
            "d_model": 1024,  # try much longer and shorter sequence lengths,
            "n_heads": 16,
            "n_layers": 32,
            "d_ff": 2048,
            "dropout": 0.16,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 98,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "sma_periods": [3, 7, 17, 25, 36, 46],
            "target_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [4, 9, 15, 21, 37, 46],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 23000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },
            "min_epochs_per_subset": 4,
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 16, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP0011h",
            "architecture_version": "v4",
            "run_id": "0"
        },
        "augmentation_params": {  # New section
            "enabled": True,
            "t_dist_df": 6,       # Degrees of freedom for Student-t distribution
            "scale_factor": 0.08,  # Fraction of std dev to use for noise
            "subset_fraction": 0.28  # Fraction of subset to augment
        }
    },

    "vMP0012h_R": {
        "model_params": {
            "d_model": 1024,  # try much longer and shorter sequence lengths,
            "n_heads": 8,
            "n_layers": 32,
            "d_ff": 2048,
            "dropout": 0.16,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 87,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "sma_periods": [3, 7, 17, 25, 36, 46, 64],
            "target_periods": [1, 2, 3, 4, 5, 8, 13, 26, 36],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [4, 9, 15, 21, 29, 41, 56],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 23000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 8e-4, #7e-4,
                "min_lr": 1e-6, #7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 3000            # Run for 10 cycles then maintain min_lr
                }
            },
            "min_epochs_per_subset": 4,
            "weight_decay": 0.13,
            "validation_subset_size": 5,
            "validation_overlap": 3,
            "train_subset_size": 17, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 7, # 4, # 5,       # Number of training tickers to overlap between subsets
            "rewind_quantile_divisions": 8,  # or 4 for quartiles, etc.
            "rewind_min_extra_epochs": 7,     # min epochs needed to rewind to inferior checkpoint
            "prefix": "vMP0012h_R",
            "architecture_version": "v4",
            "run_id": "0"
        },
        "augmentation_params": {  # New section
            "enabled": True,
            "t_dist_df": 6,       # Degrees of freedom for Student-t distribution
            "scale_factor": 0.09, # 0.08,  # Fraction of std dev to use for noise
            "subset_fraction": 0.29  # 0.27 # Fraction of subset to augment
        }
    },


    "vMLX01a": {
        "model_params": {
            "d_model": 64,  # try much longer and shorter sequence lengths,
            "n_heads": 4,
            "n_layers": 3,
            "d_ff": 256,
            "dropout": 0.1,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 24,  # try much longer and shorter sequence lengths
            "return_periods": [1, 2, 3],
            "sma_periods": [3, 7, 17],
            "target_periods": [1, 2, 3],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [4, 9],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 16, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMLX01a",
            "architecture_version": "v4mlx",
            "run_id": "0"
        }
    },

    "vMP000hcat_in_test": {
        "model_params": {
            "d_model": 32,  # try much longer and shorter sequence lengths,
            "n_heads": 4,
            "n_layers": 8,
            "d_ff": 64,
            "dropout": 0.01,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "use_relative_pos": False,
            "use_hope_pos": True,
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 30,  # try much longer and shorter sequence lengths
            "return_periods": [1, 4, 8],
            "sma_periods": [5, 17],
            "target_periods": [1, 4, 8],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12],
            'price_change_bins': {
                'n_bins': 10,
                'min_val': -0.1,  # Optional: -10% return
                'max_val': 0.1    # Optional: +10% return
            },
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12, 16, 28, 34],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 3000, # 1000
            "print_every": 10,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 16, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "vMP000hcat_in_test",
            "architecture_version": "v3",
            "run_id": "0"
        }
    },

    "simple0": {
        "model_params": {
            "d_model": 32,  # try much longer and shorter sequence lengths,
            "n_heads": 4,
            "n_layers": 8,
            "d_ff": 64,
            "dropout": 0.01,
            "use_multi_scale": False,  # next try True with [1, 2, 4] or something similar
            "temporal_scales": [1, 4, 6]  # try longer temporal scales
        },
        "data_params": {
            "sequence_length": 30,  # try much longer and shorter sequence lengths
            "return_periods": [1, 4, 8],
            "sma_periods": [5, 17],
            "target_periods": [1, 4, 8],
            "use_volatility": True,
            "use_momentum": True,
            "momentum_periods": [6, 12],
            'price_change_bins': {
                'n_bins': 10,
                'min_val': -0.1,  # Optional: -10% return
                'max_val': 0.1    # Optional: +10% return
            },
            "use_volatility": False,
            "use_momentum": True,
            "momentum_periods": [6, 12, 16],
            "reverse_tickers": False
        },
        "train_params": {
            "seed": 6123,
            "epochs": 100, # 1000
            "print_every": 1,
            "batch_size": 64,
            "patience": 12, # 8
            "max_checkpoints": 7,
            "min_delta": 1e-10,  # Minimum change in loss to be considered an improvement
            "scheduler": {
                "type": "cyclical",
                "base_lr": 7e-5,
                "max_lr": 7e-4,
                "min_lr": 7e-6,
                "warmup_epochs": 6,
                "cycle_params": {
                    "cycle_length": 12,     # 20,     # 20 epochs per cycle - try non-divisile by 10
                    "decay_factor": 0.8,   #0.9,   # Decay peaks by 15% each cycle
                    "cycles": 600            # Run for 10 cycles then maintain min_lr
                }
            },  
            "weight_decay": 0.13,
            "validation_subset_size": 3,
            "validation_overlap": 1,
            "train_subset_size": 16, # 16,  # Number of training tickers to use in each subset
            "train_overlap": 4, # 5,       # Number of training tickers to overlap between subsets
            "prefix": "simple0",
            "architecture_version": "v3",
            "run_id": "0",
            "wandb_project": "simple0"
        }
    },
}
