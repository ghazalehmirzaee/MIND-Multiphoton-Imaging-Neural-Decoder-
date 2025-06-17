"""
Fixed configuration with proper Random Forest class balancing and scientific colors.
"""
import torch

DEFAULT_CONFIG = {
    # Data parameters
    "data": {
        "window_size": 15,
        "step_size": 1,
        "test_size": 0.15,
        "val_size": 0.15,
        "batch_size": 32,
        "num_workers": 4,
        "binary_classification": True,
        "mat_file": "data/raw/SFL13_5_8112021_003_new_modified.mat",
        "xlsx_file": "data/raw/SFL13_5_8112021_003_new.xlsx"
    },

    # Model parameters
    "models": {
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",  # Better for imbalanced data
            "n_jobs": -1,
            "random_state": 42,
            "criterion": "gini",
            "bootstrap": True
        },
        "svm": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
            "random_state": 42,
            "n_components": 0.95,
            "use_pca": True
        },
        "mlp": {
            "hidden_layer_sizes": (64, 128, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "adaptive",
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 15,
            "random_state": 42
        },
        "fcnn": {
            "hidden_dims": [256, 128, 64],
            "output_dim": 2,
            "dropout_rate": 0.4,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 15,
            "random_state": 42
        },
        "cnn": {
            "n_filters": [32, 64, 128],  # Reduced for better performance
            "kernel_size": 5,  # Larger kernel for temporal patterns
            "output_dim": 2,
            "dropout_rate": 0.2,  # Reduced dropout
            "learning_rate": 0.0005,  # Lower learning rate
            "weight_decay": 1e-4,
            "batch_size": 32,
            "num_epochs": 50,
            "patience": 10,
            "random_state": 42
        }
    },

    # Training parameters
    "training": {
        "optimize_hyperparams": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "outputs/results"
    },

    # W&B parameters
    "wandb": {
        "use_wandb": True,
        "project_name": "mind-calcium-imaging",
        "entity": None
    },

    # Visualization parameters with scientific colors
    "visualization": {
        "output_dir": "outputs/figures",
        "dpi": 300,
        "format": "png",
        "signal_colors": {
            "calcium_signal": "#356d9e",  # Scientific blue
            "deltaf_signal": "#4c8b64",  # Scientific green
            "deconv_signal": "#a85858"  # Scientific red
        },
        "signal_gradients": {
            "calcium_signal": ["#f0f4f9", "#c6dcef", "#7fb0d3", "#356d9e"],
            "deltaf_signal": ["#f6f9f4", "#d6ead9", "#9dcaa7", "#4c8b64"],
            "deconv_signal": ["#fdf3f3", "#f0d0d0", "#d49c9c", "#a85858"]
        },
        "signal_display_names": {
            "calcium_signal": "Calcium",
            "deltaf_signal": "ΔF/F",
            "deconv_signal": "Deconvolved"
        }
    },

    # Binary classification parameters
    "classification": {
        "task": "binary",
        "labels": ["No footstep", "Contralateral"],
        "n_classes": 2
    }
}


def get_config():
    """Get default configuration with device check."""
    if not torch.cuda.is_available() and DEFAULT_CONFIG["training"]["device"] == "cuda":
        DEFAULT_CONFIG["training"]["device"] = "cpu"

    return DEFAULT_CONFIG

