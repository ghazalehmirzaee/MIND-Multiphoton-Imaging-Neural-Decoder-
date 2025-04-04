# main.py
import os
import argparse
import numpy as np
import wandb
import random
import torch
from project_setup import process_main, set_seeds
from classical_ml_models import ClassicalMLModels
from deep_learning_models import DeepLearningModels  # Changed import to PyTorch version
from visualize_results import ResultVisualizer


# Set seeds at the very beginning
def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# Initialize seeds
set_seeds(42)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calcium Imaging ML Pipeline (PyTorch)")

    # Data parameters
    parser.add_argument('--matlab_file', type=str, required=True,
                        help='Path to MATLAB file containing calcium imaging data')
    parser.add_argument('--behavior_file', type=str, default=None,
                        help='Path to behavior data Excel file (optional)')

    # Processing parameters
    parser.add_argument('--window_size', type=int, default=10,
                        help='Window size for sliding window')
    parser.add_argument('--step_size', type=int, default=5,
                        help='Step size for sliding window')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for deep learning models')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for deep learning models')

    # Execution flags
    parser.add_argument('--skip_processing', action='store_true',
                        help='Skip data processing step')
    parser.add_argument('--skip_classical', action='store_true',
                        help='Skip classical ML models')
    parser.add_argument('--skip_deep_learning', action='store_true',
                        help='Skip deep learning models')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip results visualization')

    # W&B parameters
    parser.add_argument('--project_name', type=str, default='MIND',
                        help='W&B project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the W&B experiment')

    # PyTorch specific parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training (cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()

    # Set device for PyTorch
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Process data
    if not args.skip_processing:
        print("=== Processing Data ===")
        data_dict = process_main(
            args.matlab_file, args.behavior_file,
            args.window_size, args.step_size,
            args.experiment_name
        )

        if data_dict is None:
            print("Data processing failed. Exiting.")
            return
    else:
        print("=== Loading Processed Data ===")
        try:
            data_dict = np.load('data/processed/processed_calcium_data.npz', allow_pickle=True)
            data_dict = dict(data_dict.items())  # Convert to dictionary
        except Exception as e:
            print(f"Error loading processed data: {e}")
            print("Please run data processing first.")
            return

    # Train classical ML models
    if not args.skip_classical:
        print("=== Training Classical ML Models ===")
        wandb_run = wandb.init(
            project=args.project_name,
            name=f"{args.experiment_name}_classical_ml" if args.experiment_name else "classical_ml_models",
            entity="mirzaeeghazal"
        )
        ml_models = ClassicalMLModels(data_dict, wandb_run)
        ml_models.train_and_evaluate_all()
        ml_models.test_best_models()
        ml_models.feature_importance()
        ml_models.save_results()
        ml_models.save_models()
        wandb_run.finish()

    # Train deep learning models
    if not args.skip_deep_learning:
        print("=== Training Deep Learning Models (PyTorch) ===")
        wandb_run = wandb.init(
            project=args.project_name,
            name=f"{args.experiment_name}_deep_learning_pytorch" if args.experiment_name else "deep_learning_models_pytorch",
            entity="mirzaeeghazal"
        )
        # Initialize DeepLearningModels with PyTorch-specific parameters
        dl_models = DeepLearningModels(data_dict, wandb_run)
        dl_models.train_and_evaluate_all(args.epochs, args.batch_size)
        dl_models.test_best_models()
        dl_models.visualize_activations()
        dl_models.save_results()
        dl_models.save_models()
        wandb_run.finish()

    # Visualize results
    if not args.skip_visualization:
        print("=== Visualizing Results ===")
        wandb_run = wandb.init(
            project=args.project_name,
            name=f"{args.experiment_name}_visualization" if args.experiment_name else "results_visualization",
            entity="mirzaeeghazal"
        )
        visualizer = ResultVisualizer(
            'results/metrics/classical_ml_results.json',
            'results/metrics/deep_learning_results.json',
            wandb_run
        )
        summary = visualizer.visualize_all()
        print("=== Summary Report ===")
        print(summary)
        wandb_run.finish()

    print("=== Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()

