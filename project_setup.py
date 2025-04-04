# project_setup.py
import os
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb
from tqdm import tqdm
import random
import torch


# Set random seeds for reproducibility
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


# Call this at the beginning to ensure reproducibility
set_seeds()


# Create project directory structure
def create_project_structure():
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'models/checkpoints',
        'results/figures',
        'results/metrics',
        'notebooks'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("Project directory structure created successfully.")


# Function to load MATLAB data
def load_matlab_data(file_path):
    """Load calcium imaging data from MATLAB files with the three specified signal types."""
    try:
        mat_data = sio.loadmat(file_path)

        # Extract the three specific signal types as specified
        calcium_signal = mat_data['calciumsignal']  # Raw signal
        deltaf_cells_not_excluded = mat_data['deltaf_cells_not_excluded']  # ΔF/F for valid neurons
        deconv_mat_wanted = mat_data['DeconvMat_wanted']  # Deconvolved for valid neurons

        # Extract valid neuron indices for reference
        valid_neurons = mat_data['cells_not_excluded'].flatten() - 1  # Convert to 0-based indexing

        print(f"Loaded data dimensions:")
        print(f"Calcium signal: {calcium_signal.shape}")
        print(f"ΔF/F (valid neurons): {deltaf_cells_not_excluded.shape}")
        print(f"Deconvolved (valid neurons): {deconv_mat_wanted.shape}")
        print(f"Number of valid neurons: {len(valid_neurons)}")

        return {
            'calcium_signal': calcium_signal,
            'deltaf_cells_not_excluded': deltaf_cells_not_excluded,
            'deconv_mat_wanted': deconv_mat_wanted,
            'valid_neurons': valid_neurons
        }
    except Exception as e:
        print(f"Error loading MATLAB file: {e}")
        return None


# Function to load behavioral data from Excel
def load_behavioral_data(file_path):
    """Load behavioral data from Excel file."""
    try:
        print(f"Loading behavioral data from Excel file: {file_path}")
        behavior_df = pd.read_excel(file_path)

        # Check if necessary columns exist
        required_columns = ['Frame Start', 'Frame End', 'Foot (L/R)']
        for col in required_columns:
            if col not in behavior_df.columns:
                print(f"Warning: Required column '{col}' not found in the Excel file.")
                # Try to detect similar columns and rename them
                if 'Frame' in behavior_df.columns and 'Start' in behavior_df.columns:
                    behavior_df['Frame Start'] = behavior_df['Start']
                if 'Frame' in behavior_df.columns and 'End' in behavior_df.columns:
                    behavior_df['Frame End'] = behavior_df['End']
                if 'Foot' in behavior_df.columns:
                    behavior_df['Foot (L/R)'] = behavior_df['Foot']

        # Display data info
        print(f"Successfully loaded behavioral data with {len(behavior_df)} entries")
        print("Columns in behavioral data:", behavior_df.columns.tolist())
        print("Sample data:")
        print(behavior_df.head())

        return behavior_df
    except Exception as e:
        print(f"Error loading behavioral data: {e}")
        return None


# Function to align neural and behavioral data
def align_neural_behavioral_data(neural_data, behavior_df):
    """
    Align neural recording frames with behavioral events.
    Assigns: 0 for no footstep, 1 for RIGHT foot (contralateral), 2 for LEFT foot (ipsilateral).
    """
    # Extract frame information
    frame_starts = behavior_df['Frame Start'].values
    frame_ends = behavior_df['Frame End'].values
    foot_sides = behavior_df['Foot (L/R)'].values

    # Create label arrays
    num_frames = neural_data['calcium_signal'].shape[0]
    labels = np.zeros(num_frames)

    # Create different labels as specified: RIGHT=1, LEFT=2
    for i in range(len(frame_starts)):
        start = int(frame_starts[i])
        end = int(frame_ends[i])

        # Check if frames are within data range
        if start < num_frames and end < num_frames:
            # Right foot (contralateral) = 1, Left foot (ipsilateral) = 2
            label_value = 1 if foot_sides[i] == 'R' else 2
            labels[start:end + 1] = label_value

    # Count instances of each class
    class_counts = np.bincount(labels.astype(int))
    print(f"Label distribution:")
    print(f"No footstep (0): {class_counts[0]} frames")
    if len(class_counts) > 1:
        print(f"Right foot (1): {class_counts[1]} frames")
    if len(class_counts) > 2:
        print(f"Left foot (2): {class_counts[2]} frames")

    # Add labels to the data dictionary
    neural_data['labels'] = labels
    return neural_data


# Preprocess and prepare data for ML
def preprocess_data(data_dict, window_size=10, step_size=2):
    """
    Process neural data with sliding windows and prepare for ML.
    Based on the paper, using window_size=10, step_size=2 for initial models.
    """
    print("Preprocessing data with sliding windows...")

    # Extract data
    calcium_signal = data_dict['calcium_signal']
    deltaf_cells_not_excluded = data_dict['deltaf_cells_not_excluded']
    deconv_mat_wanted = data_dict['deconv_mat_wanted']

    # Check if labels exist
    if 'labels' not in data_dict:
        print("Error: No behavioral labels found. Please provide behavioral data.")
        return None

    labels = data_dict['labels']

    # Get neurons dimensions
    n_frames, n_calcium_neurons = calcium_signal.shape
    _, n_deltaf_neurons = deltaf_cells_not_excluded.shape
    _, n_deconv_neurons = deconv_mat_wanted.shape

    print(f"Number of frames: {n_frames}")
    print(f"Calcium signal neurons: {n_calcium_neurons}")
    print(f"ΔF/F neurons: {n_deltaf_neurons}")
    print(f"Deconvolved neurons: {n_deconv_neurons}")

    # Create sliding windows
    def create_windows(data, labels):
        n_samples = data.shape[0]
        windows = []
        window_labels = []

        # Use tqdm for progress tracking
        for i in tqdm(range(0, n_samples - window_size + 1, step_size), desc="Creating windows"):
            window = data[i:i + window_size, :]
            # Use the label from the last frame in the window as in the paper
            label = labels[i + window_size - 1]

            windows.append(window.flatten())
            window_labels.append(label)

        return np.array(windows), np.array(window_labels)

    # Create windowed data for each signal type
    print("Processing calcium signal...")
    X_calcium, y_calcium = create_windows(calcium_signal, labels)

    print("Processing ΔF/F signal...")
    X_deltaf, y_deltaf = create_windows(deltaf_cells_not_excluded, labels)

    print("Processing deconvolved signal...")
    X_deconv, y_deconv = create_windows(deconv_mat_wanted, labels)

    print(f"Created {X_calcium.shape[0]} windows for each signal type")

    # Normalize data
    print("Normalizing data...")
    scaler_calcium = StandardScaler()
    X_calcium_norm = scaler_calcium.fit_transform(X_calcium)

    scaler_deltaf = StandardScaler()
    X_deltaf_norm = scaler_deltaf.fit_transform(X_deltaf)

    scaler_deconv = StandardScaler()
    X_deconv_norm = scaler_deconv.fit_transform(X_deconv)

    return {
        'X_calcium': X_calcium_norm,
        'y_calcium': y_calcium,
        'X_deltaf': X_deltaf_norm,
        'y_deltaf': y_deltaf,
        'X_deconv': X_deconv_norm,
        'y_deconv': y_deconv,
        'window_size': window_size,
        'n_calcium_neurons': n_calcium_neurons,
        'n_deltaf_neurons': n_deltaf_neurons,
        'n_deconv_neurons': n_deconv_neurons,
        'scalers': {
            'calcium': scaler_calcium,
            'deltaf': scaler_deltaf,
            'deconv': scaler_deconv
        }
    }


# Split data into train/validation/test sets
def split_data(processed_data, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train, validation, and test sets with stratification."""
    result = {}

    # Process each signal type
    for signal_type in ['calcium', 'deltaf', 'deconv']:
        print(f"Splitting {signal_type} data...")
        X = processed_data[f'X_{signal_type}']
        y = processed_data[f'y_{signal_type}']

        # First split: training + validation vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: training vs validation
        # Adjust validation size to get the right proportion from the remaining data
        adjusted_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=adjusted_val_size,
            random_state=random_state, stratify=y_train_val
        )

        # Store splits
        result[f'X_train_{signal_type}'] = X_train
        result[f'X_val_{signal_type}'] = X_val
        result[f'X_test_{signal_type}'] = X_test
        result[f'y_train_{signal_type}'] = y_train
        result[f'y_val_{signal_type}'] = y_val
        result[f'y_test_{signal_type}'] = y_test

        # Print class distribution
        for subset_name, y_subset in [('train', y_train), ('validation', y_val), ('test', y_test)]:
            class_counts = np.bincount(y_subset.astype(int))
            print(f"{subset_name} set class distribution:")
            print(f"  No footstep (0): {class_counts[0]} samples")
            if len(class_counts) > 1:
                print(f"  Right foot (1): {class_counts[1]} samples")
            if len(class_counts) > 2:
                print(f"  Left foot (2): {class_counts[2]} samples")

    return result


# Calculate class weights to handle imbalanced data
def calculate_class_weights(y_train):
    """Calculate class weights inversely proportional to class frequencies."""
    classes = np.unique(y_train)
    class_counts = np.bincount(y_train.astype(int))
    total_samples = len(y_train)

    # Fix: Convert max class to int explicitly to handle float values
    max_class = int(np.max(classes))
    full_class_counts = np.zeros(max_class + 1)

    for cls in classes:
        cls_int = int(cls)  # Convert class to integer explicitly
        if cls_int < len(class_counts):
            full_class_counts[cls_int] = class_counts[cls_int]

    # Calculate weights
    class_weights = {}
    for cls in classes:
        cls_int = int(cls)  # Convert class to integer explicitly
        if full_class_counts[cls_int] > 0:
            class_weights[cls_int] = total_samples / (len(classes) * full_class_counts[cls_int])
        else:
            class_weights[cls_int] = 1.0

    print(f"Calculated class weights: {class_weights}")
    return class_weights


# Initialize W&B
def init_wandb(project_name="calcium-imaging-ml", experiment_name=None):
    """Initialize Weights & Biases for experiment tracking."""
    if experiment_name is None:
        experiment_name = "default_experiment"

    wandb.init(project=project_name, name=experiment_name, entity="mirzaeeghazal")

    # Define metadata for the project
    wandb.config.update({
        "description": "Comparing ML models on different calcium imaging signal types",
        "data_source": "Two-photon calcium imaging during mouse reaching task",
        "signal_types": ["Raw calcium", "ΔF/F", "Deconvolved"]
    })

    print(f"Initialized W&B project: {project_name}, experiment: {experiment_name}")
    return wandb


# Visualize sample data
def visualize_sample_data(data_dict, output_dir='results/figures'):
    """Visualize sample data from each signal type with 200-250 selected neurons."""
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducible visualizations
    np.random.seed(42)

    # Select a subset of neurons (200-250 as mentioned in the outline)
    signal_types = {
        'calcium': ('calcium', 'n_calcium_neurons', 'Raw Calcium Signal'),
        'deltaf': ('deltaf', 'n_deltaf_neurons', 'ΔF/F Signal'),
        'deconv': ('deconv', 'n_deconv_neurons', 'Deconvolved Signal')
    }

    # Create visualizations for each signal type
    for key, (signal_key, n_neurons_key, signal_name) in signal_types.items():
        X = data_dict[f'X_{signal_key}']
        window_size = data_dict['window_size']
        n_total_neurons = data_dict[n_neurons_key]

        # Select 200-250 neurons or all if fewer are available
        n_neurons_to_show = min(250, n_total_neurons)
        n_neurons_to_show = max(200, n_neurons_to_show)
        if n_total_neurons < 200:
            n_neurons_to_show = n_total_neurons

        print(f"Visualizing {n_neurons_to_show} neurons for {signal_name}...")

        # Select random neurons if we have more than we want to display
        if n_total_neurons > n_neurons_to_show:
            selected_neurons = np.random.choice(n_total_neurons, n_neurons_to_show, replace=False)
        else:
            selected_neurons = np.arange(n_total_neurons)

        # Get a random sample
        sample_idx = np.random.randint(0, X.shape[0])
        sample = X[sample_idx].reshape(window_size, -1)

        # Select only the chosen neurons
        sample = sample[:, selected_neurons]

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(sample.T, cmap='viridis', xticklabels=5, yticklabels=50)
        plt.title(f'Sample {signal_name} Activity - {n_neurons_to_show} Neurons')
        plt.xlabel('Time Steps')
        plt.ylabel('Neurons')

        # Save figure
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{signal_key}_sample.png', dpi=300)
        plt.close()

    # Create time series comparison for each signal type
    plt.figure(figsize=(15, 10))

    # Select a few neurons to display
    display_neurons = sorted(np.random.choice(n_neurons_to_show, 3, replace=False))

    for i, neuron_rel_idx in enumerate(display_neurons):
        for j, (signal_key, (signal_name, _, _)) in enumerate(zip(['calcium', 'deltaf', 'deconv'],
                                                                  signal_types.values())):
            X = data_dict[f'X_{signal_key}']
            window_size = data_dict['window_size']

            # Get the same sample for all signals
            sample = X[sample_idx].reshape(window_size, -1)

            # Get absolute neuron index
            neuron_abs_idx = selected_neurons[neuron_rel_idx]

            # Plot neuron activity
            plt.subplot(len(display_neurons), 3, i * 3 + j + 1)
            plt.plot(range(window_size), sample[:, neuron_rel_idx])
            plt.title(f'Neuron #{neuron_abs_idx} - {signal_name}')
            plt.xlabel('Time Step')
            plt.ylabel('Activity')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/signal_comparison.png', dpi=300)
    plt.close()

    print(f"Sample visualizations saved to {output_dir}")
    return


# Save processed data
def save_processed_data(data_dict, file_path):
    """Save processed data to NPZ file."""
    np.savez(file_path, **data_dict)
    print(f"Data saved to {file_path}")


# Main processing function
def process_main(matlab_file_path, behavior_file_path=None, window_size=10, step_size=2, experiment_name=None):
    """Main function to process data and prepare for ML."""
    # Set seed for reproducibility
    set_seeds(42)

    # Create project structure
    create_project_structure()

    # Initialize W&B
    wandb_run = init_wandb(experiment_name=experiment_name)

    # Log parameters
    wandb.config.update({
        "window_size": window_size,
        "step_size": step_size,
        "matlab_file": os.path.basename(matlab_file_path),
        "behavior_file": os.path.basename(behavior_file_path) if behavior_file_path else None
    })

    # Load data
    neural_data = load_matlab_data(matlab_file_path)

    if neural_data is None:
        print("Failed to load neural data. Exiting.")
        wandb.finish()
        return None

    # Load and align behavioral data if provided
    behavior_loaded = False
    if behavior_file_path:
        behavior_df = load_behavioral_data(behavior_file_path)
        if behavior_df is not None:
            neural_data = align_neural_behavioral_data(neural_data, behavior_df)
            behavior_loaded = True

    if not behavior_loaded:
        print("No behavioral data loaded. Exiting.")
        wandb.finish()
        return None

    # Preprocess data
    processed_data = preprocess_data(neural_data, window_size, step_size)

    if processed_data is None:
        print("Failed to preprocess data. Exiting.")
        wandb.finish()
        return None

    # Split data
    split_data_dict = split_data(processed_data)

    # Calculate class weights for each signal type
    class_weights = {}
    for signal_type in ['calcium', 'deltaf', 'deconv']:
        y_train = split_data_dict[f'y_train_{signal_type}']
        class_weights[signal_type] = calculate_class_weights(y_train)

    # Add class weights to data
    split_data_dict['class_weights'] = class_weights

    # Combine processed and split data
    final_data = {**processed_data, **split_data_dict}

    # Visualize sample data
    visualize_sample_data(final_data)

    # Save processed data
    save_processed_data(final_data, 'data/processed/processed_calcium_data.npz')

    # Log sample figures to W&B
    wandb.log({
        "calcium_sample": wandb.Image('results/figures/calcium_sample.png'),
        "deltaf_sample": wandb.Image('results/figures/deltaf_sample.png'),
        "deconv_sample": wandb.Image('results/figures/deconv_sample.png'),
        "signal_comparison": wandb.Image('results/figures/signal_comparison.png')
    })

    # Finish W&B run
    wandb.finish()

    return final_data


# If run as a script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process calcium imaging data")
    parser.add_argument('--matlab_file', type=str, required=True,
                        help='Path to MATLAB file containing calcium imaging data')
    parser.add_argument('--behavior_file', type=str, default=None,
                        help='Path to behavior data Excel file')
    parser.add_argument('--window_size', type=int, default=10,
                        help='Window size for sliding window')
    parser.add_argument('--step_size', type=int, default=5,
                        help='Step size for sliding window')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the W&B experiment')

    args = parser.parse_args()

    process_main(args.matlab_file, args.behavior_file, args.window_size, args.step_size, args.experiment_name)

