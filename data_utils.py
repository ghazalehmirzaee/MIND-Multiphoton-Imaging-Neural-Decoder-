# data_utils.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CalciumImagingDataset(Dataset):
    """PyTorch Dataset for calcium imaging data."""

    def __init__(self, X, y):
        """
        Initialize dataset.

        Args:
            X: Input features (numpy array)
            y: Labels (numpy array)
        """
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        """Return the number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Get a sample at the specified index."""
        return self.X[idx], self.y[idx]


def create_dataloaders(data_dict, signal_type, batch_size=32, num_workers=4):
    """
    Create PyTorch DataLoaders for a specific signal type.

    Args:
        data_dict: Dictionary containing the processed data
        signal_type: Type of signal to use (calcium, deltaf, or deconv)
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader

    Returns:
        Dictionary containing train, validation, and test DataLoaders
    """
    # Extract data for the specified signal type
    X_train = data_dict[f'X_train_{signal_type}']
    y_train = data_dict[f'y_train_{signal_type}']
    X_val = data_dict[f'X_val_{signal_type}']
    y_val = data_dict[f'y_val_{signal_type}']
    X_test = data_dict[f'X_test_{signal_type}']
    y_test = data_dict[f'y_test_{signal_type}']

    # Create datasets
    train_dataset = CalciumImagingDataset(X_train, y_train)
    val_dataset = CalciumImagingDataset(X_val, y_val)
    test_dataset = CalciumImagingDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Get input dimensions for model initialization
    window_size = data_dict['window_size']
    n_neurons = data_dict[f'n_{signal_type}_neurons']

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'input_dim': X_train.shape[1],  # Flattened dimension for MLP
        'window_size': window_size,
        'n_neurons': n_neurons,
        'n_classes': len(np.unique(y_train))
    }

