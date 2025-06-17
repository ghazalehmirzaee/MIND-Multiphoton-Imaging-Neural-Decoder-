"""
Data processing utilities for calcium imaging data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional, Any, Union
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class SlidingWindowDataset(Dataset):
    """
    Dataset that creates sliding windows of neural activity for binary classification.

    This dataset takes calcium imaging signals and creates windows of activity
    for each frame, along with the corresponding behavior label.
    """

    def __init__(self,
                 signal: np.ndarray,
                 labels: np.ndarray,
                 window_size: int = 15,
                 step_size: int = 1,
                 remove_zero_labels: bool = False):
        """
        Initialize a sliding window dataset.
        """
        self.signal = signal
        self.labels = labels
        self.window_size = window_size
        self.step_size = step_size

        # Calculate valid indices for windows
        self.valid_indices = []

        # Create sliding windows
        n_frames = signal.shape[0]

        for i in range(0, n_frames - window_size + 1, step_size):
            # Get the label for this window (use the label of the last frame in the window)
            window_label = labels[i + window_size - 1]

            # If we're removing windows with zero labels, check the label
            if remove_zero_labels and window_label == 0:
                continue

            self.valid_indices.append(i)

        logger.info(f"Created dataset with {len(self.valid_indices)} windows")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the starting index for this window
        start_idx = self.valid_indices[idx]

        # Extract the window
        window = self.signal[start_idx:start_idx + self.window_size, :]

        # Get the label for this window (use the label of the last frame in the window)
        label = self.labels[start_idx + self.window_size - 1]

        # Convert to tensors
        window_tensor = torch.FloatTensor(window)
        label_tensor = torch.LongTensor([label])

        return window_tensor, label_tensor.squeeze()


def create_datasets(calcium_signals: Dict[str, np.ndarray],
                    frame_labels: np.ndarray,
                    window_size: int = 15,
                    step_size: int = 1,
                    test_size: float = 0.15,
                    val_size: float = 0.15,
                    random_state: int = 42) -> Dict[str, Dict[str, SlidingWindowDataset]]:
    """
    Create train, validation, and test datasets for each signal type.
    """
    logger.info("Creating datasets from calcium signals")

    # Create dataset dictionary
    datasets = {}

    # Process each signal type
    for signal_name, signal in calcium_signals.items():
        if signal is None:
            logger.warning(f"Skipping {signal_name} because it is None")
            continue

        logger.info(f"Processing {signal_name}")

        # Create windows for the entire dataset
        full_dataset = SlidingWindowDataset(signal, frame_labels,
                                            window_size=window_size,
                                            step_size=step_size)

        # Create indices for train/val/test split
        indices = np.arange(len(full_dataset))

        # Split into train+val and test
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=None)

        # Calculate actual validation size as a fraction of train+val
        actual_val_size = val_size / (1 - test_size)

        # Split train+val into train and val
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=actual_val_size, random_state=random_state, stratify=None)

        # Create subsets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

        # Store datasets
        datasets[signal_name] = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

        # Log split sizes
        logger.info(
            f"{signal_name} split sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    return datasets


def create_data_loaders(datasets: Dict[str, Dict[str, torch.utils.data.Dataset]],
                        batch_size: int = 32,
                        num_workers: int = 4) -> Dict[str, Dict[str, torch.utils.data.DataLoader]]:
    """
    Create DataLoader objects for each dataset.
    """
    logger.info(f"Creating DataLoaders with batch_size={batch_size}, num_workers={num_workers}")

    dataloaders = {}

    for signal_name, signal_datasets in datasets.items():
        dataloaders[signal_name] = {}

        for split_name, dataset in signal_datasets.items():
            # Use different batch sizes for different splits if needed
            current_batch_size = batch_size

            # Create DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=current_batch_size,
                shuffle=(split_name == 'train'),  # Only shuffle training data
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True
            )

            dataloaders[signal_name][split_name] = dataloader

            logger.info(f"Created DataLoader for {signal_name}/{split_name} with {len(dataloader)} batches")

    return dataloaders


def get_dataset_dimensions(datasets: Dict[str, Dict[str, torch.utils.data.Dataset]]) -> Dict[str, Tuple[int, int]]:
    """
    Get the dimensions (window_size, n_features) for each dataset.

    Parameters
    ----------
    datasets : Dict[str, Dict[str, torch.utils.data.Dataset]]
        Dictionary of datasets for each signal type and split

    Returns
    -------
    Dict[str, Tuple[int, int]]
        Dictionary of dimensions (window_size, n_features) for each signal type
    """
    dimensions = {}

    for signal_name, signal_datasets in datasets.items():
        # Get the first dataset (train)
        dataset = signal_datasets['train']

        # Get the first sample
        X, _ = dataset[0]

        # Get dimensions
        if isinstance(X, torch.Tensor):
            dimensions[signal_name] = (X.shape[0], X.shape[1])
        else:
            # Handle case where X is not a tensor (e.g., for classical models)
            dimensions[signal_name] = (X.shape[0], X.shape[1])

    return dimensions

