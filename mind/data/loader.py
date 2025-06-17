"""
Data loader for calcium imaging data with proper binary classification.
"""
import os
import numpy as np
import pandas as pd
import h5py
import scipy.io
import hdf5storage
from typing import Dict, Tuple, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


def load_calcium_signals(mat_file_path: str) -> Dict[str, np.ndarray]:
    """
    Load calcium imaging signals from MATLAB file.

    Parameters
    ----------
    mat_file_path : str
        Path to the MATLAB file containing calcium imaging data

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing the three types of signals:
        - 'calcium_signal': Raw fluorescence data
        - 'deltaf_signal': ΔF/F signal
        - 'deconv_signal': Deconvolved signal
    """
    logger.info(f"Loading calcium signals from {mat_file_path}")

    try:
        # First try using scipy.io.loadmat (for older MATLAB files)
        try:
            data = scipy.io.loadmat(mat_file_path)
            calcium_signal = data.get('calciumsignal_wanted', None)
            deltaf_signal = data.get('deltaf_cells_not_excluded', None)
            deconv_signal = data.get('DeconvMat_wanted', None)
        except NotImplementedError:
            # If scipy.io.loadmat fails, try hdf5storage
            data = hdf5storage.loadmat(mat_file_path)
            calcium_signal = data.get('calciumsignal_wanted', None)
            deltaf_signal = data.get('deltaf_cells_not_excluded', None)
            deconv_signal = data.get('DeconvMat_wanted', None)

        # Check that we have at least one signal type
        if calcium_signal is None and deltaf_signal is None and deconv_signal is None:
            logger.error(f"No calcium signals found in {mat_file_path}")
            raise ValueError(f"No calcium signals found in {mat_file_path}")

        # Log shapes of signals
        if calcium_signal is not None:
            logger.info(f"Raw calcium signal shape: {calcium_signal.shape}")
        if deltaf_signal is not None:
            logger.info(f"ΔF/F signal shape: {deltaf_signal.shape}")
        if deconv_signal is not None:
            logger.info(f"Deconvolved signal shape: {deconv_signal.shape}")

        return {
            'calcium_signal': calcium_signal,
            'deltaf_signal': deltaf_signal,
            'deconv_signal': deconv_signal
        }

    except Exception as e:
        logger.error(f"Error loading {mat_file_path}: {e}")
        raise


def load_behavioral_data(xlsx_file_path: str) -> pd.DataFrame:
    """
    Load behavioral data from Excel file.

    Parameters
    ----------
    xlsx_file_path : str
        Path to the Excel file containing behavioral data

    Returns
    -------
    pd.DataFrame
        DataFrame containing behavioral annotations
    """
    logger.info(f"Loading behavioral data from {xlsx_file_path}")

    try:
        # Load the Excel file
        behavior_data = pd.read_excel(xlsx_file_path)

        # Check if the expected columns exist
        expected_columns = ['Foot (L/R)', 'Frame Start', 'Frame End']
        missing_columns = [col for col in expected_columns if col not in behavior_data.columns]

        if missing_columns:
            logger.error(f"Missing required columns in {xlsx_file_path}: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.info(f"Loaded behavioral data with {len(behavior_data)} events")
        return behavior_data

    except Exception as e:
        logger.error(f"Error loading {xlsx_file_path}: {e}")
        raise


def match_behavior_to_frames(behavior_data: pd.DataFrame, num_frames: int,
                             binary_classification: bool = True) -> np.ndarray:
    """
    Create frame-by-frame behavior labels from behavioral events.
    """
    logger.info(f"Creating frame-by-frame behavior labels for {num_frames} frames")
    logger.info(f"Binary classification mode: {binary_classification}")

    # Initialize array of zeros (no footstep)
    frame_labels = np.zeros(num_frames, dtype=np.int32)

    try:
        # Map footstep events to frames
        for _, row in behavior_data.iterrows():
            # Get the foot side, start frame and end frame using the actual column names
            foot = str(row['Foot (L/R)']).lower()
            start_frame = int(row['Frame Start'])
            end_frame = int(row['Frame End'])

            # Ensure frames are within the valid range
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames - 1))

            # Determine the label based on the foot
            if 'right' in foot or 'r' == foot or 'contra' in foot:
                label = 1  # Contralateral (right) footstep
            elif 'left' in foot or 'l' == foot or 'ipsi' in foot:
                if binary_classification:
                    # In binary classification, we ignore ipsilateral footsteps
                    continue
                else:
                    label = 2  # Ipsilateral (left) footstep
            else:
                logger.warning(f"Unrecognized foot value: {foot}, skipping event")
                continue

            # Assign labels to frames
            frame_labels[start_frame:end_frame + 1] = label

        # Log stats about the labels
        unique_labels, counts = np.unique(frame_labels, return_counts=True)
        label_stats = {f"Label {label}": count for label, count in zip(unique_labels, counts)}
        logger.info(f"Label distribution: {label_stats}")

        # Log percentages
        total_frames = len(frame_labels)
        for label, count in zip(unique_labels, counts):
            percentage = (count / total_frames) * 100
            logger.info(f"Label {label}: {count} frames ({percentage:.2f}%)")

        return frame_labels

    except Exception as e:
        logger.error(f"Error creating behavior labels: {e}")
        raise


def load_and_align_data(mat_file_path: str, xlsx_file_path: str,
                        binary_classification: bool = True) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Load and align calcium imaging data with behavioral labels.

    Parameters
    ----------
    mat_file_path : str
        Path to the MATLAB file containing calcium imaging data
    xlsx_file_path : str
        Path to the Excel file containing behavioral data
    binary_classification : bool, optional
        If True, create binary labels (0 for no footstep, 1 for contralateral/right footstep)

    Returns
    -------
    Tuple[Dict[str, np.ndarray], np.ndarray]
        Tuple containing:
        - Dictionary of calcium signals
        - Array of behavior labels
    """

    # Load calcium signals
    calcium_signals = load_calcium_signals(mat_file_path)

    # Determine the number of frames
    num_frames = None
    for signal_type, signal in calcium_signals.items():
        if signal is not None:
            num_frames = signal.shape[0]
            break

    if num_frames is None:
        raise ValueError("No valid calcium signals found")

    # Load behavioral data
    behavior_data = load_behavioral_data(xlsx_file_path)

    # Match behavior to frames (binary classification by default)
    frame_labels = match_behavior_to_frames(behavior_data, num_frames, binary_classification)

    return calcium_signals, frame_labels


def find_most_active_neurons(calcium_signals: Dict[str, np.ndarray],
                             n_neurons: int = 20,
                             signal_type: str = 'deconv_signal') -> np.ndarray:
    """
    Find the most active neurons based on calcium transient activity.

    Parameters
    ----------
    calcium_signals : Dict[str, np.ndarray]
        Dictionary of calcium signals
    n_neurons : int
        Number of top neurons to return
    signal_type : str
        Type of signal to use for finding active neurons

    Returns
    -------
    np.ndarray
        Indices of the most active neurons
    """
    signal = calcium_signals[signal_type]
    if signal is None:
        # Fallback to other signal types
        for alt_signal in ['deltaf_signal', 'calcium_signal']:
            if calcium_signals[alt_signal] is not None:
                signal = calcium_signals[alt_signal]
                break

    # Calculate activity metrics
    # For deconvolved signals, use sum of transients
    # For other signals, use variance
    if signal_type == 'deconv_signal':
        activity_metric = np.sum(signal > 0, axis=0)  # Count of active frames
    else:
        activity_metric = np.var(signal, axis=0)  # Variance

    # Get indices of top neurons
    top_indices = np.argsort(activity_metric)[::-1][:n_neurons]

    return top_indices


