"""
Neuron activity vs. model importance comparison module.

This module creates visualizations comparing the overlap between neurons deemed important
by ML models and neurons with the highest activity in calcium signals.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from pathlib import Path
import os
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Set
import seaborn as sns

# Import from existing visualization components
from mind.visualization.config import (
    SIGNAL_COLORS,
    SIGNAL_DISPLAY_NAMES,
    MODEL_DISPLAY_NAMES,
    set_publication_style,
    FIGURE_SIZES
)

# Import utilities from the neuron bubble chart module
from mind.visualization.neuron_importance import (
    load_data,
    extract_neuron_importance,
    approximate_neuron_positions
)

logger = logging.getLogger(__name__)


def calculate_neuron_activity(calcium_signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate total activity for each neuron across the time series.

    For raw and ΔF/F signals, calculates the sum of activity.
    For deconvolved signal, counts the total estimated spikes.

    Parameters
    ----------
    calcium_signals : Dict[str, np.ndarray]
        Dictionary containing all three signal types

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping signal type to array of total activity per neuron
    """
    activity_sums = {}

    for signal_type, signal in calcium_signals.items():
        if signal is None:
            continue

        # Get dimensions
        n_frames, n_neurons = signal.shape

        if signal_type == 'deconv_signal':
            # For deconvolved signal, count spike events (activity > threshold)
            # This is effectively counting the total number of inferred spikes
            spike_threshold = 0.01  # Adjust based on your data
            activity_sums[signal_type] = np.sum(signal > spike_threshold, axis=0)
            logger.info(f"Calculated spike counts for {n_neurons} neurons in deconvolved signal")

        elif signal_type == 'deltaf_signal':
            # For ΔF/F, sum positive changes (calcium influx events)
            # This focuses on calcium increases rather than decreases
            positive_signal = np.maximum(signal, 0)
            activity_sums[signal_type] = np.sum(positive_signal, axis=0)
            logger.info(f"Calculated ΔF/F activity sums for {n_neurons} neurons")

        else:  # Raw calcium signal
            # For raw signal, sum total fluorescence after baseline subtraction
            # Baseline is estimated as the 10th percentile of each neuron's trace
            baselines = np.percentile(signal, 10, axis=0)
            baseline_subtracted = signal - baselines[np.newaxis, :]
            positive_signal = np.maximum(baseline_subtracted, 0)
            activity_sums[signal_type] = np.sum(positive_signal, axis=0)
            logger.info(f"Calculated raw calcium activity sums for {n_neurons} neurons")

    return activity_sums


def get_top_active_neurons(activity_sums: Dict[str, np.ndarray], top_n: int = 20) -> Dict[str, np.ndarray]:
    """
    Identify the top N most active neurons for each signal type.

    Parameters
    ----------
    activity_sums : Dict[str, np.ndarray]
        Dictionary mapping signal type to array of total activity per neuron
    top_n : int, optional
        Number of top neurons to return, by default 20

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping signal type to array of top neuron indices
    """
    top_active_neurons = {}

    for signal_type, sums in activity_sums.items():
        # Get indices of top n neurons by activity
        top_indices = np.argsort(sums)[::-1][:top_n]
        top_active_neurons[signal_type] = top_indices

        logger.info(f"Identified top {top_n} active neurons for {signal_type}")
        logger.info(f"Top neuron indices: {top_indices}")

    return top_active_neurons


def extract_importance_values_from_results(results: Dict[str, Dict[str, Any]],
                                         model_name: str,
                                         signal_type: str,
                                         n_neurons: int) -> Optional[np.ndarray]:
    """
    Extract importance values from model results.

    This function tries to extract neuron importance values directly from the results
    dictionary rather than estimating or simulating them.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results dictionary
    model_name : str
        Name of the model
    signal_type : str
        Type of signal
    n_neurons : int
        Total number of neurons

    Returns
    -------
    Optional[np.ndarray]
        Array of importance values for each neuron, or None if not available
    """
    # Check if model exists in results
    if model_name not in results:
        logger.warning(f"Model {model_name} not found in results")
        return None

    # Check if signal type exists for this model
    if signal_type not in results[model_name]:
        logger.warning(f"Signal type {signal_type} not found for model {model_name}")
        return None

    # Try to get importance from importance_summary
    try:
        importance_summary = results[model_name][signal_type].get('importance_summary', {})

        # First try to get neuron importance directly
        if 'neuron_importance' in importance_summary:
            importance = np.array(importance_summary['neuron_importance'])
            if len(importance) == n_neurons:
                logger.info(f"Extracted neuron importance from results for {model_name} - {signal_type}")
                return importance

        # If not available, try to get importance matrix and calculate neuron importance
        if 'importance_matrix' in importance_summary:
            importance_matrix = np.array(importance_summary['importance_matrix'])
            # Calculate neuron importance by averaging across time
            neuron_importance = importance_matrix.mean(axis=0)
            if len(neuron_importance) == n_neurons:
                logger.info(f"Calculated neuron importance from importance matrix for {model_name} - {signal_type}")
                return neuron_importance

    except Exception as e:
        logger.warning(f"Error extracting importance from results: {e}")

    logger.warning(f"Could not extract importance values for {model_name} - {signal_type}")
    return None


def get_model_importance(results: Dict[str, Dict[str, Any]],
                        calcium_signals: Dict[str, np.ndarray],
                        model_name: str,
                        top_n: int = 20) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Get importance values and top neurons for a specific model.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results dictionary
    calcium_signals : Dict[str, np.ndarray]
        Dictionary of calcium signals
    model_name : str
        Name of the model
    top_n : int, optional
        Number of top neurons to return, by default 20

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary mapping signal type to (importance, top_indices)
    """
    importance_dict = {}

    for signal_type, signal in calcium_signals.items():
        if signal is None:
            continue

        n_neurons = signal.shape[1]

        # Extract importance values from results
        importance = extract_importance_values_from_results(results, model_name, signal_type, n_neurons)

        if importance is not None:
            # Get top neuron indices
            top_indices = np.argsort(importance)[::-1][:top_n]
            importance_dict[signal_type] = (importance, top_indices)
            logger.info(f"Extracted top {top_n} neurons for {model_name} - {signal_type}")
        else:
            # For models like SVM where importance might not be available,
            # try to use feature_importance extraction functions
            try:
                # Get model object if available (usually for deep learning models)
                if 'model' in results[model_name][signal_type]:
                    model = results[model_name][signal_type]['model']
                    # If model has get_feature_importance method
                    if hasattr(model, 'get_feature_importance'):
                        # Get importance matrix
                        importance_matrix = model.get_feature_importance(signal.shape[0], n_neurons)
                        # Calculate neuron importance by averaging across time
                        neuron_importance = importance_matrix.mean(axis=0)
                        # Get top indices
                        top_indices = np.argsort(neuron_importance)[::-1][:top_n]
                        importance_dict[signal_type] = (neuron_importance, top_indices)
                        logger.info(f"Extracted importance from model for {model_name} - {signal_type}")
                        continue
            except Exception as e:
                logger.warning(f"Error extracting importance from model: {e}")

            # If we still don't have importance, look for top_100_neurons in results
            try:
                if 'top_100_neurons' in results[model_name][signal_type]:
                    top_neurons = np.array(results[model_name][signal_type]['top_100_neurons'])
                    # Create importance array with high values for top neurons
                    importance = np.zeros(n_neurons)
                    # Only use the top N neurons
                    if len(top_neurons) > top_n:
                        top_neurons = top_neurons[:top_n]
                    importance[top_neurons] = 1.0
                    importance_dict[signal_type] = (importance, top_neurons)
                    logger.info(f"Using top_100_neurons from results for {model_name} - {signal_type}")
                    continue
            except Exception as e:
                logger.warning(f"Error extracting top neurons from results: {e}")

    return importance_dict


def calculate_overlap_metrics(model_important_neurons: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
                              top_active_neurons: Dict[str, np.ndarray],
                              top_n: int = 20) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Calculate the overlap between model-important neurons and most active neurons.

    Parameters
    ----------
    model_important_neurons : Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        Dictionary mapping model name to a dictionary mapping signal type to (importance, top_indices)
    top_active_neurons : Dict[str, np.ndarray]
        Dictionary mapping signal type to array of top active neuron indices
    top_n : int, optional
        Number of top neurons to consider, by default 20

    Returns
    -------
    Dict[str, Dict[str, Dict[str, Any]]]
        Dictionary mapping model name to a dictionary mapping signal type to overlap metrics
    """
    overlap_metrics = {}

    for model_name, signal_importance in model_important_neurons.items():
        overlap_metrics[model_name] = {}

        for signal_type, importance_data in signal_importance.items():
            if signal_type not in top_active_neurons:
                continue

            # Extract importance values and top indices
            if isinstance(importance_data, tuple) and len(importance_data) == 2:
                importance_values, important_indices = importance_data
            else:
                logger.warning(f"Unexpected format for importance data: {type(importance_data)}")
                continue

            # Convert to lists before creating sets
            # Handle both arrays and lists
            if hasattr(important_indices, 'tolist'):
                important_indices_list = important_indices[:top_n].tolist()
            else:
                important_indices_list = list(important_indices[:top_n])

            if hasattr(top_active_neurons[signal_type], 'tolist'):
                active_indices_list = top_active_neurons[signal_type][:top_n].tolist()
            else:
                active_indices_list = list(top_active_neurons[signal_type][:top_n])

            # Get sets of important and active neurons
            important_set = set(important_indices_list)
            active_set = set(active_indices_list)

            # Calculate overlap
            overlap_set = important_set.intersection(active_set)
            overlap_count = len(overlap_set)
            overlap_percentage = (overlap_count / top_n) * 100

            # Store metrics
            overlap_metrics[model_name][signal_type] = {
                'overlap_count': overlap_count,
                'overlap_percentage': overlap_percentage,
                'overlap_neurons': list(overlap_set),
                'important_only': list(important_set - active_set),
                'active_only': list(active_set - important_set)
            }

            logger.info(f"{model_name} vs {signal_type} overlap: {overlap_count} of top {top_n} "
                        f"important neurons are in top {top_n} active neurons ({overlap_percentage:.1f}% overlap)")

    return overlap_metrics


def plot_neuron_scatter(ax, positions, highlighted_indices, color, roi_matrix=None,
                        marker='x', marker_size=50, alpha=0.8, highlight_label=None):
    """
    Create a scatter plot of neurons with highlighted indices.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    positions : np.ndarray
        Array of (x, y) positions for each neuron
    highlighted_indices : np.ndarray or list
        Indices of neurons to highlight
    color : str
        Color for the highlighted neurons
    roi_matrix : np.ndarray, optional
        ROI matrix to plot in the background, by default None
    marker : str, optional
        Marker style for highlighted neurons, by default 'x'
    marker_size : int, optional
        Size of markers, by default 50
    alpha : float, optional
        Transparency level, by default 0.8
    highlight_label : str, optional
        Label for highlighted points in legend, by default None
    """
    # Plot ROI matrix if provided
    if roi_matrix is not None:
        ax.imshow(roi_matrix, cmap='gray', alpha=0.3)
    else:
        ax.set_facecolor('#f5f5f5')  # Light gray background

    # Plot all positions as gray dots
    ax.scatter(positions[:, 0], positions[:, 1], s=10, color='gray', alpha=0.5, label='All neurons')

    # Plot highlighted positions with specified color and marker
    if len(highlighted_indices) > 0:
        # Convert indices to list if they're not already
        if hasattr(highlighted_indices, 'tolist'):
            indices_list = highlighted_indices.tolist()
        else:
            indices_list = list(highlighted_indices)

        highlighted_positions = positions[indices_list]
        ax.scatter(highlighted_positions[:, 0], highlighted_positions[:, 1],
                   s=marker_size, color=color, marker=marker, alpha=alpha,
                   label=highlight_label, linewidths=1.5, edgecolors='white')

        # Add labels for the top 5 neurons
        for i, idx in enumerate(indices_list[:5]):
            ax.text(positions[idx, 0], positions[idx, 1], f"#{idx}",
                    fontsize=8, color='black', fontweight='bold',
                    ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])


def create_side_by_side_comparison(positions, model_important_indices, active_indices,
                                   signal_type, roi_matrix=None, top_n=20):
    """
    Create side-by-side scatter plots comparing model-important and most active neurons.

    Parameters
    ----------
    positions : np.ndarray
        Array of (x, y) positions for each neuron
    model_important_indices : np.ndarray or list
        Indices of neurons deemed important by the model
    active_indices : np.ndarray or list
        Indices of most active neurons
    signal_type : str
        Type of signal ('calcium_signal', 'deltaf_signal', or 'deconv_signal')
    roi_matrix : np.ndarray, optional
        ROI matrix to plot in the background, by default None
    top_n : int, optional
        Number of top neurons to consider, by default 20

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Get signal color
    signal_color = SIGNAL_COLORS[signal_type]
    signal_name = SIGNAL_DISPLAY_NAMES[signal_type]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Calculate overlap
    # Convert to lists before creating sets
    if hasattr(model_important_indices, 'tolist'):
        model_list = model_important_indices[:top_n].tolist()
    else:
        model_list = list(model_important_indices[:top_n])

    if hasattr(active_indices, 'tolist'):
        active_list = active_indices[:top_n].tolist()
    else:
        active_list = list(active_indices[:top_n])

    model_set = set(model_list)
    active_set = set(active_list)
    overlap_set = model_set.intersection(active_set)
    overlap_count = len(overlap_set)
    overlap_percentage = (overlap_count / top_n) * 100

    # Plot model-important neurons
    plot_neuron_scatter(
        ax1, positions, model_important_indices[:top_n], signal_color, roi_matrix,
        highlight_label=f"Top {top_n} by Model Importance"
    )
    ax1.set_title(f"Model-Important Neurons\n{signal_name} Signal",
                  fontsize=14, fontweight='bold', color=signal_color)

    # Plot most active neurons
    plot_neuron_scatter(
        ax2, positions, active_indices[:top_n], signal_color, roi_matrix,
        highlight_label=f"Top {top_n} by Activity"
    )
    ax2.set_title(f"Most Active Neurons\n{signal_name} Signal",
                  fontsize=14, fontweight='bold', color=signal_color)

    # Add colored border
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(signal_color)
            spine.set_linewidth(2)

        # Add legend
        ax.legend(loc='upper right')

    # Add overlap information as figure text
    fig.text(0.5, 0.02,
             f"Overlap: {overlap_count} of {top_n} neurons ({overlap_percentage:.1f}%)\n"
             f"Model-important and highly-active neurons",
             fontsize=12, fontweight='bold', ha='center')

    return fig


def plot_model_activity_comparison(
        calcium_signals: Dict[str, np.ndarray],
        excluded_cells: np.ndarray,
        roi_matrix: np.ndarray,
        model_importance: Dict[str, Tuple[np.ndarray, np.ndarray]],
        signal_type: str,
        model_name: str = 'random_forest',
        top_n: int = 20,
        output_path: Optional[str] = None,
        show_plot: bool = True
) -> plt.Figure:
    """
    Create a side-by-side comparison of model-important and most active neurons.

    Parameters
    ----------
    calcium_signals : Dict[str, np.ndarray]
        Dictionary containing all three signal types
    excluded_cells : np.ndarray
        Array of excluded cell indices
    roi_matrix : np.ndarray
        ROI matrix for visualization
    model_importance : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary mapping signal type to (importance, top_indices)
    signal_type : str
        Type of signal to analyze ('calcium_signal', 'deltaf_signal', or 'deconv_signal')
    model_name : str, optional
        Name of model to analyze, by default 'random_forest'
    top_n : int, optional
        Number of top neurons to consider, by default 20
    output_path : Optional[str], optional
        Path to save the figure, by default None
    show_plot : bool, optional
        Whether to display the plot, by default True

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    set_publication_style()

    if signal_type not in calcium_signals or calcium_signals[signal_type] is None:
        logger.error(f"Signal type {signal_type} not found in calcium signals")
        return None

    # Get calcium signal dimensions
    n_frames, n_neurons = calcium_signals[signal_type].shape

    # Calculate neuron positions
    positions = approximate_neuron_positions(roi_matrix, n_neurons)

    if signal_type not in model_importance:
        logger.error(f"Could not extract importance for {signal_type}")
        return None

    # Get model-important neuron indices
    _, model_important_indices = model_importance[signal_type]

    # Calculate neuron activity sums
    activity_sums = calculate_neuron_activity(calcium_signals)

    if signal_type not in activity_sums:
        logger.error(f"Could not calculate activity sums for {signal_type}")
        return None

    # Get most active neuron indices
    top_active_indices = np.argsort(activity_sums[signal_type])[::-1][:top_n]

    # Create visualization
    fig = create_side_by_side_comparison(
        positions, model_important_indices, top_active_indices,
        signal_type, roi_matrix, top_n
    )

    # Set main title
    model_display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    fig.suptitle(
        f"Comparing {model_display_name} Important Neurons vs. Most Active Neurons\n"
        f"Signal Type: {SIGNAL_DISPLAY_NAMES[signal_type]}",
        fontsize=16, fontweight='bold'
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    # Save figure if requested
    if output_path:
        # Make sure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save figure
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")

    # Show figure if requested
    if show_plot:
        plt.show()

    return fig


def create_comparison_grid(
        mat_file_path: str,
        results: Dict[str, Dict[str, Any]],
        output_dir: str,
        model_names: List[str] = ['random_forest', 'cnn'],
        top_n: int = 20,
        show_plot: bool = False
) -> plt.Figure:
    """
    Create a grid of comparison visualizations for all models and signal types.

    Parameters
    ----------
    mat_file_path : str
        Path to MATLAB file with calcium signals and ROI matrix
    results : Dict[str, Dict[str, Any]]
        Results dictionary
    output_dir : str
        Directory to save figure
    model_names : List[str], optional
        Names of models to analyze, by default ['random_forest', 'cnn']
    top_n : int, optional
        Number of top neurons to consider, by default 20
    show_plot : bool, optional
        Whether to display plot, by default False

    Returns
    -------
    plt.Figure
        The created figure
    """
    # Load data
    calcium_signals, roi_matrix, excluded_cells = load_data(mat_file_path)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define signal types
    signal_types = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    # Create grid figure
    fig, axes = plt.subplots(len(model_names), len(signal_types),
                             figsize=(6*len(signal_types), 5*len(model_names)))

    # Extract model importance for all models and signal types
    model_importance = {}
    for model_name in model_names:
        model_importance[model_name] = get_model_importance(results, calcium_signals, model_name, top_n)

    # Calculate activity sums and get top active neurons
    activity_sums = calculate_neuron_activity(calcium_signals)
    top_active_neurons = get_top_active_neurons(activity_sums, top_n)

    # Calculate overlap metrics
    overlap_metrics = calculate_overlap_metrics(model_importance, top_active_neurons, top_n)

    # Calculate neuron positions
    n_neurons = calcium_signals['calcium_signal'].shape[1] if 'calcium_signal' in calcium_signals else 0
    if n_neurons > 0:
        positions = approximate_neuron_positions(roi_matrix, n_neurons)
    else:
        logger.error("Could not determine neuron count")
        return None

    # Create a subplot for each model-signal combination
    for i, model_name in enumerate(model_names):
        for j, signal_type in enumerate(signal_types):
            if len(model_names) == 1:
                if len(signal_types) == 1:
                    ax = axes
                else:
                    ax = axes[j]
            else:
                if len(signal_types) == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]

            # Skip if signal not available
            if signal_type not in calcium_signals or calcium_signals[signal_type] is None:
                ax.text(0.5, 0.5, f"No {signal_type} data",
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            # Get model-important neuron indices
            if model_name in model_importance and signal_type in model_importance[model_name]:
                importance_data = model_importance[model_name][signal_type]
                if isinstance(importance_data, tuple) and len(importance_data) == 2:
                    _, model_important_indices = importance_data
                else:
                    ax.text(0.5, 0.5, f"Invalid importance data for {model_name}",
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                    continue
            else:
                ax.text(0.5, 0.5, f"No importance data for {model_name}",
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            # Get most active neuron indices
            if signal_type in top_active_neurons:
                active_indices = top_active_neurons[signal_type]
            else:
                ax.text(0.5, 0.5, f"No activity data for {signal_type}",
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            # Get signal color
            signal_color = SIGNAL_COLORS[signal_type]

            # Calculate overlap
            # Convert to lists before creating sets
            if hasattr(model_important_indices, 'tolist'):
                model_list = model_important_indices[:top_n].tolist()
            else:
                model_list = list(model_important_indices[:top_n])

            if hasattr(active_indices, 'tolist'):
                active_list = active_indices[:top_n].tolist()
            else:
                active_list = list(active_indices[:top_n])

            model_set = set(model_list)
            active_set = set(active_list)
            overlap_set = model_set.intersection(active_set)
            overlap_count = len(overlap_set)
            overlap_percentage = (overlap_count / top_n) * 100

            # Create ROI background with low alpha
            if roi_matrix is not None:
                ax.imshow(roi_matrix, cmap='gray', alpha=0.15)
            else:
                ax.set_facecolor('#f8f8f8')

            # Plot all neurons as gray dots
            ax.scatter(positions[:, 0], positions[:, 1], s=5, color='gray', alpha=0.3)

            # Plot model-important neurons as circles
            if hasattr(model_important_indices, 'tolist'):
                model_indices = model_important_indices[:top_n].tolist()
            else:
                model_indices = list(model_important_indices[:top_n])

            model_positions = positions[model_indices]
            ax.scatter(model_positions[:, 0], model_positions[:, 1],
                      s=50, color=signal_color, marker='o', alpha=0.6,
                      label='Model Important', edgecolors='white', linewidths=0.5)

            # Plot active neurons as x markers
            if hasattr(active_indices, 'tolist'):
                active_list = active_indices[:top_n].tolist()
            else:
                active_list = list(active_indices[:top_n])

            active_positions = positions[active_list]
            ax.scatter(active_positions[:, 0], active_positions[:, 1],
                      s=50, color='black', marker='x', alpha=0.7,
                      label='Most Active', linewidths=1.5)

            # Plot overlap neurons with special marker
            if overlap_count > 0:
                overlap_indices = list(overlap_set)
                overlap_positions = positions[overlap_indices]
                ax.scatter(overlap_positions[:, 0], overlap_positions[:, 1],
                          s=120, facecolor='none', edgecolors=signal_color, marker='o',
                          linewidths=2, label='Overlap')

            # Set title
            model_display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            signal_display_name = SIGNAL_DISPLAY_NAMES.get(signal_type, signal_type)

            ax.set_title(f"{model_display_name} vs. {signal_display_name}\n"
                        f"Overlap: {overlap_count}/{top_n} ({overlap_percentage:.1f}%)",
                        fontsize=12, color=signal_color)

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Add colored border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(signal_color)
                spine.set_linewidth(1.5)

            # Add legend only to the first subplot
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=9)

    # Set main title
    fig.suptitle("Comparison of Model-Important vs. Most Active Neurons",
                fontsize=18, fontweight='bold')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path = output_dir / "model_activity_comparison_grid.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison grid to {output_path}")

    # Save metrics as text file
    metrics_path = output_dir / "model_activity_overlap_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("Model-Important vs. Most Active Neuron Overlap Metrics\n")
        f.write("=" * 60 + "\n\n")

        for model_name in model_names:
            model_display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            f.write(f"{model_display_name}:\n")
            f.write("-" * 30 + "\n")

            for signal_type in signal_types:
                if signal_type in calcium_signals and calcium_signals[signal_type] is not None:
                    signal_display_name = SIGNAL_DISPLAY_NAMES.get(signal_type, signal_type)

                    metrics = overlap_metrics.get(model_name, {}).get(signal_type, {})

                    if metrics:
                        f.write(f"{signal_display_name} Signal:\n")
                        f.write(f"  Overlap: {metrics['overlap_count']} of {top_n} neurons ")
                        f.write(f"({metrics['overlap_percentage']:.1f}%)\n")
                        f.write(f"  Overlapping neurons: {metrics['overlap_neurons']}\n")
                        f.write(f"  Model-important only: {metrics['important_only']}\n")
                        f.write(f"  Active-only: {metrics['active_only']}\n\n")

            f.write("\n")

    logger.info(f"Saved overlap metrics to {metrics_path}")

    # Show figure if requested
    if show_plot:
        plt.show()

    return fig


def analyze_neuron_activity_importance(
        mat_file_path: str,
        results: Dict[str, Dict[str, Any]],
        output_dir: str,
        model_names: List[str] = ['random_forest', 'cnn'],
        top_n: int = 20
) -> Dict[str, Any]:
    """
    Analyze the relationship between neuron activity and model importance.

    This function:
    1. Calculates activity sums for all neurons
    2. Extracts model-important neurons
    3. Computes overlap metrics
    4. Creates visualizations
    5. Saves detailed metrics

    Parameters
    ----------
    mat_file_path : str
        Path to MATLAB file with calcium signals and ROI matrix
    results : Dict[str, Dict[str, Any]]
        Results dictionary from model training
    output_dir : str
        Directory to save output files
    model_names : List[str], optional
        Names of models to analyze, by default ['random_forest', 'cnn']
    top_n : int, optional
        Number of top neurons to consider, by default 20

    Returns
    -------
    Dict[str, Any]
        Analysis results including metrics and file paths
    """
    # Set up publication style
    set_publication_style()

    # Load data
    calcium_signals, roi_matrix, excluded_cells = load_data(mat_file_path)

    # Create output directory
    output_dir = Path(output_dir)
    neuron_dir = output_dir / 'neuron_analysis'
    neuron_dir.mkdir(parents=True, exist_ok=True)

    # Calculate neuron activity sums
    activity_sums = calculate_neuron_activity(calcium_signals)

    # Get top active neurons
    top_active_neurons = get_top_active_neurons(activity_sums, top_n)

    # Extract model importance for each model and signal type
    model_importance = {}
    for model_name in model_names:
        model_importance[model_name] = get_model_importance(results, calcium_signals, model_name, top_n)

    # Calculate overlap metrics
    overlap_metrics = calculate_overlap_metrics(model_importance, top_active_neurons, top_n)

    # Generate grid visualization
    grid_fig = create_comparison_grid(
        mat_file_path=mat_file_path,
        results=results,
        output_dir=neuron_dir,
        model_names=model_names,
        top_n=top_n,
        show_plot=False
    )

    # Generate individual visualizations for each model and signal type
    individual_figs = []
    for model_name in model_names:
        if model_name in model_importance:
            for signal_type in ['calcium_signal', 'deltaf_signal', 'deconv_signal']:
                if signal_type in calcium_signals and calcium_signals[signal_type] is not None and signal_type in model_importance[model_name]:
                    output_path = neuron_dir / f"{model_name}_{signal_type}_comparison.png"

                    fig = plot_model_activity_comparison(
                        calcium_signals=calcium_signals,
                        excluded_cells=excluded_cells,
                        roi_matrix=roi_matrix,
                        model_importance=model_importance[model_name],
                        signal_type=signal_type,
                        model_name=model_name,
                        top_n=top_n,
                        output_path=str(output_path),
                        show_plot=False
                    )

                    if fig is not None:
                        individual_figs.append((model_name, signal_type, fig))

    # Generate correlation analysis between activity and importance
    correlation_results = {}
    for model_name in model_names:
        if model_name in model_importance:
            correlation_results[model_name] = {}

            for signal_type in activity_sums.keys():
                if signal_type in model_importance[model_name]:
                    importance_data = model_importance[model_name][signal_type]
                    if isinstance(importance_data, tuple) and len(importance_data) == 2:
                        importance_values, _ = importance_data
                        activity_values = activity_sums[signal_type]

                        # Ensure arrays have the same length
                        min_length = min(len(importance_values), len(activity_values))
                        importance_values = importance_values[:min_length]
                        activity_values = activity_values[:min_length]

                        # Calculate correlation
                        if np.any(importance_values) and np.any(activity_values):
                            correlation = np.corrcoef(importance_values, activity_values)[0, 1]
                            correlation_results[model_name][signal_type] = correlation

                            logger.info(f"{model_name} {signal_type} correlation: {correlation:.4f}")

    # Create correlation visualization
    corr_fig = plt.figure(figsize=(10, 6))
    ax = corr_fig.add_subplot(111)

    bar_positions = []
    bar_heights = []
    bar_colors = []
    bar_labels = []

    position = 0
    for model_name in correlation_results.keys():
        model_display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

        for signal_type, correlation in correlation_results[model_name].items():
            signal_display_name = SIGNAL_DISPLAY_NAMES.get(signal_type, signal_type)
            signal_color = SIGNAL_COLORS[signal_type]

            bar_positions.append(position)
            bar_heights.append(correlation)
            bar_colors.append(signal_color)
            bar_labels.append(f"{model_display_name}\n{signal_display_name}")

            position += 1

        # Add gap between models
        position += 0.5

    # Plot bars
    bars = ax.bar(bar_positions, bar_heights, color=bar_colors, width=0.7)

    # Add labels and styling
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=0)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title('Correlation Between Neuron Activity and Model Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-0.5, 1.0)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        text_color = 'black' if height < 0.7 else 'white'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', color=text_color, fontweight='bold')

    # Save correlation figure
    corr_path = neuron_dir / "activity_importance_correlation.png"
    corr_fig.savefig(corr_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved correlation analysis to {corr_path}")

    # Save comprehensive analysis report
    report_path = neuron_dir / "activity_importance_analysis.txt"
    with open(report_path, 'w') as f:
        f.write("Neuron Activity vs. Model Importance Analysis\n")
        f.write("=" * 50 + "\n\n")

        f.write("OVERLAP METRICS\n")
        f.write("-" * 30 + "\n\n")

        for model_name, signal_metrics in overlap_metrics.items():
            model_display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            f.write(f"{model_display_name}:\n")

            for signal_type, metrics in signal_metrics.items():
                signal_display_name = SIGNAL_DISPLAY_NAMES.get(signal_type, signal_type)
                f.write(f"  {signal_display_name} Signal:\n")
                f.write(f"    Overlap: {metrics['overlap_count']} of {top_n} neurons ")
                f.write(f"({metrics['overlap_percentage']:.1f}%)\n")
                f.write(f"    Overlapping neurons: {metrics['overlap_neurons']}\n\n")

            f.write("\n")

        f.write("CORRELATION ANALYSIS\n")
        f.write("-" * 30 + "\n\n")

        for model_name, correlations in correlation_results.items():
            model_display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            f.write(f"{model_display_name}:\n")

            for signal_type, corr in correlations.items():
                signal_display_name = SIGNAL_DISPLAY_NAMES.get(signal_type, signal_type)
                f.write(f"  {signal_display_name} Signal: {corr:.4f}\n")

            f.write("\n")

        f.write("SUMMARY\n")
        f.write("-" * 30 + "\n\n")

        # Find best model and signal type based on overlap
        best_overlap = 0
        best_model = ""
        best_signal = ""

        for model_name, signal_metrics in overlap_metrics.items():
            for signal_type, metrics in signal_metrics.items():
                if metrics['overlap_percentage'] > best_overlap:
                    best_overlap = metrics['overlap_percentage']
                    best_model = model_name
                    best_signal = signal_type

        if best_model and best_signal:
            best_model_display = MODEL_DISPLAY_NAMES.get(best_model, best_model)
            best_signal_display = SIGNAL_DISPLAY_NAMES.get(best_signal, best_signal)

            f.write(f"Best overlap observed for {best_model_display} model with {best_signal_display} signal:\n")
            f.write(f"{best_overlap:.1f}% overlap between most important and most active neurons.\n\n")

        # Find highest correlation
        best_corr = -1
        best_corr_model = ""
        best_corr_signal = ""

        for model_name, correlations in correlation_results.items():
            for signal_type, corr in correlations.items():
                if corr > best_corr:
                    best_corr = corr
                    best_corr_model = model_name
                    best_corr_signal = signal_type

        if best_corr_model and best_corr_signal:
            best_corr_model_display = MODEL_DISPLAY_NAMES.get(best_corr_model, best_corr_model)
            best_corr_signal_display = SIGNAL_DISPLAY_NAMES.get(best_corr_signal, best_corr_signal)

            f.write(f"Highest correlation observed for {best_corr_model_display} model with {best_corr_signal_display} signal:\n")
            f.write(f"Correlation coefficient: {best_corr:.4f}\n\n")

        f.write("Interpretation:\n")
        for signal_type in ['deconv_signal', 'deltaf_signal', 'calcium_signal']:
            if signal_type in activity_sums:
                signal_display_name = SIGNAL_DISPLAY_NAMES.get(signal_type, signal_type)

                model_overlaps = []
                for model_name in model_names:
                    if model_name in overlap_metrics and signal_type in overlap_metrics[model_name]:
                        model_overlaps.append((model_name, overlap_metrics[model_name][signal_type]['overlap_percentage']))

                if model_overlaps:
                    avg_overlap = sum(perc for _, perc in model_overlaps) / len(model_overlaps)
                    f.write(f"- {signal_display_name} Signal: Average overlap of {avg_overlap:.1f}% across models.\n")

        f.write("\nConclusion: ")
        if best_overlap > 60:
            f.write("Strong overlap between model-important neurons and most active neurons, ")
            f.write("confirming that the decoder is focusing on the naturally most active cells.\n")
        elif best_overlap > 40:
            f.write("Moderate overlap between model-important neurons and most active neurons, ")
            f.write("suggesting the decoder is capturing some subtler signals beyond just the most active cells.\n")
        else:
            f.write("Limited overlap between model-important neurons and most active neurons, ")
            f.write("indicating the decoder may be picking up on complex patterns beyond raw activity levels.\n")

    logger.info(f"Saved comprehensive analysis report to {report_path}")

    # Return analysis results
    results = {
        'overlap_metrics': overlap_metrics,
        'correlation_results': correlation_results,
        'top_active_neurons': top_active_neurons,
        'files': {
            'grid_visualization': str(neuron_dir / "model_activity_comparison_grid.png"),
            'correlation_visualization': str(corr_path),
            'analysis_report': str(report_path)
        }
    }

    return results

