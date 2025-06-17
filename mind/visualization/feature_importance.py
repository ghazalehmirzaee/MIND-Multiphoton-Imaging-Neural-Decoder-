# mind/visualization/feature_importance.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Any
import logging

from .config import (SIGNAL_COLORS, SIGNAL_DISPLAY_NAMES, MODEL_DISPLAY_NAMES,
                     set_publication_style, get_signal_colormap, SIGNAL_GRADIENTS,
                     FIGURE_SIZES)

logger = logging.getLogger(__name__)


def plot_temporal_importance_patterns(
        results: Dict[str, Dict[str, Any]],
        frame_labels: Optional[np.ndarray] = None,
        window_size: int = 15,
        output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create temporal importance bar plots that focuses on the most informative time steps.
    """
    set_publication_style()

    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    # Define the frame range we want to visualize (4-14, which is indices 4:15)
    start_frame = 4
    end_frame = 15  # 10 frames total
    frame_indices = slice(start_frame, end_frame)
    displayed_frames = list(range(start_frame, end_frame))

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['grid_1x3'])

    # First pass: collect all temporal importance values to determine global scaling
    all_importance_values = []
    signal_importance_data = {}

    for signal in signals:
        try:
            importance_summary = results['random_forest'][signal]['importance_summary']
            temporal_importance = np.array(importance_summary['temporal_importance'])

            # Extract only the frames we want to display (4-14)
            selected_importance = temporal_importance[frame_indices]
            signal_importance_data[signal] = selected_importance

            # Add to global collection for scaling calculation
            all_importance_values.extend(selected_importance)

        except (KeyError, TypeError, ValueError):
            # Handle missing data by creating zeros
            signal_importance_data[signal] = np.zeros(len(displayed_frames))
            logger.warning(f"No temporal data available for {signal}")

    # Calculate global y-axis limits for consistent scaling
    if all_importance_values:
        global_min = min(all_importance_values)
        global_max = max(all_importance_values)

        # Add some padding to the limits for better visualization
        padding = (global_max - global_min) * 0.1
        y_min = max(0, global_min - padding)  # Don't go below 0
        y_max = global_max + padding

        # If the range is very small, set a minimum range for visibility
        if (y_max - y_min) < 0.01:
            y_max = global_max + 0.01

    else:
        # Fallback limits if no data is available
        y_min, y_max = 0, 0.1

    # Second pass: create the actual plots with consistent scaling
    for j, signal in enumerate(signals):
        ax = axes[j]

        temporal_importance = signal_importance_data[signal]

        if len(temporal_importance) > 0 and np.any(temporal_importance):
            # Get signal-specific color scheme for consistent visualization
            signal_color = SIGNAL_COLORS[signal]
            gradient = SIGNAL_GRADIENTS[signal]

            # Create bars with gradient effect based on importance values
            # Use displayed_frames for x-axis positioning to show actual frame numbers
            bars = ax.bar(displayed_frames, temporal_importance)

            # Apply gradient coloring to bars for visual appeal and information encoding
            if temporal_importance.max() > 0:  # Avoid division by zero
                for i, bar in enumerate(bars):
                    # Calculate gradient intensity based on relative importance
                    intensity = temporal_importance[i] / temporal_importance.max()
                    color_idx = min(int(intensity * (len(gradient) - 1)), len(gradient) - 1)
                    bar.set_color(gradient[color_idx])
                    bar.set_edgecolor(signal_color)
                    bar.set_linewidth(0.5)
            else:
                # If all values are zero, use the base color
                for bar in bars:
                    bar.set_color(gradient[0])
                    bar.set_edgecolor(signal_color)
                    bar.set_linewidth(0.5)

            # Add contralateral footstep markers if behavioral data is provided
            if frame_labels is not None:
                # Track marked positions to avoid overcrowding labels
                marked_positions = []

                for i, frame_num in enumerate(displayed_frames):
                    # Check if there's a contralateral footstep at this frame
                    if frame_num < len(frame_labels) and frame_labels[frame_num] == 1:
                        # Add distinctive red triangle marker above the bar
                        marker_y = temporal_importance[i] + y_max * 0.08
                        ax.plot(frame_num, marker_y, 'v', color='red', markersize=10,
                                markeredgecolor='darkred', markeredgewidth=1.5,
                                zorder=10)

                        # Add "CONTRA" label above the marker for clarity
                        label_y = marker_y + y_max * 0.05
                        ax.text(frame_num, label_y, 'CONTRA',
                                ha='center', va='bottom', fontsize=8,
                                color='darkred', fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2",
                                          facecolor='white', edgecolor='darkred',
                                          alpha=0.8))

                        marked_positions.append(frame_num)

                # Add informative legend if footstep markers are present
                if marked_positions:
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], marker='v', color='red', linestyle='None',
                               markersize=10, markeredgecolor='darkred',
                               markeredgewidth=1.5,
                               label='Contralateral Footstep Event')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right',
                              framealpha=0.95, fontsize=9,
                              fancybox=True, shadow=True)

        else:
            # Handle cases where temporal data is not available
            ax.text(0.5, 0.5, 'No temporal data',
                    ha='center', va='center', transform=ax.transAxes)

        # Set axis labels and styling with conditional y-axis labeling
        ax.set_xlabel('Time Step', fontsize=12)

        # Only add y-axis label to the first subplot for cleaner presentation
        if j == 0:
            ax.set_ylabel('Mean Feature Importance', fontsize=12)

        ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]}',
                     fontsize=14, fontweight='bold', color=SIGNAL_COLORS[signal])
        ax.grid(True, alpha=0.3, axis='y')

        # Apply consistent styling across all subplots
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(SIGNAL_COLORS[signal])
        ax.spines['left'].set_linewidth(2)

        # Set consistent y-axis limits across all subplots
        ax.set_ylim(y_min, y_max)

        # Set x-axis limits and ticks to show frames 4-14
        ax.set_xlim(start_frame - 0.5, end_frame - 0.5)
        ax.set_xticks(displayed_frames)
        ax.set_xticklabels([str(f) for f in displayed_frames])

    # Create informative title that reflects the focused frame range
    title = f'Temporal Importance Patterns'
    if frame_labels is not None:
        title += ' with Movement Events'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    if output_dir:
        output_path = Path(output_dir) / 'temporal_importance_patterns.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved temporal importance patterns to {output_path}")

    return fig


def plot_top_neuron_importance(
        results: Dict[str, Dict[str, Any]],
        output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create horizontal bar plots with scatter overlay showing individual importance values.

    This visualization combines mean importance (as horizontal bars) with the distribution
    of importance values across time steps (as scatter points) for each top neuron.
    """
    set_publication_style()

    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for j, signal in enumerate(signals):
        ax = axes[j]

        try:
            importance_summary = results['random_forest'][signal]['importance_summary']
            importance_matrix = np.array(importance_summary['importance_matrix'])
            neuron_importance = np.array(importance_summary['neuron_importance'])
            top_indices = np.array(importance_summary['top_neuron_indices'])[:20]

            # Get importance values for top neurons
            top_importance = neuron_importance[top_indices]

            # Get signal color and gradient for consistent styling
            signal_color = SIGNAL_COLORS[signal]
            gradient = SIGNAL_GRADIENTS[signal]

            # Create horizontal bars with enhanced visibility (darker background)
            y_positions = np.arange(len(top_importance))

            # Use a more visible background color (darker than before)
            # Choose a color that's 60% of the way through the gradient for better visibility
            background_color_idx = min(int(0.6 * (len(gradient) - 1)), len(gradient) - 1)
            background_color = gradient[background_color_idx]

            bars = ax.barh(y_positions, top_importance,
                           color=background_color, alpha=0.7,
                           edgecolor=signal_color, linewidth=1.5)

            # Get individual importance values for each neuron across time steps
            for i, neuron_idx in enumerate(top_indices):
                # Extract importance values for this neuron across all time steps
                neuron_temporal_importance = importance_matrix[:, neuron_idx]

                # Create scatter points with slight jitter for better visibility
                jitter = np.random.normal(0, 0.08, len(neuron_temporal_importance))
                y_scatter = np.full(len(neuron_temporal_importance), i) + jitter

                # Plot scatter points with gradient color mapping
                scatter = ax.scatter(neuron_temporal_importance, y_scatter,
                                     alpha=0.7, s=25, c=neuron_temporal_importance,
                                     cmap=get_signal_colormap(signal),
                                     edgecolors='white', linewidth=0.8,
                                     zorder=5)

                # Add prominent mean value marker
                ax.plot([top_importance[i]], [i], 'o',
                        color=signal_color, markersize=10,
                        markeredgecolor='white', markeredgewidth=2.5,
                        zorder=10)

            # Customize y-axis with simplified labels (ranking instead of specific neuron IDs)
            ax.set_yticks(y_positions)
            # Create simplified ranking labels instead of showing specific neuron IDs
            rank_labels = [f'#{i + 1}' for i in range(len(top_importance))]
            ax.set_yticklabels(rank_labels, fontsize=10)

            # Set axis labels and title
            ax.set_xlabel('Feature Importance', fontsize=12)

            # Only add "Neuron ID" label to the first subplot for cleaner presentation
            if j == 0:
                ax.set_ylabel('Neuron Rank', fontsize=12)

            ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]} - Top 20 Neurons',
                         fontsize=14, fontweight='bold', color=signal_color)

            # Add grid and styling for better readability
            ax.grid(True, alpha=0.3, axis='x')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(signal_color)
            ax.spines['bottom'].set_linewidth(2)

            # Invert y-axis so highest importance neuron appears at top
            ax.invert_yaxis()

            # Add colorbar for scatter points (only on rightmost subplot)
            if j == 2:
                cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.03)
                cbar.set_label('Time-specific\nImportance', fontsize=10)
                cbar.ax.tick_params(labelsize=8)

            # Create comprehensive legend (only on the first subplot to avoid repetition)
            if j == 0:
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='s', color='w',
                           markerfacecolor=background_color, markersize=12,
                           alpha=0.7, markeredgecolor=signal_color,
                           markeredgewidth=1.5, label='Mean Importance Range'),
                    Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=signal_color, markersize=10,
                           markeredgecolor='white', markeredgewidth=2.5,
                           label='Mean Value'),
                    Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=gradient[2], markersize=6,
                           alpha=0.7, markeredgecolor='white',
                           markeredgewidth=0.8, label='Individual Time Points')
                ]
                ax.legend(handles=legend_elements, loc='lower right',
                          framealpha=0.95, fontsize=9,
                          fancybox=True, shadow=True)

        except (KeyError, TypeError, ValueError) as e:
            # Handle cases where neuron data is not available
            ax.text(0.5, 0.5, f'No neuron data\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('Top 20 Neuron Importance Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / 'top_neuron_importance.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved top neuron importance to {output_path}")

    return fig


