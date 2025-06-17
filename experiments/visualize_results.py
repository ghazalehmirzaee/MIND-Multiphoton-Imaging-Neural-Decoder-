# """
# Visualize results from experiments.
# """
# import os
# import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import json
# import argparse
# from typing import Dict, List, Any, Optional, Union
# import logging
#
# # Add project root to path
# sys.path.append(str(Path(__file__).parent.parent))
#
# from mind.utils.logging import setup_logging
# from mind.visualization.performance import (
#     plot_performance_bars, plot_confusion_matrix,
#     plot_performance_improvement, plot_performance_radar
# )
# from mind.visualization.feature_importance import (
#     plot_temporal_importance, plot_neuron_importance,
#     plot_importance_heatmap
# )
#
# # Set up logging
# logger = logging.getLogger(__name__)
#
#
# def load_results(results_dir: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
#     """
#     Load experiment results from JSON files.
#
#     Parameters
#     ----------
#     results_dir : Union[str, Path]
#         Directory containing result JSON files
#
#     Returns
#     -------
#     Dict[str, Dict[str, Any]]
#         Dictionary of results organized by model name and signal type
#     """
#     logger.info(f"Loading results from {results_dir}")
#
#     results_dir = Path(results_dir)
#     results = {}
#
#     # Find all JSON files in the results directory
#     json_files = list(results_dir.glob("*.json"))
#     logger.info(f"Found {len(json_files)} result files")
#
#     for json_file in json_files:
#         try:
#             # Load JSON file
#             with open(json_file, 'r') as f:
#                 result_data = json.load(f)
#
#             # Get metadata
#             metadata = result_data.get('metadata', {})
#             model_name = metadata.get('model_name')
#             signal_type = metadata.get('signal_type')
#
#             # Skip files without proper metadata
#             if model_name is None or signal_type is None:
#                 logger.warning(f"Skipping file {json_file} due to missing metadata")
#                 continue
#
#             # Add to results dictionary
#             if model_name not in results:
#                 results[model_name] = {}
#
#             results[model_name][signal_type] = result_data
#
#             logger.info(f"Loaded results for {model_name} on {signal_type}")
#
#         except Exception as e:
#             logger.error(f"Error loading {json_file}: {e}")
#
#     return results
#
#
# def create_performance_dataframe(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
#     """
#     Create a DataFrame of performance metrics from results.
#
#     Parameters
#     ----------
#     results : Dict[str, Dict[str, Any]]
#         Dictionary of results
#
#     Returns
#     -------
#     pd.DataFrame
#         DataFrame of performance metrics
#     """
#     data = []
#
#     for model_name, model_results in results.items():
#         for signal_type, signal_results in model_results.items():
#             # Get metrics
#             metrics = signal_results.get('metrics', {})
#
#             # Add row to data
#             data.append({
#                 'Model': model_name,
#                 'Signal Type': signal_type,
#                 'Accuracy': metrics.get('accuracy', 0.0),
#                 'Precision': metrics.get('precision', 0.0),
#                 'Recall': metrics.get('recall', 0.0),
#                 'F1 Score': metrics.get('f1_score', 0.0),
#                 'ROC AUC': metrics.get('roc_auc', 0.0) if 'roc_auc' in metrics else np.nan,
#                 'Train Time': signal_results.get('train_time', 0.0)
#             })
#
#     return pd.DataFrame(data)
#
#
# def main():
#     # Parse command-line arguments
#     parser = argparse.ArgumentParser(description="Visualize experiment results")
#     parser.add_argument("--results_dir", type=str, default="outputs/results",
#                         help="Directory containing result files")
#     parser.add_argument("--output_dir", type=str, default="outputs/figures",
#                         help="Directory to save figures")
#     parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"],
#                         help="Format for output figures")
#
#     args = parser.parse_args()
#
#     # Set up logging
#     setup_logging(log_level='INFO', console=True)
#
#     try:
#         # Load results
#         results = load_results(args.results_dir)
#
#         if not results:
#             logger.error(f"No results found in {args.results_dir}")
#             return
#
#         # Create output directory
#         output_dir = Path(args.output_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)
#
#         # Create performance DataFrame
#         df = create_performance_dataframe(results)
#
#         # Save performance table to CSV
#         df.to_csv(output_dir / "performance_table.csv", index=False)
#         logger.info(f"Saved performance table to {output_dir / 'performance_table.csv'}")
#
#         # Create performance visualizations
#         logger.info("Creating performance visualizations")
#
#         # Radar charts
#         plot_performance_radar(df, output_dir=output_dir)
#
#         # Bar charts for each metric
#         for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
#             plot_performance_bars(df, metric=metric, output_dir=output_dir)
#
#         # Performance improvement chart
#         plot_performance_improvement(df, output_dir=output_dir)
#
#         # Confusion matrices
#         for model_name, model_results in results.items():
#             for signal_type, signal_results in model_results.items():
#                 confusion_matrix = np.array(signal_results.get('confusion_matrix', []))
#                 if confusion_matrix.size > 0:
#                     plot_confusion_matrix(
#                         confusion_matrix, model_name, signal_type, output_dir=output_dir
#                     )
#
#         # Feature importance visualizations
#         logger.info("Creating feature importance visualizations")
#
#         for model_name, model_results in results.items():
#             for signal_type, signal_results in model_results.items():
#                 # Get importance summary
#                 importance_summary = signal_results.get('importance_summary', {})
#
#                 if importance_summary:
#                     # Get importance matrix
#                     importance_matrix = np.array(importance_summary.get('importance_matrix', []))
#
#                     if importance_matrix.size > 0:
#                         # Plot temporal importance
#                         plot_temporal_importance(
#                             importance_matrix, model_name, signal_type, output_dir=output_dir
#                         )
#
#                         # Plot neuron importance
#                         top_neuron_indices = np.array(importance_summary.get('top_neuron_indices', []))
#                         if top_neuron_indices.size > 0:
#                             plot_neuron_importance(
#                                 importance_matrix, top_neuron_indices, model_name, signal_type, output_dir=output_dir
#                             )
#
#                         # Plot importance heatmap
#                         plot_importance_heatmap(
#                             importance_matrix, model_name, signal_type, output_dir=output_dir
#                         )
#
#         logger.info(f"All visualizations saved to {output_dir}")
#
#     except Exception as e:
#         logger.error(f"Error visualizing results: {e}", exc_info=True)
#
#
# if __name__ == "__main__":
#     main()
#


"""
Visualize results from experiments.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, List, Any, Optional, Union
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mind.utils.logging import setup_logging
from mind.visualization.performance import (
    plot_confusion_matrix_grid, plot_roc_curve_grid,
    plot_precision_recall_grid, plot_performance_radar,
    plot_model_performance_heatmap
)
from mind.visualization.feature_importance import (
    plot_temporal_importance_patterns, plot_top_neuron_importance
)

# Set up logging
logger = logging.getLogger(__name__)


def load_results(results_dir: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load experiment results from JSON files.

    Parameters
    ----------
    results_dir : Union[str, Path]
        Directory containing result JSON files

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary of results organized by model name and signal type
    """
    logger.info(f"Loading results from {results_dir}")

    results_dir = Path(results_dir)
    results = {}

    # Find all JSON files in the results directory
    json_files = list(results_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} result files")

    for json_file in json_files:
        try:
            # Load JSON file
            with open(json_file, 'r') as f:
                result_data = json.load(f)

            # Get metadata
            metadata = result_data.get('metadata', {})
            model_name = metadata.get('model_name')
            signal_type = metadata.get('signal_type')

            # Skip files without proper metadata
            if model_name is None or signal_type is None:
                logger.warning(f"Skipping file {json_file} due to missing metadata")
                continue

            # Add to results dictionary
            if model_name not in results:
                results[model_name] = {}

            results[model_name][signal_type] = result_data

            logger.info(f"Loaded results for {model_name} on {signal_type}")

        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")

    return results


def create_performance_dataframe(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a DataFrame of performance metrics from results.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Dictionary of results

    Returns
    -------
    pd.DataFrame
        DataFrame of performance metrics
    """
    data = []

    for model_name, model_results in results.items():
        for signal_type, signal_results in model_results.items():
            # Get metrics
            metrics = signal_results.get('metrics', {})

            # Add row to data
            data.append({
                'Model': model_name,
                'Signal Type': signal_type,
                'Accuracy': metrics.get('accuracy', 0.0),
                'Precision': metrics.get('precision', 0.0),
                'Recall': metrics.get('recall', 0.0),
                'F1 Score': metrics.get('f1_score', 0.0),
                'ROC AUC': metrics.get('roc_auc', 0.0) if 'roc_auc' in metrics else np.nan,
                'Train Time': signal_results.get('train_time', 0.0)
            })

    return pd.DataFrame(data)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("--results_dir", type=str, default="outputs/results",
                        help="Directory containing result files")
    parser.add_argument("--output_dir", type=str, default="outputs/figures",
                        help="Directory to save figures")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"],
                        help="Format for output figures")

    args = parser.parse_args()

    # Set up logging
    setup_logging(log_level='INFO', console=True)

    try:
        # Load results
        results = load_results(args.results_dir)

        if not results:
            logger.error(f"No results found in {args.results_dir}")
            return

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create performance DataFrame
        df = create_performance_dataframe(results)

        # Save performance table to CSV
        df.to_csv(output_dir / "performance_table.csv", index=False)
        logger.info(f"Saved performance table to {output_dir / 'performance_table.csv'}")

        # Create performance visualizations
        logger.info("Creating performance visualizations")

        # Performance radar charts
        plot_performance_radar(results, output_dir=output_dir)

        # Performance heatmap
        plot_model_performance_heatmap(results, output_dir=output_dir)

        # Confusion matrices grid
        plot_confusion_matrix_grid(results, output_dir=output_dir)

        # ROC curves grid
        plot_roc_curve_grid(results, output_dir=output_dir)

        # Precision-recall curves grid
        plot_precision_recall_grid(results, output_dir=output_dir)

        # Feature importance visualizations
        logger.info("Creating feature importance visualizations")

        # Temporal importance patterns (with your new changes)
        plot_temporal_importance_patterns(results, output_dir=output_dir)

        # Top neuron importance (with your new changes)
        plot_top_neuron_importance(results, output_dir=output_dir)

        logger.info(f"All visualizations saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error visualizing results: {e}", exc_info=True)


if __name__ == "__main__":
    main()

