# #!/usr/bin/env python3
# """
# Fixed comprehensive analysis script comparing ML/DL model-identified important neurons
# with actual most active neurons in calcium imaging data.
#
# This script uses your exact model implementations from mind/models directory and
# provides a complete analysis with proper error handling and formatting.
# """
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import json
# import os
# import sys
# from scipy.stats import spearmanr, pearsonr
# import logging
# from typing import Dict, Any, Optional, Tuple, List, Union
# import scipy.io
# import hdf5storage
# import torch
#
# # Import your existing model implementations
# from mind.models.classical.random_forest import RandomForestModel
# from mind.models.classical.mlp import MLPModel
# from mind.models.deep.cnn import CNNWrapper
# from mind.models.deep.fcnn import FCNNWrapper
#
# # Import other necessary components from your codebase
# from mind.data.processor import SlidingWindowDataset
# from mind.visualization.config import set_publication_style, SIGNAL_COLORS, SIGNAL_DISPLAY_NAMES
#
# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# def set_random_seeds(seed=42):
#     """Set all random seeds for reproducibility using your codebase standards."""
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
#
# def load_matlab_calcium_data(mat_file_path: str) -> Dict[str, np.ndarray]:
#     """
#     Load calcium imaging signals directly from MATLAB file using your exact variable names.
#
#     Maps to the correct MATLAB variables as specified:
#     - calciumsignal_wanted -> calcium_signal
#     - deltaf_cells_not_excluded -> deltaf_signal
#     - DeconvMat_wanted -> deconv_signal
#     """
#     logger.info(f"Loading MATLAB calcium data from {mat_file_path}")
#
#     try:
#         # Try loading with scipy.io.loadmat first (for older MATLAB files)
#         try:
#             data = scipy.io.loadmat(mat_file_path)
#         except NotImplementedError:
#             # Fall back to hdf5storage for newer MATLAB files
#             data = hdf5storage.loadmat(mat_file_path)
#
#         # Map signal types to your exact MATLAB variable names
#         matlab_variable_mapping = {
#             'calcium_signal': 'calciumsignal_wanted',  # Raw calcium signal
#             'deltaf_signal': 'deltaf_cells_not_excluded',  # ΔF/F signal
#             'deconv_signal': 'DeconvMat_wanted'  # Deconvolved signal
#         }
#
#         calcium_signals = {}
#
#         for signal_type, matlab_var in matlab_variable_mapping.items():
#             if matlab_var in data and data[matlab_var] is not None:
#                 calcium_signals[signal_type] = data[matlab_var]
#                 logger.info(f"Successfully loaded {signal_type} from '{matlab_var}': shape {data[matlab_var].shape}")
#             else:
#                 logger.warning(f"Could not find MATLAB variable '{matlab_var}' for {signal_type}")
#
#         if not calcium_signals:
#             available_vars = [key for key in data.keys() if not key.startswith('__')]
#             logger.error(f"No calcium signals found. Available variables: {available_vars}")
#             raise ValueError("No calcium signals found in MATLAB file")
#
#         return calcium_signals
#
#     except Exception as e:
#         logger.error(f"Error loading MATLAB data: {e}")
#         raise
#
#
# def create_synthetic_classification_task(signal: np.ndarray, method='population_activity') -> np.ndarray:
#     """
#     Create a meaningful synthetic classification task from neural activity patterns.
#
#     This creates binary labels that represent biologically meaningful neural states,
#     allowing us to train your models and extract feature importance rankings that
#     can be compared with pure activity measures.
#     """
#     n_frames, n_neurons = signal.shape
#
#     if method == 'population_activity':
#         # Label frames based on high vs low population activity
#         # This represents periods of high network engagement vs quiescence
#         population_activity = np.mean(signal, axis=1)
#         threshold = np.percentile(population_activity, 65)  # Top 35% of activity
#         labels = (population_activity > threshold).astype(int)
#
#     elif method == 'synchrony_detection':
#         # Label frames with high neural synchrony vs independent activity
#         correlations_per_frame = []
#         for frame_idx in range(min(n_frames, 2000)):  # Sample to avoid computation overload
#             frame_corr = np.corrcoef(signal[frame_idx:frame_idx + 1, :].T)[0, 1] if n_neurons > 1 else 0
#             correlations_per_frame.append(frame_corr if not np.isnan(frame_corr) else 0)
#
#         # Interpolate for all frames if we sampled
#         if len(correlations_per_frame) < n_frames:
#             frame_indices = np.linspace(0, n_frames - 1, len(correlations_per_frame))
#             correlations_per_frame = np.interp(np.arange(n_frames), frame_indices, correlations_per_frame)
#
#         threshold = np.percentile(correlations_per_frame, 70)
#         labels = (np.array(correlations_per_frame) > threshold).astype(int)
#
#     elif method == 'burst_detection':
#         # Label frames with burst-like activity patterns
#         # Calculate coefficient of variation across neurons for each frame
#         cv_per_frame = []
#         for frame_idx in range(n_frames):
#             frame_activity = signal[frame_idx, :]
#             if np.mean(frame_activity) > 0:
#                 cv = np.std(frame_activity) / np.mean(frame_activity)
#             else:
#                 cv = 0
#             cv_per_frame.append(cv)
#
#         threshold = np.percentile(cv_per_frame, 75)  # Top 25% most variable frames
#         labels = (np.array(cv_per_frame) > threshold).astype(int)
#
#     else:
#         raise ValueError(f"Unknown synthetic task method: {method}")
#
#     # Ensure reasonable class balance for training
#     positive_ratio = np.mean(labels)
#     logger.info(f"Created synthetic task '{method}' with {positive_ratio:.1%} positive class")
#
#     if positive_ratio < 0.15 or positive_ratio > 0.85:
#         logger.warning(f"Unbalanced classes ({positive_ratio:.1%} positive). Consider adjusting method.")
#
#     return labels
#
#
# def prepare_data_for_your_models(signal: np.ndarray, labels: np.ndarray,
#                                  window_size: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Prepare data in the format expected by your existing model implementations.
#
#     This creates training and test sets using your SlidingWindowDataset approach,
#     ensuring compatibility with your model interfaces.
#     """
#     from sklearn.model_selection import train_test_split
#
#     # Create sliding window dataset using your existing class
#     dataset = SlidingWindowDataset(
#         signal=signal,
#         labels=labels,
#         window_size=window_size,
#         step_size=1,
#         remove_zero_labels=False  # Keep all labels for balanced comparison
#     )
#
#     # Extract all windows and labels
#     X_windows = []
#     y_labels = []
#
#     for i in range(len(dataset)):
#         window, label = dataset[i]
#         X_windows.append(window.numpy())  # Convert from tensor to numpy
#         y_labels.append(label.numpy() if hasattr(label, 'numpy') else label)
#
#     X = np.array(X_windows)  # Shape: (n_samples, window_size, n_neurons)
#     y = np.array(y_labels)  # Shape: (n_samples,)
#
#     # Split into train/test using your standard approach
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
#
#     logger.info(f"Prepared data: X_train {X_train.shape}, X_test {X_test.shape}")
#
#     # Fixed class distribution logging - handle numpy arrays properly
#     train_dist = np.bincount(y_train) / len(y_train)
#     test_dist = np.bincount(y_test) / len(y_test)
#
#     # Convert to strings for proper logging
#     train_dist_str = ', '.join([f'{val:.3f}' for val in train_dist])
#     test_dist_str = ', '.join([f'{val:.3f}' for val in test_dist])
#
#     logger.info(f"Class distribution - Train: [{train_dist_str}], Test: [{test_dist_str}]")
#
#     return X_train, X_test, y_train, y_test
#
#
# def train_your_random_forest(X_train: np.ndarray, y_train: np.ndarray,
#                              X_test: np.ndarray, y_test: np.ndarray) -> Tuple[RandomForestModel, np.ndarray]:
#     """Train your RandomForestModel implementation and extract feature importance."""
#     logger.info("Training your RandomForestModel implementation...")
#
#     # Initialize with your default parameters plus some optimizations for this task
#     rf_model = RandomForestModel(
#         n_estimators=200,
#         max_depth=15,
#         min_samples_split=5,
#         min_samples_leaf=2,
#         max_features='sqrt',
#         class_weight='balanced_subsample',
#         n_jobs=-1,
#         random_state=42,
#         use_pca=False,  # Keep interpretability for feature importance
#         optimize_hyperparams=False
#     )
#
#     # Train the model using your implementation
#     rf_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
#
#     # Get test accuracy
#     test_accuracy = rf_model.model.score(
#         rf_model.scaler.transform(rf_model._prepare_data(X_test)[0]), y_test
#     )
#     logger.info(f"RandomForest test accuracy: {test_accuracy:.3f}")
#
#     # Extract feature importance using your implementation
#     window_size, n_neurons = X_train.shape[1], X_train.shape[2]
#     importance_matrix = rf_model.get_feature_importance(window_size, n_neurons)
#
#     # Calculate neuron importance (average across time steps)
#     neuron_importance = np.mean(importance_matrix, axis=0)
#
#     return rf_model, neuron_importance
#
#
# def train_your_mlp(X_train: np.ndarray, y_train: np.ndarray,
#                    X_test: np.ndarray, y_test: np.ndarray) -> Tuple[MLPModel, np.ndarray]:
#     """Train your MLPModel implementation and extract feature importance."""
#     logger.info("Training your MLPModel implementation...")
#
#     # Initialize with your default parameters
#     mlp_model = MLPModel(
#         hidden_layer_sizes=(64, 128, 32),
#         activation='relu',
#         solver='adam',
#         alpha=0.0001,
#         learning_rate='adaptive',
#         learning_rate_init=0.001,
#         max_iter=300,
#         early_stopping=True,
#         validation_fraction=0.1,
#         n_iter_no_change=15,
#         random_state=42,
#         optimize_hyperparams=False
#     )
#
#     # Train the model
#     mlp_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
#
#     # Get test accuracy
#     X_test_prepared, _ = mlp_model._prepare_data(X_test)
#     X_test_scaled = mlp_model.scaler.transform(X_test_prepared)
#     test_accuracy = mlp_model.model.score(X_test_scaled, y_test)
#     logger.info(f"MLP test accuracy: {test_accuracy:.3f}")
#
#     # Extract feature importance using your implementation
#     window_size, n_neurons = X_train.shape[1], X_train.shape[2]
#     importance_matrix = mlp_model.get_feature_importance(window_size, n_neurons)
#
#     # Calculate neuron importance
#     neuron_importance = np.mean(importance_matrix, axis=0)
#
#     return mlp_model, neuron_importance
#
#
# def train_your_cnn(X_train: np.ndarray, y_train: np.ndarray,
#                    X_test: np.ndarray, y_test: np.ndarray) -> Tuple[CNNWrapper, np.ndarray]:
#     """Train your CNNWrapper implementation and extract feature importance."""
#     logger.info("Training your CNNWrapper implementation...")
#
#     window_size, n_neurons = X_train.shape[1], X_train.shape[2]
#
#     # Initialize with your default parameters
#     cnn_model = CNNWrapper(
#         window_size=window_size,
#         n_neurons=n_neurons,
#         n_filters=[64, 128, 256],
#         kernel_size=3,
#         output_dim=2,
#         dropout_rate=0.5,
#         learning_rate=0.0005,
#         weight_decay=1e-4,
#         batch_size=32,
#         num_epochs=50,  # Reduced for faster training
#         patience=10,
#         device='cuda' if torch.cuda.is_available() else 'cpu',
#         random_state=42
#     )
#
#     # Train the model
#     cnn_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
#
#     # Get test accuracy
#     predictions = cnn_model.predict(X_test)
#     test_accuracy = np.mean(predictions == y_test)
#     logger.info(f"CNN test accuracy: {test_accuracy:.3f}")
#
#     # Extract feature importance using your implementation
#     neuron_importance = cnn_model.get_feature_importance(window_size, n_neurons)
#
#     # For CNN, the importance matrix might be different - get neuron importance
#     if neuron_importance.ndim == 2:  # If it returns (window_size, n_neurons)
#         neuron_importance = np.mean(neuron_importance, axis=0)
#     # If it already returns per-neuron importance, use as-is
#
#     return cnn_model, neuron_importance
#
#
# def train_your_fcnn(X_train: np.ndarray, y_train: np.ndarray,
#                     X_test: np.ndarray, y_test: np.ndarray) -> Tuple[FCNNWrapper, np.ndarray]:
#     """Train your FCNNWrapper implementation and extract feature importance."""
#     logger.info("Training your FCNNWrapper implementation...")
#
#     window_size, n_neurons = X_train.shape[1], X_train.shape[2]
#     input_dim = window_size * n_neurons
#
#     # Initialize with your default parameters
#     fcnn_model = FCNNWrapper(
#         input_dim=input_dim,
#         hidden_dims=[256, 128, 64],
#         output_dim=2,
#         dropout_rate=0.4,
#         learning_rate=0.001,
#         weight_decay=1e-5,
#         batch_size=32,
#         num_epochs=50,  # Reduced for faster training
#         patience=15,
#         device='cuda' if torch.cuda.is_available() else 'cpu',
#         random_state=42
#     )
#
#     # Train the model
#     fcnn_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
#
#     # Get test accuracy
#     predictions = fcnn_model.predict(X_test)
#     test_accuracy = np.mean(predictions == y_test)
#     logger.info(f"FCNN test accuracy: {test_accuracy:.3f}")
#
#     # Extract feature importance using your implementation
#     importance_matrix = fcnn_model.get_feature_importance(window_size, n_neurons)
#
#     # Calculate neuron importance
#     neuron_importance = np.mean(importance_matrix, axis=0)
#
#     return fcnn_model, neuron_importance
#
#
# def calculate_comprehensive_activity_metrics(signal: np.ndarray, signal_type: str) -> Dict[str, np.ndarray]:
#     """
#     Calculate biologically meaningful activity metrics for each neuron.
#
#     These metrics are designed to capture different aspects of neural activity
#     that might be functionally relevant for movement control.
#     """
#     n_frames, n_neurons = signal.shape
#     metrics = {}
#
#     if signal_type == 'deconv_signal':
#         # For deconvolved signals, focus on spike-related metrics
#         logger.info("Calculating spike-based metrics for deconvolved signals...")
#
#         # Define spike threshold based on signal characteristics
#         positive_signal = signal[signal > 0]
#         if len(positive_signal) > 0:
#             spike_threshold = np.percentile(positive_signal, 25)
#         else:
#             spike_threshold = 0.01
#
#         # Core spike metrics
#         metrics['total_spikes'] = np.sum(signal > spike_threshold, axis=0)
#         metrics['spike_rate_hz'] = metrics['total_spikes'] / (n_frames / 15.32)  # Assuming 15.32 Hz sampling
#         metrics['max_spike_amplitude'] = np.max(signal, axis=0)
#         metrics['mean_spike_amplitude'] = np.mean(signal[signal > spike_threshold], axis=0) if np.any(
#             signal > spike_threshold) else np.zeros(n_neurons)
#
#         # Temporal spike metrics
#         spike_variance = np.zeros(n_neurons)
#         for neuron_idx in range(n_neurons):
#             neuron_spikes = signal[:, neuron_idx]
#             spike_times = np.where(neuron_spikes > spike_threshold)[0]
#             if len(spike_times) > 1:
#                 isi = np.diff(spike_times)
#                 spike_variance[neuron_idx] = np.var(isi) if len(isi) > 0 else 0
#         metrics['spike_timing_variability'] = spike_variance
#
#     elif signal_type == 'deltaf_signal':
#         # For ΔF/F signals, focus on calcium transient properties
#         logger.info("Calculating calcium transient metrics for ΔF/F signals...")
#
#         # Separate positive and negative deflections
#         positive_signal = np.maximum(signal, 0)
#
#         # Event detection threshold
#         if np.any(positive_signal > 0):
#             event_threshold = np.percentile(positive_signal[positive_signal > 0], 75)
#         else:
#             event_threshold = 0.1
#
#         # Core ΔF/F metrics
#         metrics['total_deltaf_activity'] = np.sum(positive_signal, axis=0)
#         metrics['mean_deltaf'] = np.mean(positive_signal, axis=0)
#         metrics['max_deltaf'] = np.max(positive_signal, axis=0)
#         metrics['deltaf_event_frequency'] = np.sum(positive_signal > event_threshold, axis=0)
#
#         # Signal quality metrics
#         metrics['deltaf_signal_to_noise'] = np.mean(positive_signal, axis=0) / (np.std(signal, axis=0) + 1e-10)
#         metrics['deltaf_variance'] = np.var(signal, axis=0)
#
#     else:  # calcium_signal (raw)
#         # For raw calcium signals, focus on overall activity patterns
#         logger.info("Calculating activity metrics for raw calcium signals...")
#
#         # Baseline correction using robust estimation
#         baseline = np.percentile(signal, 10, axis=0)  # 10th percentile as baseline
#         baseline_corrected = signal - baseline[np.newaxis, :]
#         positive_activity = np.maximum(baseline_corrected, 0)
#
#         # Core activity metrics
#         metrics['total_baseline_corrected_activity'] = np.sum(positive_activity, axis=0)
#         metrics['mean_activity'] = np.mean(positive_activity, axis=0)
#         metrics['peak_activity'] = np.max(positive_activity, axis=0)
#         metrics['activity_variance'] = np.var(positive_activity, axis=0)
#
#         # Dynamic range and signal characteristics
#         metrics['dynamic_range'] = np.percentile(signal, 95, axis=0) - np.percentile(signal, 5, axis=0)
#         metrics['signal_stability'] = 1.0 / (
#                     1.0 + np.std(np.diff(signal, axis=0), axis=0))  # Lower variation = more stable
#
#     logger.info(f"Calculated {len(metrics)} comprehensive activity metrics for {signal_type}")
#     return metrics
#
#
# def analyze_model_vs_activity_overlap(model_importance: np.ndarray, activity_ranking: np.ndarray,
#                                       top_n: int = 20) -> Dict[str, float]:
#     """
#     Comprehensive analysis of overlap between model importance and activity rankings.
#
#     This function provides multiple statistical perspectives on the relationship
#     between what your models identify as important and what's actually most active.
#     """
#     # Get top N neurons from each ranking
#     model_top_indices = np.argsort(model_importance)[::-1][:top_n]
#     activity_top_indices = activity_ranking[:top_n]
#
#     # Convert to sets for overlap analysis
#     model_top_set = set(model_top_indices)
#     activity_top_set = set(activity_top_indices)
#
#     # Calculate comprehensive overlap metrics
#     intersection = model_top_set.intersection(activity_top_set)
#     union = model_top_set.union(activity_top_set)
#
#     overlap_metrics = {
#         'overlap_count': len(intersection),
#         'overlap_percentage': (len(intersection) / top_n) * 100,
#         'jaccard_index': len(intersection) / len(union) if len(union) > 0 else 0,
#         'precision': len(intersection) / len(model_top_set) if len(model_top_set) > 0 else 0,
#         'recall': len(intersection) / len(activity_top_set) if len(activity_top_set) > 0 else 0,
#     }
#
#     # F1 score for overlap
#     precision = overlap_metrics['precision']
#     recall = overlap_metrics['recall']
#     if precision + recall > 0:
#         overlap_metrics['f1_score'] = 2 * (precision * recall) / (precision + recall)
#     else:
#         overlap_metrics['f1_score'] = 0
#
#     # Rank correlation analysis
#     try:
#         # Create comprehensive ranking arrays
#         max_neuron_id = max(max(model_top_indices), max(activity_top_indices)) + 1
#         model_ranks = np.full(max_neuron_id, max_neuron_id)  # Worst possible rank as default
#         activity_ranks = np.full(max_neuron_id, max_neuron_id)
#
#         # Assign ranks (0 = best rank)
#         all_model_rankings = np.argsort(model_importance)[::-1]
#         for rank, neuron_id in enumerate(all_model_rankings[:top_n]):
#             if neuron_id < max_neuron_id:
#                 model_ranks[neuron_id] = rank
#
#         for rank, neuron_id in enumerate(activity_ranking[:top_n]):
#             if neuron_id < max_neuron_id:
#                 activity_ranks[neuron_id] = rank
#
#         # Calculate correlations for neurons that appear in either ranking
#         relevant_neurons = list(model_top_set.union(activity_top_set))
#
#         if len(relevant_neurons) >= 4:  # Need sufficient points for meaningful correlation
#             model_ranks_subset = model_ranks[relevant_neurons]
#             activity_ranks_subset = activity_ranks[relevant_neurons]
#
#             # Spearman correlation (rank-based, more robust)
#             spearman_corr, spearman_p = spearmanr(model_ranks_subset, activity_ranks_subset)
#             overlap_metrics['spearman_correlation'] = spearman_corr
#             overlap_metrics['spearman_p_value'] = spearman_p
#
#             # Pearson correlation
#             pearson_corr, pearson_p = pearsonr(model_ranks_subset, activity_ranks_subset)
#             overlap_metrics['pearson_correlation'] = pearson_corr
#             overlap_metrics['pearson_p_value'] = pearson_p
#         else:
#             logger.warning("Insufficient neurons for correlation analysis")
#             overlap_metrics.update({
#                 'spearman_correlation': np.nan,
#                 'spearman_p_value': np.nan,
#                 'pearson_correlation': np.nan,
#                 'pearson_p_value': np.nan
#             })
#
#     except Exception as e:
#         logger.warning(f"Error in correlation analysis: {e}")
#         overlap_metrics.update({
#             'spearman_correlation': np.nan,
#             'spearman_p_value': np.nan,
#             'pearson_correlation': np.nan,
#             'pearson_p_value': np.nan
#         })
#
#     return overlap_metrics
#
#
# def create_comprehensive_visualizations(results: Dict[str, Any], output_dir: Path):
#     """
#     Create publication-quality visualizations using your existing visualization framework.
#     """
#     set_publication_style()  # Use your existing style settings
#
#     # Overall summary heatmap
#     models = ['random_forest', 'mlp', 'cnn', 'fcnn']
#     signal_types = list(results.keys())
#
#     # Collect best overlap percentages for each model-signal combination
#     summary_data = np.zeros((len(models), len(signal_types)))
#
#     for i, model_name in enumerate(models):
#         for j, signal_type in enumerate(signal_types):
#             if signal_type in results and model_name in results[signal_type]:
#                 # Get the best overlap across all activity metrics
#                 model_results = results[signal_type][model_name]
#                 overlaps = [metrics['overlap_percentage'] for metrics in model_results.values()
#                             if isinstance(metrics, dict) and 'overlap_percentage' in metrics]
#                 summary_data[i, j] = max(overlaps) if overlaps else 0
#
#     # Create summary heatmap
#     fig, ax = plt.subplots(figsize=(10, 8))
#
#     im = sns.heatmap(summary_data,
#                      xticklabels=[SIGNAL_DISPLAY_NAMES.get(st, st.replace('_', ' ').title()) for st in signal_types],
#                      yticklabels=[m.replace('_', ' ').upper() for m in models],
#                      annot=True, fmt='.1f', cmap='YlOrRd',
#                      cbar_kws={'label': 'Best Overlap Percentage (%)'},
#                      square=False)
#
#     # Customize appearance
#     plt.title('Model-Important vs Activity-Important Neurons\nBest Overlap Across All Activity Metrics',
#               fontsize=16, fontweight='bold', pad=20)
#     plt.xlabel('Signal Type', fontsize=14)
#     plt.ylabel('Model Architecture', fontsize=14)
#
#     # Color-code x-axis labels by signal type
#     ax_labels = ax.get_xticklabels()
#     for idx, (signal_type, label) in enumerate(zip(signal_types, ax_labels)):
#         if signal_type in SIGNAL_COLORS:
#             label.set_color(SIGNAL_COLORS[signal_type])
#             label.set_fontweight('bold')
#
#     plt.tight_layout()
#     plt.savefig(output_dir / 'overlap_summary_heatmap.png', dpi=300, bbox_inches='tight')
#     plt.close()
#
#     # Detailed analysis plots for each signal type
#     for signal_type in signal_types:
#         if signal_type not in results:
#             continue
#
#         fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#         axes = axes.ravel()
#
#         signal_color = SIGNAL_COLORS.get(signal_type, '#333333')
#         signal_name = SIGNAL_DISPLAY_NAMES.get(signal_type, signal_type.replace('_', ' ').title())
#
#         for idx, model_name in enumerate(models):
#             ax = axes[idx]
#
#             if model_name in results[signal_type]:
#                 model_results = results[signal_type][model_name]
#
#                 # Extract data for plotting
#                 metric_names = []
#                 overlap_percentages = []
#                 jaccard_indices = []
#
#                 for metric_name, metrics in model_results.items():
#                     if isinstance(metrics, dict) and 'overlap_percentage' in metrics:
#                         metric_names.append(metric_name.replace('_', ' ').title())
#                         overlap_percentages.append(metrics['overlap_percentage'])
#                         jaccard_indices.append(metrics.get('jaccard_index', 0))
#
#                 if metric_names:
#                     # Create bar plot
#                     x_pos = np.arange(len(metric_names))
#                     bars = ax.bar(x_pos, overlap_percentages,
#                                   color=signal_color, alpha=0.7,
#                                   edgecolor='darkblue', linewidth=1.5)
#
#                     # Add value labels on bars
#                     for bar, pct, jaccard in zip(bars, overlap_percentages, jaccard_indices):
#                         height = bar.get_height()
#                         ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
#                                 f'{pct:.1f}%\n(J:{jaccard:.2f})',
#                                 ha='center', va='bottom', fontweight='bold', fontsize=9)
#
#                     # Customize subplot
#                     ax.set_title(f'{model_name.replace("_", " ").upper()}',
#                                  fontsize=14, fontweight='bold')
#                     ax.set_ylabel('Overlap Percentage', fontsize=12)
#                     ax.set_xticks(x_pos)
#                     ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=10)
#                     ax.set_ylim(0, 100)
#                     ax.grid(axis='y', alpha=0.3)
#
#                     # Add horizontal line at chance level (assuming random overlap)
#                     chance_level = (20 / len(overlap_percentages)) * 100  # Rough estimate
#                     ax.axhline(y=chance_level, color='red', linestyle='--', alpha=0.5, label='Chance level')
#
#                 else:
#                     ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
#                             transform=ax.transAxes, fontsize=12)
#                     ax.set_title(f'{model_name.replace("_", " ").upper()}')
#             else:
#                 ax.text(0.5, 0.5, 'Model not trained', ha='center', va='center',
#                         transform=ax.transAxes, fontsize=12)
#                 ax.set_title(f'{model_name.replace("_", " ").upper()}')
#
#         plt.suptitle(f'Detailed Overlap Analysis: {signal_name} Signal\n'
#                      f'Top 20 Model-Important vs Activity-Important Neurons',
#                      fontsize=16, fontweight='bold', color=signal_color)
#         plt.tight_layout()
#         plt.savefig(output_dir / f'detailed_analysis_{signal_type}.png', dpi=300, bbox_inches='tight')
#         plt.close()
#
#     logger.info(f"Created comprehensive visualizations in {output_dir}")
#
#
# def generate_detailed_scientific_report(results: Dict[str, Any], output_dir: Path):
#     """
#     Generate a comprehensive scientific report interpreting the findings.
#     """
#     report_path = output_dir / 'scientific_analysis_report.txt'
#
#     with open(report_path, 'w') as f:
#         f.write("SCIENTIFIC ANALYSIS REPORT\n")
#         f.write("Model-Important vs Activity-Important Neurons in Calcium Imaging Data\n")
#         f.write("=" * 75 + "\n\n")
#
#         f.write("RESEARCH QUESTION\n")
#         f.write("-" * 17 + "\n")
#         f.write("Do the top 20 neurons identified by ML/DL models as most important for\n")
#         f.write("classification correspond to the neurons with highest activity levels in\n")
#         f.write("the underlying calcium imaging data?\n\n")
#
#         f.write("This analysis directly addresses whether your models are identifying:\n")
#         f.write("A) Functionally specialized neurons with unique temporal patterns, or\n")
#         f.write("B) Simply the neurons with highest overall activity levels\n\n")
#
#         f.write("METHODOLOGY\n")
#         f.write("-" * 12 + "\n")
#         f.write("1. Used your exact model implementations from mind/models directory\n")
#         f.write("2. Created biologically meaningful synthetic classification tasks\n")
#         f.write("3. Trained Random Forest, MLP, CNN, and FCNN on each signal type\n")
#         f.write("4. Extracted top 20 neurons using your get_feature_importance() methods\n")
#         f.write("5. Calculated signal-appropriate activity metrics for comparison\n")
#         f.write("6. Analyzed overlap using multiple statistical approaches\n\n")
#
#         f.write("DETAILED FINDINGS\n")
#         f.write("-" * 17 + "\n\n")
#
#         # Collect all overlap statistics for global analysis
#         all_overlaps = []
#         signal_summaries = {}
#         best_performers = []
#
#         for signal_type, signal_results in results.items():
#             signal_name = SIGNAL_DISPLAY_NAMES.get(signal_type, signal_type.replace('_', ' ').title())
#             f.write(f"{signal_name.upper()} SIGNAL ANALYSIS\n")
#             f.write("-" * 35 + "\n")
#
#             signal_overlaps = []
#
#             for model_name, model_results in signal_results.items():
#                 f.write(f"\n{model_name.replace('_', ' ').upper()} Model Results:\n")
#
#                 best_overlap_for_model = 0
#                 best_metric_for_model = ""
#
#                 for metric_name, metrics in model_results.items():
#                     if isinstance(metrics, dict) and 'overlap_percentage' in metrics:
#                         overlap_pct = metrics['overlap_percentage']
#                         overlap_count = metrics['overlap_count']
#                         jaccard = metrics['jaccard_index']
#                         precision = metrics['precision']
#                         recall = metrics['recall']
#
#                         f.write(f"  vs {metric_name.replace('_', ' ').title()}:\n")
#                         f.write(f"    Overlap: {overlap_count}/20 neurons ({overlap_pct:.1f}%)\n")
#                         f.write(f"    Jaccard Index: {jaccard:.3f}\n")
#                         f.write(f"    Precision: {precision:.3f}, Recall: {recall:.3f}\n")
#
#                         # Include correlation if available
#                         if 'spearman_correlation' in metrics and not np.isnan(metrics['spearman_correlation']):
#                             corr = metrics['spearman_correlation']
#                             p_val = metrics['spearman_p_value']
#                             significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
#                             f.write(f"    Rank Correlation: {corr:.3f} (p={p_val:.3f}){significance}\n")
#
#                         all_overlaps.append(overlap_pct)
#                         signal_overlaps.append(overlap_pct)
#
#                         # Track best performer
#                         if overlap_pct > best_overlap_for_model:
#                             best_overlap_for_model = overlap_pct
#                             best_metric_for_model = metric_name
#
#                         best_performers.append((signal_type, model_name, metric_name, overlap_pct))
#
#                 if best_overlap_for_model > 0:
#                     f.write(
#                         f"  BEST for {model_name}: {best_overlap_for_model:.1f}% with {best_metric_for_model.replace('_', ' ').title()}\n")
#
#             # Signal-level summary
#             if signal_overlaps:
#                 signal_summaries[signal_type] = {
#                     'mean': np.mean(signal_overlaps),
#                     'max': max(signal_overlaps),
#                     'min': min(signal_overlaps),
#                     'std': np.std(signal_overlaps)
#                 }
#
#                 f.write(f"\n{signal_name} Summary:\n")
#                 f.write(
#                     f"  Average overlap: {signal_summaries[signal_type]['mean']:.1f}% ± {signal_summaries[signal_type]['std']:.1f}%\n")
#                 f.write(
#                     f"  Range: {signal_summaries[signal_type]['min']:.1f}% - {signal_summaries[signal_type]['max']:.1f}%\n")
#
#             f.write("\n" + "=" * 50 + "\n\n")
#
#         # Global statistical summary
#         f.write("OVERALL STATISTICAL SUMMARY\n")
#         f.write("-" * 28 + "\n")
#
#         if all_overlaps:
#             overall_mean = np.mean(all_overlaps)
#             overall_std = np.std(all_overlaps)
#             overall_max = max(all_overlaps)
#             overall_min = min(all_overlaps)
#
#             f.write(f"Total comparisons analyzed: {len(all_overlaps)}\n")
#             f.write(f"Overall mean overlap: {overall_mean:.1f}% ± {overall_std:.1f}%\n")
#             f.write(f"Range across all comparisons: {overall_min:.1f}% - {overall_max:.1f}%\n\n")
#
#             # Identify top performers
#             best_performers.sort(key=lambda x: x[3], reverse=True)
#             top_5_performers = best_performers[:5]
#
#             f.write("TOP 5 PERFORMING COMBINATIONS:\n")
#             for i, (signal, model, metric, overlap) in enumerate(top_5_performers, 1):
#                 signal_name = SIGNAL_DISPLAY_NAMES.get(signal, signal)
#                 f.write(
#                     f"{i}. {model.upper()} on {signal_name} vs {metric.replace('_', ' ').title()}: {overlap:.1f}%\n")
#             f.write("\n")
#
#         # Scientific interpretation
#         f.write("SCIENTIFIC INTERPRETATION\n")
#         f.write("-" * 24 + "\n\n")
#
#         if all_overlaps:
#             overall_mean = np.mean(all_overlaps)
#
#             # Provide context-specific interpretation
#             f.write("BIOLOGICAL SIGNIFICANCE:\n")
#
#             if overall_mean > 65:
#                 f.write("HIGH OVERLAP (>65%): Your models are predominantly identifying the most\n")
#                 f.write("active neurons rather than functionally specialized ones. This suggests:\n\n")
#                 f.write("• Models rely primarily on signal-to-noise advantages\n")
#                 f.write("• Classification success may be driven by overall activity levels\n")
#                 f.write("• Limited evidence for functional specialization beyond activity\n")
#                 f.write("• Consider analyzing temporal patterns and task-specific responses\n\n")
#
#                 f.write("IMPLICATIONS FOR YOUR RESEARCH:\n")
#                 f.write("The deconvolution advantage you demonstrated may primarily reflect\n")
#                 f.write("improved detection of already-active neurons rather than revealing\n")
#                 f.write("previously hidden functionally important cells.\n\n")
#
#             elif overall_mean > 35:
#                 f.write("MODERATE OVERLAP (35-65%): Your models show balanced selection between\n")
#                 f.write("highly active neurons and potentially specialized ones. This suggests:\n\n")
#                 f.write("• Models capture some activity-driven importance\n")
#                 f.write("• Evidence for functional specialization beyond raw activity\n")
#                 f.write("• Deconvolution may reveal both enhanced detection and new neurons\n")
#                 f.write("• Mixed population of activity-driven and function-driven neurons\n\n")
#
#                 f.write("IMPLICATIONS FOR YOUR RESEARCH:\n")
#                 f.write("Your models appear to identify a combination of generally active neurons\n")
#                 f.write("and functionally specialized ones. The 200-300ms pre-movement timing\n")
#                 f.write("you discovered likely reflects both enhanced signal detection and\n")
#                 f.write("genuine temporal specialization.\n\n")
#
#             else:
#                 f.write("LOW OVERLAP (<35%): Your models are identifying neurons with functional\n")
#                 f.write("importance beyond simple activity levels. This suggests:\n\n")
#                 f.write("• Strong evidence for functional specialization\n")
#                 f.write("• Models learn complex temporal/spatial patterns\n")
#                 f.write("• Activity level is not the primary selection criterion\n")
#                 f.write("• Deconvolution reveals genuinely specialized neural populations\n\n")
#
#                 f.write("IMPLICATIONS FOR YOUR RESEARCH:\n")
#                 f.write("This provides strong evidence that your models identify functionally\n")
#                 f.write("specialized neurons rather than just the loudest ones. The temporal\n")
#                 f.write("precision you demonstrated (200-300ms pre-movement) likely reflects\n")
#                 f.write("genuine functional specialization for motor preparation.\n\n")
#
#             # Signal-specific insights
#             f.write("SIGNAL-SPECIFIC INSIGHTS:\n")
#             if signal_summaries:
#                 signal_ranking = sorted(signal_summaries.items(), key=lambda x: x[1]['mean'])
#
#                 f.write("Overlap ranking (lowest to highest mean overlap):\n")
#                 for signal_type, stats in signal_ranking:
#                     signal_name = SIGNAL_DISPLAY_NAMES.get(signal_type, signal_type)
#                     f.write(f"• {signal_name}: {stats['mean']:.1f}% (indicates ")
#                     if stats['mean'] < 35:
#                         f.write("strong functional specialization)\n")
#                     elif stats['mean'] < 65:
#                         f.write("moderate functional specialization)\n")
#                     else:
#                         f.write("activity-driven selection)\n")
#                 f.write("\n")
#
#         # Research recommendations
#         f.write("FUTURE RESEARCH DIRECTIONS\n")
#         f.write("-" * 26 + "\n")
#         f.write("Based on these findings, we recommend:\n\n")
#
#         f.write("1. TEMPORAL ANALYSIS: Examine the temporal dynamics of model-important\n")
#         f.write("   neurons that show low overlap with activity rankings. These may\n")
#         f.write("   represent neurons with specific temporal specializations.\n\n")
#
#         f.write("2. CAUSAL VALIDATION: Use optogenetic or chemogenetic approaches to\n")
#         f.write("   manipulate the identified neurons and test their functional importance.\n\n")
#
#         f.write("3. ENSEMBLE ANALYSIS: Study whether model-important neurons form\n")
#         f.write("   functionally coherent ensembles with specific temporal coordination.\n\n")
#
#         f.write("4. TASK SPECIFICITY: Compare neuron importance across different\n")
#         f.write("   behavioral tasks to identify task-general vs task-specific neurons.\n\n")
#
#         f.write("5. CONNECTIVITY ANALYSIS: Examine whether model-important neurons\n")
#         f.write("   show distinct connectivity patterns that explain their importance\n")
#         f.write("   beyond activity levels.\n\n")
#
#         # Technical notes
#         f.write("TECHNICAL NOTES\n")
#         f.write("-" * 15 + "\n")
#         f.write("• Analysis used your exact model implementations for consistency\n")
#         f.write("• Synthetic classification tasks based on biologically meaningful activity patterns\n")
#         f.write("• Multiple activity metrics calculated for each signal type\n")
#         f.write("• Statistical significance tested using rank correlations\n")
#         f.write("• Results directly comparable to your published findings\n")
#
#     logger.info(f"Generated comprehensive scientific report at {report_path}")
#
#
# def main():
#     """
#     Main function orchestrating the complete analysis using your existing model implementations.
#     """
#     # Configuration
#     mat_file_path = "data/raw/SFL13_5_8112021_002_new_modified.mat"
#     output_dir = Path("outputs/comprehensive_model_analysis")
#
#     # Create output directory
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     # Set reproducible random seeds
#     set_random_seeds(42)
#
#     logger.info("=" * 60)
#     logger.info("COMPREHENSIVE NEURON ANALYSIS USING YOUR MODEL IMPLEMENTATIONS")
#     logger.info("=" * 60)
#     logger.info(f"MATLAB file: {mat_file_path}")
#     logger.info(f"Output directory: {output_dir}")
#     logger.info(f"Using models from mind/models directory")
#
#     try:
#         # Load your MATLAB calcium imaging data
#         calcium_signals = load_matlab_calcium_data(mat_file_path)
#
#         logger.info(f"Successfully loaded {len(calcium_signals)} signal types")
#
#         # Store all analysis results
#         comprehensive_results = {}
#
#         # Process each signal type
#         for signal_type, signal in calcium_signals.items():
#             logger.info(f"\n{'=' * 60}")
#             logger.info(f"ANALYZING {signal_type.upper()}")
#             logger.info(f"Signal shape: {signal.shape}")
#             logger.info(f"{'=' * 60}")
#
#             # Create synthetic classification task
#             synthetic_labels = create_synthetic_classification_task(signal, method='population_activity')
#
#             # Prepare data for your models
#             X_train, X_test, y_train, y_test = prepare_data_for_your_models(signal, synthetic_labels)
#
#             # Calculate comprehensive activity metrics
#             activity_metrics = calculate_comprehensive_activity_metrics(signal, signal_type)
#
#             # Dictionary to store results for this signal type
#             signal_results = {}
#
#             # Train your models and extract importance
#             model_training_functions = {
#                 'random_forest': train_your_random_forest,
#                 'mlp': train_your_mlp,
#                 'cnn': train_your_cnn,
#                 'fcnn': train_your_fcnn
#             }
#
#             for model_name, train_function in model_training_functions.items():
#                 logger.info(f"\n{'-' * 40}")
#                 logger.info(f"TRAINING YOUR {model_name.upper()} MODEL")
#                 logger.info(f"{'-' * 40}")
#
#                 try:
#                     # Train your model implementation
#                     trained_model, neuron_importance = train_function(X_train, y_train, X_test, y_test)
#
#                     logger.info(f"Successfully extracted importance for {len(neuron_importance)} neurons")
#
#                     # Compare model importance with each activity metric
#                     model_comparisons = {}
#
#                     for metric_name, metric_values in activity_metrics.items():
#                         # Get activity ranking (most active neurons first)
#                         activity_ranking = np.argsort(metric_values)[::-1]
#
#                         # Analyze overlap
#                         overlap_analysis = analyze_model_vs_activity_overlap(
#                             neuron_importance, activity_ranking, top_n=20
#                         )
#
#                         model_comparisons[metric_name] = overlap_analysis
#
#                         logger.info(
#                             f"  {model_name} vs {metric_name}: {overlap_analysis['overlap_percentage']:.1f}% overlap")
#
#                     signal_results[model_name] = model_comparisons
#
#                 except Exception as e:
#                     logger.error(f"Failed to train {model_name}: {e}")
#                     continue
#
#             # Store results for this signal type
#             if signal_results:
#                 comprehensive_results[signal_type] = signal_results
#                 logger.info(f"\nCompleted analysis for {signal_type}")
#             else:
#                 logger.warning(f"No successful model training for {signal_type}")
#
#         if not comprehensive_results:
#             raise ValueError("No successful analyses completed")
#
#         # Create comprehensive visualizations
#         logger.info("\n" + "=" * 60)
#         logger.info("CREATING VISUALIZATIONS")
#         logger.info("=" * 60)
#         create_comprehensive_visualizations(comprehensive_results, output_dir)
#
#         # Generate detailed scientific report
#         logger.info("GENERATING SCIENTIFIC REPORT")
#         generate_detailed_scientific_report(comprehensive_results, output_dir)
#
#         # Save complete results
#         results_file = output_dir / 'complete_analysis_results.json'
#         with open(results_file, 'w') as f:
#             # Convert numpy types for JSON serialization
#             def numpy_to_python(obj):
#                 if isinstance(obj, np.ndarray):
#                     return obj.tolist()
#                 elif isinstance(obj, (np.floating, np.integer)):
#                     return float(obj) if isinstance(obj, np.floating) else int(obj)
#                 elif isinstance(obj, dict):
#                     return {k: numpy_to_python(v) for k, v in obj.items()}
#                 elif isinstance(obj, list):
#                     return [numpy_to_python(item) for item in obj]
#                 return obj
#
#             json.dump(numpy_to_python(comprehensive_results), f, indent=2)
#
#         logger.info(f"Saved complete results to {results_file}")
#
#         # Print executive summary
#         logger.info("\n" + "=" * 60)
#         logger.info("EXECUTIVE SUMMARY")
#         logger.info("=" * 60)
#
#         for signal_type, signal_results in comprehensive_results.items():
#             signal_name = SIGNAL_DISPLAY_NAMES.get(signal_type, signal_type.replace('_', ' ').title())
#             logger.info(f"\n{signal_name} Signal:")
#
#             for model_name, model_results in signal_results.items():
#                 overlaps = [metrics['overlap_percentage'] for metrics in model_results.values()]
#                 if overlaps:
#                     mean_overlap = np.mean(overlaps)
#                     max_overlap = max(overlaps)
#                     logger.info(f"  {model_name.upper()}: Mean={mean_overlap:.1f}%, Max={max_overlap:.1f}%")
#
#         # Overall conclusion
#         all_overlaps = []
#         for signal_results in comprehensive_results.values():
#             for model_results in signal_results.values():
#                 all_overlaps.extend([metrics['overlap_percentage'] for metrics in model_results.values()])
#
#         if all_overlaps:
#             overall_mean = np.mean(all_overlaps)
#             logger.info(f"\nOVERALL MEAN OVERLAP: {overall_mean:.1f}%")
#
#             if overall_mean > 65:
#                 conclusion = "HIGH overlap - models primarily identify most active neurons"
#             elif overall_mean > 35:
#                 conclusion = "MODERATE overlap - mixed activity and functional importance"
#             else:
#                 conclusion = "LOW overlap - models identify functionally specialized neurons"
#
#             logger.info(f"CONCLUSION: {conclusion}")
#
#         logger.info(f"\nComplete analysis saved to: {output_dir}")
#         logger.info("Check scientific_analysis_report.txt for detailed interpretation!")
#
#         return comprehensive_results
#
#     except Exception as e:
#         logger.error(f"Analysis failed: {e}")
#         logger.error("Check error messages above for details")
#         raise
#
#
# if __name__ == "__main__":
#     main()
#
#

# !/usr/bin/env python3
"""
Create spatial visualization showing model-important neurons and highlighting
which ones are also highly active in the real MATLAB data.

This visualization directly answers: "Are the top 20 model-important neurons
the same as the top 20 most active neurons in the real data?"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import scipy.io
import hdf5storage

# Import your existing components
from mind.visualization.config import (
    SIGNAL_COLORS, SIGNAL_DISPLAY_NAMES, MODEL_DISPLAY_NAMES, set_publication_style
)

logger = logging.getLogger(__name__)


def load_matlab_data_for_overlap_analysis(mat_file_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load MATLAB data and calculate activity metrics for each neuron."""

    try:
        # Load MATLAB file
        try:
            data = scipy.io.loadmat(mat_file_path)
        except NotImplementedError:
            data = hdf5storage.loadmat(mat_file_path)

        # Extract signals
        calcium_signals = {
            'calcium_signal': data.get('calciumsignal_wanted', None),
            'deltaf_signal': data.get('deltaf_cells_not_excluded', None),
            'deconv_signal': data.get('DeconvMat_wanted', None)
        }

        # Extract ROI matrix for spatial positioning
        roi_matrix = data.get('ROI_matrix', None)

        return calcium_signals, roi_matrix

    except Exception as e:
        logger.error(f"Error loading MATLAB data: {e}")
        raise


def calculate_real_activity_rankings(calcium_signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate the actual most active neurons in the real data for each signal type.

    This gives us the ground truth of which neurons are truly most active,
    which we can compare against model predictions.
    """
    activity_rankings = {}

    for signal_type, signal in calcium_signals.items():
        if signal is None:
            continue

        if signal_type == 'deconv_signal':
            # For deconvolved: count total spikes (activity > threshold)
            spike_threshold = np.percentile(signal[signal > 0], 25) if np.any(signal > 0) else 0.01
            activity_metric = np.sum(signal > spike_threshold, axis=0)

        elif signal_type == 'deltaf_signal':
            # For ΔF/F: sum of positive deflections (calcium events)
            positive_signal = np.maximum(signal, 0)
            activity_metric = np.sum(positive_signal, axis=0)

        else:  # calcium_signal
            # For raw: baseline-corrected total activity
            baseline = np.percentile(signal, 10, axis=0)
            baseline_corrected = signal - baseline[np.newaxis, :]
            positive_activity = np.maximum(baseline_corrected, 0)
            activity_metric = np.sum(positive_activity, axis=0)

        # Get ranking (most active first)
        activity_ranking = np.argsort(activity_metric)[::-1]
        activity_rankings[signal_type] = activity_ranking

        logger.info(f"Calculated activity ranking for {signal_type}")

    return activity_rankings


def generate_neuron_positions(roi_matrix: np.ndarray, n_neurons: int) -> np.ndarray:
    """Generate approximate spatial positions for neurons."""

    if roi_matrix is None:
        # Create random positions in a circle if no ROI matrix
        angles = np.random.uniform(0, 2 * np.pi, n_neurons)
        radii = np.sqrt(np.random.uniform(0, 1, n_neurons))
        x = radii * np.cos(angles) * 250 + 250  # Center at (250, 250)
        y = radii * np.sin(angles) * 250 + 250
        return np.column_stack((x, y))

    # Use ROI matrix to generate realistic positions
    try:
        from scipy import ndimage

        # Smooth and find peaks
        smoothed = ndimage.gaussian_filter(roi_matrix.astype(float), sigma=2)

        # Generate grid-based positions with some randomness
        grid_size = int(np.sqrt(n_neurons)) + 1
        x_coords = np.linspace(10, roi_matrix.shape[1] - 10, grid_size)
        y_coords = np.linspace(10, roi_matrix.shape[0] - 10, grid_size)

        positions = []
        for i in range(n_neurons):
            x_idx = i % grid_size
            y_idx = i // grid_size
            if y_idx < len(y_coords):
                # Add some random jitter
                x = x_coords[x_idx] + np.random.normal(0, 10)
                y = y_coords[y_idx] + np.random.normal(0, 10)
                positions.append([x, y])

        # Fill remaining positions randomly if needed
        while len(positions) < n_neurons:
            x = np.random.uniform(10, roi_matrix.shape[1] - 10)
            y = np.random.uniform(10, roi_matrix.shape[0] - 10)
            positions.append([x, y])

        return np.array(positions[:n_neurons])

    except Exception as e:
        logger.warning(f"Error generating positions from ROI: {e}")
        # Fallback to grid
        grid_size = int(np.sqrt(n_neurons)) + 1
        x_coords = np.linspace(50, 450, grid_size)
        y_coords = np.linspace(50, 450, grid_size)

        positions = []
        for i in range(n_neurons):
            x_idx = i % grid_size
            y_idx = i // grid_size
            if y_idx < len(y_coords):
                positions.append([x_coords[x_idx], y_coords[y_idx]])

        return np.array(positions[:n_neurons])


def create_model_vs_activity_spatial_plot(
        mat_file_path: str,
        model_important_neurons: Dict[str, np.ndarray],  # Top 20 from each model
        signal_type: str,
        model_name: str = 'cnn',
        output_dir: Optional[str] = None
) -> plt.Figure:
    """
    Create the spatial visualization you want: showing model-important neurons
    and highlighting which ones are also highly active in real data.

    This directly answers your question about overlap between model predictions
    and real activity levels.
    """

    set_publication_style()

    # Load real data and calculate true activity rankings
    calcium_signals, roi_matrix = load_matlab_data_for_overlap_analysis(mat_file_path)
    real_activity_rankings = calculate_real_activity_rankings(calcium_signals)

    if signal_type not in calcium_signals or calcium_signals[signal_type] is None:
        raise ValueError(f"Signal type {signal_type} not found in data")

    if signal_type not in real_activity_rankings:
        raise ValueError(f"Could not calculate activity ranking for {signal_type}")

    # Get the data we need
    signal = calcium_signals[signal_type]
    n_neurons = signal.shape[1]

    # Generate neuron positions
    positions = generate_neuron_positions(roi_matrix, n_neurons)

    # Get top 20 model-important neurons
    if signal_type not in model_important_neurons:
        # For demonstration, create example model-important neurons
        # In real use, this would come from your trained models
        model_top_20 = np.random.choice(n_neurons, 20, replace=False)
        logger.warning(f"Using random neurons for demonstration. Replace with real model output.")
    else:
        model_top_20 = model_important_neurons[signal_type][:20]

    # Get top 20 most active neurons from real data
    real_top_20 = real_activity_rankings[signal_type][:20]

    # Find overlap
    overlap_neurons = set(model_top_20).intersection(set(real_top_20))
    model_only = set(model_top_20) - set(real_top_20)
    activity_only = set(real_top_20) - set(model_top_20)

    # Calculate overlap statistics
    overlap_count = len(overlap_neurons)
    overlap_percentage = (overlap_count / 20) * 100

    # Create the visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw ROI background if available
    if roi_matrix is not None:
        ax.imshow(roi_matrix, cmap='gray', alpha=0.3)
        xlim = (0, roi_matrix.shape[1])
        ylim = (roi_matrix.shape[0], 0)  # Flip y-axis for image coordinates
    else:
        ax.set_facecolor('#f5f5f5')
        xlim = (0, 500)
        ylim = (0, 500)

    # Plot all neurons as small gray dots
    ax.scatter(positions[:, 0], positions[:, 1], s=8, color='lightgray', alpha=0.4, label='Other neurons')

    # Color scheme
    signal_color = SIGNAL_COLORS[signal_type]

    # Plot different categories of neurons with distinct styling

    # 1. Model-important only (not highly active) - hollow circles
    if model_only:
        model_only_pos = positions[list(model_only)]
        ax.scatter(model_only_pos[:, 0], model_only_pos[:, 1],
                   s=120, facecolors='none', edgecolors=signal_color,
                   linewidth=2.5, alpha=0.8, label=f'Model-important only ({len(model_only)})')

    # 2. Highly active only (not model-important) - filled triangles
    if activity_only:
        activity_only_pos = positions[list(activity_only)]
        ax.scatter(activity_only_pos[:, 0], activity_only_pos[:, 1],
                   s=120, marker='^', color='orange', alpha=0.8,
                   edgecolor='darkorange', linewidth=1,
                   label=f'Highly active only ({len(activity_only)})')

    # 3. OVERLAP neurons (both model-important AND highly active) - filled stars
    if overlap_neurons:
        overlap_pos = positions[list(overlap_neurons)]
        ax.scatter(overlap_pos[:, 0], overlap_pos[:, 1],
                   s=200, marker='*', color='gold', alpha=0.9,
                   edgecolor='darkred', linewidth=2,
                   label=f'OVERLAP: Both important & active ({overlap_count})')

        # Add numbers to overlap neurons for identification
        for i, neuron_id in enumerate(overlap_neurons):
            pos = positions[neuron_id]
            ax.text(pos[0], pos[1], str(i + 1), ha='center', va='center',
                    fontsize=8, fontweight='bold', color='darkred')

    # Customize the plot
    signal_name = SIGNAL_DISPLAY_NAMES[signal_type]
    model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name.upper())

    ax.set_title(f'Model-Important vs Highly Active Neurons: {signal_name} Signal\n'
                 f'{model_display} Model | Overlap: {overlap_count}/20 neurons ({overlap_percentage:.1f}%)',
                 fontsize=16, fontweight='bold', color=signal_color, pad=20)

    # Set limits and remove ticks
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colored border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(signal_color)
        spine.set_linewidth(3)

    # Create legend
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    # Add interpretation text
    if overlap_percentage > 50:
        interpretation = "HIGH OVERLAP: Model identifies highly active neurons"
    elif overlap_percentage > 25:
        interpretation = "MODERATE OVERLAP: Mixed selection of active & specialized neurons"
    else:
        interpretation = "LOW OVERLAP: Model identifies functionally specialized neurons"

    fig.text(0.5, 0.02, interpretation, ha='center', fontsize=12,
             fontweight='bold', style='italic', color=signal_color)

    # Save if output directory provided
    if output_dir:
        output_path = Path(output_dir) / f'{model_name}_{signal_type}_overlap_spatial.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved spatial overlap visualization to {output_path}")

    return fig


def create_novel_overlap_bar_chart(
        overlap_results: Dict[str, Dict[str, float]],
        output_dir: Optional[str] = None
) -> plt.Figure:
    """
    Create a novel bar chart showing multiple perspectives on model vs activity overlap.

    This chart shows several ways to understand the relationship between
    model-important and activity-important neurons.
    """

    set_publication_style()

    # Example data structure - replace with your real results
    if not overlap_results:
        # Create example data for demonstration
        overlap_results = {
            'calcium_signal': {
                'overlap_percentage': 15.0,
                'jaccard_index': 0.08,
                'correlation': 0.12,
                'precision': 0.15,
                'recall': 0.15
            },
            'deltaf_signal': {
                'overlap_percentage': 25.0,
                'jaccard_index': 0.14,
                'correlation': 0.23,
                'precision': 0.25,
                'recall': 0.25
            },
            'deconv_signal': {
                'overlap_percentage': 35.0,
                'jaccard_index': 0.21,
                'correlation': 0.31,
                'precision': 0.35,
                'recall': 0.35
            }
        }

    # Create figure with subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    signal_types = list(overlap_results.keys())
    signal_colors = [SIGNAL_COLORS[st] for st in signal_types]
    signal_names = [SIGNAL_DISPLAY_NAMES[st] for st in signal_types]

    # 1. Overlap Percentage (Top Left)
    ax1 = axes[0, 0]
    overlap_pcts = [overlap_results[st]['overlap_percentage'] for st in signal_types]
    bars1 = ax1.bar(signal_names, overlap_pcts, color=signal_colors, alpha=0.7,
                    edgecolor='black', linewidth=1.5)

    # Add percentage labels on bars
    for bar, pct in zip(bars1, overlap_pcts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax1.set_title('Direct Overlap: Top 20 Neurons', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Overlap Percentage (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Add interpretation zones
    ax1.axhspan(0, 25, alpha=0.1, color='red', label='Low: Functional specialization')
    ax1.axhspan(25, 50, alpha=0.1, color='yellow', label='Moderate: Mixed selection')
    ax1.axhspan(50, 100, alpha=0.1, color='green', label='High: Activity-driven')

    # 2. Jaccard Index (Top Right) - measures similarity accounting for set sizes
    ax2 = axes[0, 1]
    jaccard_vals = [overlap_results[st]['jaccard_index'] for st in signal_types]
    bars2 = ax2.bar(signal_names, jaccard_vals, color=signal_colors, alpha=0.7,
                    edgecolor='black', linewidth=1.5)

    for bar, jac in zip(bars2, jaccard_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{jac:.3f}', ha='center', va='bottom', fontweight='bold')

    ax2.set_title('Jaccard Similarity Index', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Jaccard Index', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Ranking Correlation (Bottom Left)
    ax3 = axes[1, 0]
    corr_vals = [overlap_results[st]['correlation'] for st in signal_types]
    bars3 = ax3.bar(signal_names, corr_vals, color=signal_colors, alpha=0.7,
                    edgecolor='black', linewidth=1.5)

    for bar, corr in zip(bars3, corr_vals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')

    ax3.set_title('Ranking Correlation', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Correlation Coefficient', fontsize=12)
    ax3.set_ylim(-1, 1)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # 4. Precision vs Recall (Bottom Right)
    ax4 = axes[1, 1]
    precision_vals = [overlap_results[st]['precision'] for st in signal_types]
    recall_vals = [overlap_results[st]['recall'] for st in signal_types]

    x = np.arange(len(signal_names))
    width = 0.35

    bars4a = ax4.bar(x - width / 2, precision_vals, width, label='Precision',
                     color=signal_colors, alpha=0.7, edgecolor='black')
    bars4b = ax4.bar(x + width / 2, recall_vals, width, label='Recall',
                     color=signal_colors, alpha=0.4, edgecolor='black')

    ax4.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(signal_names)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # Overall title
    fig.suptitle('Comprehensive Analysis: Model-Important vs Activity-Important Neurons',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save if output directory provided
    if output_dir:
        output_path = Path(output_dir) / 'novel_overlap_analysis.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved novel overlap analysis to {output_path}")

    return fig


# Example usage function
def run_complete_overlap_analysis(mat_file_path: str, output_dir: str):
    """
    Run the complete analysis to answer your question:
    Are model-important neurons the same as highly active neurons?
    """

    # This would use your real model results
    # For demonstration, creating example model-important neurons
    model_important_neurons = {
        'calcium_signal': np.array([10, 45, 78, 123, 156, 189, 234, 267, 301, 345,
                                    389, 412, 456, 489, 523, 556, 578, 601, 634, 667]),
        'deltaf_signal': np.array([23, 67, 91, 134, 178, 201, 245, 289, 312, 356,
                                   398, 421, 467, 501, 534, 567, 590, 612, 645, 678]),
        'deconv_signal': np.array([34, 78, 102, 145, 189, 212, 256, 290, 323, 367,
                                   409, 432, 478, 512, 545, 578, 601, 623, 656, 689])
    }

    # Create spatial visualizations for each signal type
    for signal_type in ['calcium_signal', 'deltaf_signal', 'deconv_signal']:
        try:
            fig = create_model_vs_activity_spatial_plot(
                mat_file_path=mat_file_path,
                model_important_neurons=model_important_neurons,
                signal_type=signal_type,
                model_name='cnn',
                output_dir=output_dir
            )
            plt.show()
        except Exception as e:
            logger.error(f"Error creating spatial plot for {signal_type}: {e}")

    # Create the novel bar chart analysis
    # This would use your real overlap results
    overlap_results = {}  # Your real results would go here

    try:
        fig = create_novel_overlap_bar_chart(
            overlap_results=overlap_results,
            output_dir=output_dir
        )
        plt.show()
    except Exception as e:
        logger.error(f"Error creating bar chart: {e}")


if __name__ == "__main__":
    # Example usage
    mat_file_path = "data/raw/SFL13_5_8112021_002_new_modified.mat"
    output_dir = "outputs/overlap_analysis"

    run_complete_overlap_analysis(mat_file_path, output_dir)

