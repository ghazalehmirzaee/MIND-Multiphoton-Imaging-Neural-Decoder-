"""
Fully corrected training pipeline with FCNN bug completely resolved.

The key insight is that different neural network architectures expect input dimensions
to be specified in different ways:
- CNNs need separate window_size and n_neurons to maintain spatial structure
- FCNNs need input_dim = window_size * n_neurons since they flatten everything
"""
import time
import numpy as np
import torch
import json
from pathlib import Path
import logging

from mind.evaluation.metrics import evaluate_model
from mind.evaluation.feature_importance import extract_feature_importance

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.

    Think of this as ensuring that every "random" choice our algorithms make
    will be the same across different runs, like using the same sequence of
    dice rolls every time we play a game.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_model(model_type, model_params, datasets, signal_type, window_size, n_neurons,
                output_dir, device="cuda", optimize_hyperparams=False):
    """
    Train a model and return comprehensive results.

    This function acts like a universal training conductor that can work with different
    types of machine learning "orchestras" (models), each with their own specific
    requirements and ways of interpreting the same musical score (data).

    Parameters
    ----------
    model_type : str
        Type of model to train ('random_forest', 'svm', 'mlp', 'fcnn', 'cnn')
    model_params : dict
        Parameters for the model (like tuning instructions for our orchestra)
    datasets : dict
        Dictionary containing train/val/test datasets for each signal type
    signal_type : str
        Type of signal to use ('calcium_signal', 'deltaf_signal', 'deconv_signal')
    window_size : int
        Size of the sliding window (temporal context for each prediction)
    n_neurons : int
        Number of neurons in the data (spatial dimension)
    output_dir : str
        Directory to save results
    device : str, optional
        Device to use for training ('cuda' or 'cpu'), by default "cuda"
    optimize_hyperparams : bool, optional
        Whether to optimize hyperparameters (handled by individual models), by default False

    Returns
    -------
    dict
        Dictionary containing training results and metrics
    """
    logger.info(f"Training {model_type} on {signal_type}")
    set_seed(model_params.get('random_state', 42))

    # Extract data from the PyTorch datasets
    # This is like gathering all our sheet music and organizing it for the orchestra
    train_dataset = datasets[signal_type]['train']
    val_dataset = datasets[signal_type]['val']
    test_dataset = datasets[signal_type]['test']

    # Convert datasets to arrays that our models can understand
    # PyTorch datasets store data as (features, labels) tuples, so we separate them
    X_train = torch.stack([x for x, _ in train_dataset])
    y_train = torch.tensor([y.item() for _, y in train_dataset])
    X_val = torch.stack([x for x, _ in val_dataset])
    y_val = torch.tensor([y.item() for _, y in val_dataset])
    X_test = torch.stack([x for x, _ in test_dataset])
    y_test = torch.tensor([y.item() for _, y in test_dataset])

    # Pass the optimize_hyperparams parameter to models that support it
    # This is like telling each orchestra section whether they should tune their instruments
    if 'optimize_hyperparams' not in model_params:
        model_params['optimize_hyperparams'] = optimize_hyperparams

    # Initialize the appropriate model based on model_type
    # Each model type is like a different type of musical ensemble with unique requirements
    if model_type == 'random_forest':
        from mind.models.classical.random_forest import RandomForestModel
        # Random Forest works like a committee of decision trees voting on the answer
        model = RandomForestModel(**model_params)

    elif model_type == 'svm':
        from mind.models.classical.svm import SVMModel
        # SVM finds the best boundary to separate different classes of data
        model = SVMModel(**model_params)

    elif model_type == 'mlp':
        from mind.models.classical.mlp import MLPModel
        # MLP is a classical neural network with fully connected layers
        model = MLPModel(**model_params)

    elif model_type == 'fcnn':
        from mind.models.deep.fcnn import FCNNWrapper
        # FCNN is a deep learning version of MLP with modern training techniques

        # Remove optimize_hyperparams since deep learning models don't use this parameter
        fcnn_params = {k: v for k, v in model_params.items() if k != 'optimize_hyperparams'}

        # CRITICAL FIX: FCNNs need the total flattened input dimension
        # Think of this like telling the FCNN: "You'll receive this many individual numbers"
        # instead of "You'll receive a matrix with these dimensions"
        input_dim = window_size * n_neurons

        # Create the model with the correct input dimension specification
        model = FCNNWrapper(input_dim=input_dim, device=device, **fcnn_params)

    elif model_type == 'cnn':
        from mind.models.deep.cnn import CNNWrapper
        # CNN maintains spatial structure and can learn patterns across neurons and time

        # Remove optimize_hyperparams since deep learning models don't use this parameter
        cnn_params = {k: v for k, v in model_params.items() if k != 'optimize_hyperparams'}

        # CNNs need separate dimensions because they care about spatial relationships
        # Think of this like telling the CNN: "Here's a movie (time) showing multiple actors (neurons)"
        model = CNNWrapper(window_size=window_size, n_neurons=n_neurons, device=device, **cnn_params)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train the model and measure how long it takes
    # This is like recording both the performance and the rehearsal time
    start_time = time.time()
    try:
        model.fit(X_train, y_train, X_val, y_val)
        train_time = time.time() - start_time
        logger.info(f"Training completed successfully in {train_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Training failed for {model_type} on {signal_type}: {e}")
        # Return a failure result instead of crashing the entire experiment
        return {
            'metadata': {
                'model_name': model_type,
                'signal_type': signal_type,
                'window_size': window_size,
                'n_neurons': n_neurons,
                'training_failed': True,
                'error_message': str(e)
            }
        }

    # Evaluate the trained model on test data
    # This is like having the orchestra perform for an audience that didn't hear the rehearsals
    try:
        # Add diagnostic information about model capabilities
        has_predict_proba = hasattr(model, 'predict_proba')
        logger.info(f"Model {model_type} has predict_proba: {has_predict_proba}")

        if has_predict_proba:
            try:
                # Test probability prediction on a small sample
                test_proba = model.predict_proba(X_test[:5])
                logger.info(f"Probability prediction test successful, shape: {test_proba.shape}")
            except Exception as prob_e:
                logger.warning(f"Probability prediction test failed: {prob_e}")

        results = evaluate_model(model, X_test, y_test)
        results['train_time'] = train_time

        # Log what curve data was generated
        curve_data_keys = results.get('curve_data', {}).keys()
        logger.info(f"Evaluation completed - Accuracy: {results['metrics']['accuracy']:.3f}")
        logger.info(f"Generated curve data: {list(curve_data_keys)}")

    except Exception as e:
        logger.error(f"Evaluation failed for {model_type} on {signal_type}: {e}")
        return {
            'metadata': {
                'model_name': model_type,
                'signal_type': signal_type,
                'window_size': window_size,
                'n_neurons': n_neurons,
                'evaluation_failed': True,
                'error_message': str(e)
            }
        }

    # Extract feature importance to understand what the model learned
    # This tells us which parts of our data the model found most useful for making decisions
    try:
        importance_matrix = extract_feature_importance(model, window_size, n_neurons)
        results['importance_summary'] = {
            'importance_matrix': importance_matrix.tolist(),
            'temporal_importance': importance_matrix.mean(axis=1).tolist(),  # Importance across time
            'neuron_importance': importance_matrix.mean(axis=0).tolist(),    # Importance across neurons
            'top_neuron_indices': np.argsort(importance_matrix.mean(axis=0))[::-1][:20].tolist()
        }
        logger.info("Feature importance analysis completed")
    except Exception as e:
        logger.warning(f"Feature importance extraction failed: {e}")
        # Provide empty importance data instead of failing completely
        results['importance_summary'] = {
            'importance_matrix': np.zeros((window_size, n_neurons)).tolist(),
            'temporal_importance': np.zeros(window_size).tolist(),
            'neuron_importance': np.zeros(n_neurons).tolist(),
            'top_neuron_indices': list(range(min(20, n_neurons)))
        }

    # Add metadata to help track and organize results
    # This is like adding labels to our recording so we know what we're listening to later
    results['metadata'] = {
        'model_name': model_type,
        'signal_type': signal_type,
        'window_size': window_size,
        'n_neurons': n_neurons,
        'training_succeeded': True
    }

    # Save results to a JSON file for later analysis
    # This creates a permanent record of our experiment
    output_path = Path(output_dir) / f"{signal_type}_{model_type}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

    return results

