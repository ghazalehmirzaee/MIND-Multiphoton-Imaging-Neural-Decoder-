"""
Performance metrics with emphasis on deconvolved signal superiority.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate comprehensive classification metrics.
    """
    # Convert tensors to numpy if needed
    # This is like translating between different languages so everyone can understand
    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, 'cpu'):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and hasattr(y_prob, 'cpu'):
        y_prob = y_prob.cpu().numpy()

    # Ensure we have 1D arrays for labels
    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()

    # Calculate basic classification metrics
    # These are like the vital signs of model performance
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='binary', zero_division=0))
    }

    # Add probability-based metrics if available
    # These require the model to give us confidence scores, not just yes/no answers
    if y_prob is not None:
        try:
            # Handle different probability formats
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                # For binary classification with 2-column probability output
                # Use the probability of the positive class (class 1)
                prob_positive = y_prob[:, 1]
            elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
                # Single column probability
                prob_positive = y_prob.ravel()
            elif y_prob.ndim == 1:
                # Already 1D probability array
                prob_positive = y_prob
            else:
                logger.warning(f"Unexpected probability shape: {y_prob.shape}")
                prob_positive = None

            if prob_positive is not None:
                # Calculate ROC AUC
                metrics['roc_auc'] = float(roc_auc_score(y_true, prob_positive))
                logger.info(f"Successfully calculated ROC AUC: {metrics['roc_auc']:.3f}")
            else:
                logger.warning("Could not extract positive class probabilities")

        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            # Don't fail completely if ROC calculation fails

    return metrics


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and return comprehensive results including curve data.
    """
    logger.info("Starting model evaluation")

    try:
        # Get predictions - this is like asking the model for its final answers
        y_pred = model.predict(X_test)
        logger.info(f"Generated predictions for {len(y_test)} test samples")
    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        return {'error': f"Prediction failed: {e}"}

    # Try to get probability predictions - this is like asking for confidence scores
    y_prob = None
    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            logger.info(f"Generated probability predictions with shape: {y_prob.shape}")
        else:
            logger.warning(f"Model {type(model)} does not support probability predictions")
    except Exception as e:
        logger.warning(f"Could not get probability predictions: {e}")

    # Calculate basic metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)

    # Generate confusion matrix - this shows us exactly where the model makes mistakes
    try:
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        logger.info("Generated confusion matrix")
    except Exception as e:
        logger.error(f"Failed to generate confusion matrix: {e}")
        cm = np.array([[0, 0], [0, 0]])
        cm_normalized = cm

    # Generate curve data for visualizations
    curve_data = {}

    if y_prob is not None:
        try:
            # Convert to numpy and ensure correct format
            if hasattr(y_test, 'cpu'):
                y_test_np = y_test.cpu().numpy()
            else:
                y_test_np = np.array(y_test)

            if hasattr(y_prob, 'cpu'):
                y_prob_np = y_prob.cpu().numpy()
            else:
                y_prob_np = np.array(y_prob)

            # Ensure 1D arrays
            if y_test_np.ndim > 1:
                y_test_np = y_test_np.ravel()

            # Extract positive class probabilities
            if y_prob_np.ndim == 2 and y_prob_np.shape[1] == 2:
                prob_positive = y_prob_np[:, 1]
            elif y_prob_np.ndim == 2 and y_prob_np.shape[1] == 1:
                prob_positive = y_prob_np.ravel()
            elif y_prob_np.ndim == 1:
                prob_positive = y_prob_np
            else:
                raise ValueError(f"Unexpected probability shape: {y_prob_np.shape}")

            # Generate ROC curve data
            try:
                fpr, tpr, roc_thresholds = roc_curve(y_test_np, prob_positive)
                curve_data['roc'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                }
                logger.info("Generated ROC curve data")
            except Exception as e:
                logger.warning(f"Failed to generate ROC curve: {e}")

            # Generate Precision-Recall curve data
            try:
                precision, recall, pr_thresholds = precision_recall_curve(y_test_np, prob_positive)
                curve_data['precision_recall'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': pr_thresholds.tolist()
                }
                logger.info("Generated Precision-Recall curve data")
            except Exception as e:
                logger.warning(f"Failed to generate PR curve: {e}")

        except Exception as e:
            logger.error(f"Error processing probability data for curves: {e}")

    # Compile comprehensive results
    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'curve_data': curve_data
    }

    # Log summary of what we accomplished
    logger.info(f"Model evaluation complete:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  Curve data available: {list(curve_data.keys())}")

    return results

