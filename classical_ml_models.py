# classical_ml_models.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import wandb
from tqdm import tqdm
import json
import os
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


class ClassicalMLModels:
    def __init__(self, data_dict, wandb_run=None):
        # Set seed for reproducibility
        set_seeds(42)

        self.data = data_dict
        self.wandb = wandb_run
        self.models = {}
        self.results = {}
        self.signal_types = ['calcium', 'deltaf', 'deconv']
        self.model_types = ['random_forest', 'svm', 'xgboost', 'mlp']

    def train_random_forest(self, X_train, y_train, X_val, y_val, signal_type):
        """Train Random Forest model."""
        print(f"Training Random Forest on {signal_type} signal...")

        # Define parameter grid for grid search
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=42)

        # Get class weights
        class_weights = self.data.get('class_weights', {}).get(signal_type, None)
        if class_weights:
            # Convert class weights to format expected by sklearn
            rf.set_params(class_weight=class_weights)

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring='f1_weighted'
        )

        # Fit model
        print("Performing grid search for Random Forest...")
        grid_search.fit(X_train, y_train.astype(int))  # Convert labels to int

        # Get best model
        best_rf = grid_search.best_estimator_

        # Log best parameters to W&B
        if self.wandb:
            self.wandb.log({f"{signal_type}_rf_best_params": grid_search.best_params_})

        # Evaluate on validation set
        y_pred = best_rf.predict(X_val)

        # Store model
        self.models[f"{signal_type}_random_forest"] = best_rf

        print(f"RF training complete with best parameters: {grid_search.best_params_}")

        return best_rf, y_pred

    def train_svm(self, X_train, y_train, X_val, y_val, signal_type):
        """Train SVM model."""
        print(f"Training SVM on {signal_type} signal...")

        # Define parameter grid for grid search
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }

        # Initialize SVM
        svm = SVC(random_state=42, probability=True)

        # Get class weights
        class_weights = self.data.get('class_weights', {}).get(signal_type, None)
        if class_weights:
            # Convert class weights to format expected by sklearn
            svm.set_params(class_weight=class_weights)

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=svm,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring='f1_weighted'
        )

        # Fit model
        print("Performing grid search for SVM...")
        grid_search.fit(X_train, y_train.astype(int))  # Convert labels to int

        # Get best model
        best_svm = grid_search.best_estimator_

        # Log best parameters to W&B
        if self.wandb:
            self.wandb.log({f"{signal_type}_svm_best_params": grid_search.best_params_})

        # Evaluate on validation set
        y_pred = best_svm.predict(X_val)

        # Store model
        self.models[f"{signal_type}_svm"] = best_svm

        print(f"SVM training complete with best parameters: {grid_search.best_params_}")

        return best_svm, y_pred

    def train_xgboost(self, X_train, y_train, X_val, y_val, signal_type):
        """Train XGBoost model."""
        print(f"Training XGBoost on {signal_type} signal...")

        # Define parameter grid for grid search
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }

        # Initialize XGBoost
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')

        # Get class weights
        class_weights = self.data.get('class_weights', {}).get(signal_type, None)
        if class_weights:
            # XGBoost uses sample weights, so convert class weights to sample weights
            sample_weights = np.ones(len(y_train))
            for cls, weight in class_weights.items():
                # Convert cls to int and handle float labels
                cls_int = int(cls)
                sample_weights[y_train.astype(int) == cls_int] = weight
        else:
            sample_weights = None

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring='f1_weighted'
        )

        # Fit model
        print("Performing grid search for XGBoost...")
        grid_search.fit(
            X_train,
            y_train.astype(int),  # Convert labels to int
            sample_weight=sample_weights
        )

        # Get best model
        best_xgb = grid_search.best_estimator_

        # Log best parameters to W&B
        if self.wandb:
            self.wandb.log({f"{signal_type}_xgb_best_params": grid_search.best_params_})

        # Evaluate on validation set
        y_pred = best_xgb.predict(X_val)

        # Store model
        self.models[f"{signal_type}_xgboost"] = best_xgb

        print(f"XGBoost training complete with best parameters: {grid_search.best_params_}")

        return best_xgb, y_pred

    def train_mlp(self, X_train, y_train, X_val, y_val, signal_type):
        """Train Multilayer Perceptron (MLP) model."""
        print(f"Training MLP on {signal_type} signal...")

        # Define parameter grid for grid search
        param_grid = {
            'hidden_layer_sizes': [(64,), (128,), (64, 32)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01],
            'early_stopping': [True],
            'n_iter_no_change': [10]
        }

        # Initialize MLP with early stopping
        mlp = MLPClassifier(random_state=42, max_iter=1000,
                            early_stopping=True, validation_fraction=0.1)

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=mlp,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring='f1_weighted'
        )

        # Fit model
        print("Performing grid search for MLP...")
        grid_search.fit(X_train, y_train.astype(int))  # Convert labels to int

        # Get best model
        best_mlp = grid_search.best_estimator_

        # Log best parameters to W&B
        if self.wandb:
            self.wandb.log({f"{signal_type}_mlp_best_params": grid_search.best_params_})

        # Evaluate on validation set
        y_pred = best_mlp.predict(X_val)

        # Store model
        self.models[f"{signal_type}_mlp"] = best_mlp

        print(f"MLP training complete with best parameters: {grid_search.best_params_}")

        return best_mlp, y_pred


    def evaluate_model(self, y_true, y_pred, signal_type, model_type):
        """Evaluate model performance and store metrics."""
        # Convert to int to avoid any issues
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        # Get classes (excluding 0 which is background/no action)
        classes = np.unique(y_true)
        non_background_classes = classes[classes != 0] if 0 in classes else classes

        # Calculate overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        # Calculate per-class metrics
        per_class_metrics = {}
        for cls in classes:
            # Create binary labels for the current class
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)

            # Calculate metrics
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

            # Store in dictionary
            per_class_metrics[f"class_{int(cls)}"] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            }

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Store metrics
        metrics = {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist()
        }

        # Store results
        self.results[f"{signal_type}_{model_type}"] = metrics

        # Log to W&B
        if self.wandb:
            # Log metrics
            self.wandb.log({
                f"{signal_type}_{model_type}_accuracy": accuracy,
                f"{signal_type}_{model_type}_precision_macro": precision_macro,
                f"{signal_type}_{model_type}_recall_macro": recall_macro,
                f"{signal_type}_{model_type}_f1_macro": f1_macro
            })

            # Log confusion matrix
            cm_figure = plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {signal_type} - {model_type}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            # Log figure to W&B
            self.wandb.log({f"{signal_type}_{model_type}_confusion_matrix": wandb.Image(cm_figure)})
            plt.close(cm_figure)

            # Calculate and log ROC curve for each class
            if len(classes) > 1:
                # Get probability predictions
                # FIX: Remove 'test_' prefix when accessing model
                model_key = f"{signal_type.replace('test_', '')}_{model_type}"
                model = self.models[model_key]

                # Get validation data for the appropriate signal type
                signal_type_clean = signal_type.replace('test_', '')
                X_data = self.data[f'X_val_{signal_type_clean}']
                if signal_type.startswith('test_'):
                    X_data = self.data[f'X_test_{signal_type_clean}']

                if hasattr(model, 'predict_proba'):
                    y_score = model.predict_proba(X_data)

                    # Create ROC curve for each class
                    roc_figure, ax = plt.subplots(figsize=(10, 8))
                    for i, cls in enumerate(classes):
                        y_true_binary = (y_true == cls).astype(int)
                        if y_score.shape[1] > i:  # Ensure class exists in predictions
                            fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
                            roc_auc = auc(fpr, tpr)
                            ax.plot(fpr, tpr, lw=2,
                                    label=f'Class {cls} (AUC = {roc_auc:.2f})')

                    ax.plot([0, 1], [0, 1], 'k--', lw=2)
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'ROC Curve - {signal_type} - {model_type}')
                    ax.legend(loc="lower right")

                    # Log figure to W&B
                    self.wandb.log({f"{signal_type}_{model_type}_roc_curve": wandb.Image(roc_figure)})
                    plt.close(roc_figure)

        return metrics

    def train_and_evaluate_all(self):
        """Train and evaluate all models on all signal types."""
        for signal_type in self.signal_types:
            # Get data for this signal type
            X_train = self.data[f'X_train_{signal_type}']
            y_train = self.data[f'y_train_{signal_type}']
            X_val = self.data[f'X_val_{signal_type}']
            y_val = self.data[f'y_val_{signal_type}']

            # Train Random Forest
            _, y_pred_rf = self.train_random_forest(X_train, y_train, X_val, y_val, signal_type)
            self.evaluate_model(y_val, y_pred_rf, signal_type, 'random_forest')

            # Train SVM
            _, y_pred_svm = self.train_svm(X_train, y_train, X_val, y_val, signal_type)
            self.evaluate_model(y_val, y_pred_svm, signal_type, 'svm')

            # Train XGBoost
            _, y_pred_xgb = self.train_xgboost(X_train, y_train, X_val, y_val, signal_type)
            self.evaluate_model(y_val, y_pred_xgb, signal_type, 'xgboost')

            # Train MLP
            _, y_pred_mlp = self.train_mlp(X_train, y_train, X_val, y_val, signal_type)
            self.evaluate_model(y_val, y_pred_mlp, signal_type, 'mlp')

    def test_best_models(self):
        """Evaluate best models on test set."""
        test_results = {}

        for signal_type in self.signal_types:
            # Get test data
            X_test = self.data[f'X_test_{signal_type}']
            y_test = self.data[f'y_test_{signal_type}']

            signal_results = {}

            # Test each model type
            for model_type in self.model_types:
                # Get model
                model = self.models[f"{signal_type}_{model_type}"]

                # Make predictions
                y_pred = model.predict(X_test)

                # Evaluate
                metrics = self.evaluate_model(y_test, y_pred, f"test_{signal_type}", model_type)
                signal_results[model_type] = metrics

            test_results[signal_type] = signal_results

        # Store test results
        self.results['test_results'] = test_results

        return test_results

    def feature_importance(self):
        """Extract feature importance from tree-based models."""
        importance_results = {}

        # Get window size and number of neurons
        window_size = self.data['window_size']
        n_neurons = {}
        for signal_type in self.signal_types:
            n_neurons[signal_type] = self.data.get(f'n_{signal_type}_neurons',
                                                   self.data[f'X_train_{signal_type}'].shape[1] // window_size)

        for signal_type in self.signal_types:
            # Get feature importance from Random Forest
            rf_model = self.models[f"{signal_type}_random_forest"]
            rf_importance = rf_model.feature_importances_

            # Get feature importance from XGBoost
            xgb_model = self.models[f"{signal_type}_xgboost"]
            xgb_importance = xgb_model.feature_importances_

            # Get MLP coefficients (not direct feature importance but can be used as a proxy)
            mlp_model = self.models[f"{signal_type}_mlp"]
            if hasattr(mlp_model, 'coefs_') and len(mlp_model.coefs_) > 0:
                # Use absolute values of first layer weights as rough feature importance
                mlp_importance = np.abs(mlp_model.coefs_[0]).mean(axis=1)
            else:
                mlp_importance = np.ones(rf_importance.shape)  # Default if not available

            # Reshape importance scores to (window_size, n_neurons)
            current_n_neurons = n_neurons[signal_type]
            rf_importance_2d = rf_importance.reshape(window_size, current_n_neurons)
            xgb_importance_2d = xgb_importance.reshape(window_size, current_n_neurons)
            mlp_importance_2d = mlp_importance.reshape(window_size, current_n_neurons)

            # Store results
            importance_results[f"{signal_type}_rf"] = rf_importance_2d.tolist()
            importance_results[f"{signal_type}_xgb"] = xgb_importance_2d.tolist()
            importance_results[f"{signal_type}_mlp"] = mlp_importance_2d.tolist()

            # Visualize temporal importance (mean across neurons)
            temporal_importance_rf = np.mean(rf_importance_2d, axis=1)
            temporal_importance_xgb = np.mean(xgb_importance_2d, axis=1)
            temporal_importance_mlp = np.mean(mlp_importance_2d, axis=1)

            # Create figure
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))

            # Plot RF temporal importance
            ax[0].bar(range(window_size), temporal_importance_rf)
            ax[0].set_title(f'RF Temporal Importance - {signal_type}')
            ax[0].set_xlabel('Time Step')
            ax[0].set_ylabel('Mean Feature Importance')

            # Plot XGB temporal importance
            ax[1].bar(range(window_size), temporal_importance_xgb)
            ax[1].set_title(f'XGB Temporal Importance - {signal_type}')
            ax[1].set_xlabel('Time Step')
            ax[1].set_ylabel('Mean Feature Importance')

            # Plot MLP "importance"
            ax[2].bar(range(window_size), temporal_importance_mlp)
            ax[2].set_title(f'MLP Weight Magnitude - {signal_type}')
            ax[2].set_xlabel('Time Step')
            ax[2].set_ylabel('Mean Weight Magnitude')

            # Save figure
            plt.tight_layout()
            plt.savefig(f'results/figures/{signal_type}_temporal_importance.png', dpi=300)

            # Log to W&B
            if self.wandb:
                self.wandb.log({f"{signal_type}_temporal_importance": wandb.Image(fig)})

            plt.close(fig)

            # Visualize neuron importance (mean across time)
            neuron_importance_rf = np.mean(rf_importance_2d, axis=0)
            neuron_importance_xgb = np.mean(xgb_importance_2d, axis=0)
            neuron_importance_mlp = np.mean(mlp_importance_2d, axis=0)

            # Create figure
            fig, ax = plt.subplots(1, 3, figsize=(24, 6))

            # Get number of neurons to show (max 50)
            num_neurons_to_show = min(50, current_n_neurons)

            # Plot RF neuron importance (top N neurons)
            top_neurons_rf = np.argsort(neuron_importance_rf)[-num_neurons_to_show:]
            ax[0].bar(top_neurons_rf, neuron_importance_rf[top_neurons_rf])
            ax[0].set_title(f'RF Top {num_neurons_to_show} Neuron Importance - {signal_type}')
            ax[0].set_xlabel('Neuron Index')
            ax[0].set_ylabel('Mean Feature Importance')

            # Plot XGB neuron importance (top N neurons)
            top_neurons_xgb = np.argsort(neuron_importance_xgb)[-num_neurons_to_show:]
            ax[1].bar(top_neurons_xgb, neuron_importance_xgb[top_neurons_xgb])
            ax[1].set_title(f'XGB Top {num_neurons_to_show} Neuron Importance - {signal_type}')
            ax[1].set_xlabel('Neuron Index')
            ax[1].set_ylabel('Mean Feature Importance')

            # Plot MLP neuron importance (top N neurons)
            top_neurons_mlp = np.argsort(neuron_importance_mlp)[-num_neurons_to_show:]
            ax[2].bar(top_neurons_mlp, neuron_importance_mlp[top_neurons_mlp])
            ax[2].set_title(f'MLP Top {num_neurons_to_show} Neuron Weight Magnitude - {signal_type}')
            ax[2].set_xlabel('Neuron Index')
            ax[2].set_ylabel('Mean Weight Magnitude')

            # Save figure
            plt.tight_layout()
            plt.savefig(f'results/figures/{signal_type}_neuron_importance.png', dpi=300)

            # Log to W&B
            if self.wandb:
                self.wandb.log({f"{signal_type}_neuron_importance": wandb.Image(fig)})

            plt.close(fig)

        # Store feature importance results
        self.results['feature_importance'] = importance_results

        return importance_results

    def save_results(self, output_dir='results/metrics'):
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, 'classical_ml_results.json')

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=4)

        print(f"Results saved to {output_file}")

    def save_models(self, output_dir='models'):
        """Save trained models."""
        import pickle

        os.makedirs(output_dir, exist_ok=True)

        for model_name, model in self.models.items():
            model_file = os.path.join(output_dir, f"{model_name}.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)

        print(f"Models saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Load processed data
    data = np.load('data/processed/processed_calcium_data.npz', allow_pickle=True)
    data_dict = dict(data.items())  # Convert to dictionary

    # Initialize W&B
    wandb_run = wandb.init(project="MIND", name="classical_ml_comparison", entity="mirzaeeghazal")

    # Train and evaluate models
    clf = ClassicalMLModels(data_dict, wandb_run)
    clf.train_and_evaluate_all()
    clf.test_best_models()
    clf.feature_importance()

    # Save results
    clf.save_results()
    clf.save_models()

    # Finish W&B run
    wandb_run.finish()

