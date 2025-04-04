# visualize_results.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from matplotlib.gridspec import GridSpec
import wandb


class ResultVisualizer:
    def __init__(self, classical_ml_file, deep_learning_file, wandb_run=None):
        # Load results
        with open(classical_ml_file, 'r') as f:
            self.classical_results = json.load(f)

        with open(deep_learning_file, 'r') as f:
            self.dl_results = json.load(f)

        self.wandb = wandb_run
        self.signal_types = ['calcium', 'deltaf', 'deconv']
        self.classical_models = ['random_forest', 'svm', 'xgboost', 'mlp']
        self.dl_models = ['mlp', 'lstm', 'cnn']

        # Combine results
        self.combined_results = self.combine_results()

    def combine_results(self):
        """Combine results from classical ML and DL models."""
        combined = {}

        # Extract test results
        if 'test_results' in self.classical_results:
            classical_test = self.classical_results['test_results']
            for signal_type in self.signal_types:
                if signal_type in classical_test:
                    for model_type, metrics in classical_test[signal_type].items():
                        key = f"{signal_type}_{model_type}"
                        combined[key] = metrics

        if 'test_results' in self.dl_results:
            dl_test = self.dl_results['test_results']
            for signal_type in self.signal_types:
                if signal_type in dl_test:
                    for model_type, metrics in dl_test[signal_type].items():
                        key = f"{signal_type}_{model_type}"
                        combined[key] = metrics

        return combined

    def create_performance_comparison_table(self):
        """Create a table comparing performance across models and signal types."""
        # Create a DataFrame to store results
        columns = [
            'Signal Type', 'Model', 'Accuracy', 'Precision (Macro)',
            'Recall (Macro)', 'F1 (Macro)'
        ]

        data = []

        # Extract metrics from combined results
        for key, metrics in self.combined_results.items():
            # Skip keys that don't match the expected pattern
            if 'test_' in key:
                continue

            # Parse key
            parts = key.split('_')
            signal_type = parts[0]
            model_type = '_'.join(parts[1:])

            # Extract metrics
            row = [
                signal_type, model_type,
                metrics['accuracy'], metrics['precision_macro'],
                metrics['recall_macro'], metrics['f1_macro']
            ]

            data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)

        # Sort by Signal Type and F1 score
        df = df.sort_values(['Signal Type', 'F1 (Macro)'], ascending=[True, False])

        return df

    def plot_performance_comparison(self):
        """Plot performance comparison across models and signal types."""
        # Get performance table
        df = self.create_performance_comparison_table()

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()

        metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']

        for i, metric in enumerate(metrics):
            # Create pivot table
            pivot = pd.pivot_table(
                df, values=metric, index='Model', columns='Signal Type'
            )

            # Plot heatmap
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[i])
            axes[i].set_title(f'{metric} by Model and Signal Type')

            # Adjust labels
            axes[i].set_ylabel('Model')
            axes[i].set_xlabel('Signal Type')

        plt.tight_layout()

        # Save figure
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig('results/figures/performance_comparison.png', dpi=300)

        # Log to W&B
        if self.wandb:
            self.wandb.log({"performance_comparison": wandb.Image(fig)})

        plt.close(fig)

        return fig

    def plot_signal_type_comparison(self):
        """Plot performance comparison across signal types."""
        # Get performance table
        df = self.create_performance_comparison_table()

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Calculate mean performance by signal type
        signal_performance = df.groupby('Signal Type')['F1 (Macro)'].mean().reset_index()

        # Get best model for each signal type
        best_models = df.loc[df.groupby('Signal Type')['F1 (Macro)'].idxmax()]

        # Plot bar chart
        sns.barplot(x='Signal Type', y='F1 (Macro)', data=signal_performance, ax=ax, alpha=0.7)

        # Add best model annotations
        for i, row in enumerate(best_models.itertuples()):
            ax.text(i, row._4 + 0.02, f"Best: {row.Model}\nF1: {row._6:.3f}",
                    ha='center', va='bottom', fontweight='bold')

        ax.set_title('Mean F1 Score by Signal Type with Best Model')
        ax.set_ylim(top=1.0)

        # Save figure
        plt.savefig('results/figures/signal_type_comparison.png', dpi=300)

        # Log to W&B
        if self.wandb:
            self.wandb.log({"signal_type_comparison": wandb.Image(fig)})

        plt.close(fig)

        return fig

    def plot_confusion_matrices(self):
        """Plot confusion matrices for the best model of each signal type."""
        # Get performance table
        df = self.create_performance_comparison_table()

        # Get best model for each signal type
        best_models = df.loc[df.groupby('Signal Type')['F1 (Macro)'].idxmax()]

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        for i, (signal_type, ax) in enumerate(zip(self.signal_types, axes)):
            # Get best model for this signal type
            best_model = best_models[best_models['Signal Type'] == signal_type]['Model'].values[0]

            # Get confusion matrix
            cm = np.array(self.combined_results[f"{signal_type}_{best_model}"]['confusion_matrix'])

            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {signal_type} - {best_model}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')

        plt.tight_layout()

        # Save figure
        plt.savefig('results/figures/best_confusion_matrices.png', dpi=300)

        # Log to W&B
        if self.wandb:
            self.wandb.log({"best_confusion_matrices": wandb.Image(fig)})

        plt.close(fig)

        return fig

    def plot_model_comparison(self):
        """Plot performance comparison across model types."""
        # Get performance table
        df = self.create_performance_comparison_table()

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Calculate mean performance by model type
        model_performance = df.groupby('Model')['F1 (Macro)'].mean().reset_index()

        # Get best signal type for each model
        best_signals = df.loc[df.groupby('Model')['F1 (Macro)'].idxmax()]

        # Sort by F1 score
        model_performance = model_performance.sort_values('F1 (Macro)', ascending=False)

        # Plot bar chart
        sns.barplot(x='Model', y='F1 (Macro)', data=model_performance, ax=ax, alpha=0.7)

        # Add best signal type annotations
        for i, row in enumerate(best_signals.itertuples()):
            model_idx = model_performance[model_performance['Model'] == row.Model].index[0]
            ax.text(model_idx, row._6 + 0.02, f"Best with: {row._2}\nF1: {row._6:.3f}",
                    ha='center', va='bottom', fontweight='bold')

        ax.set_title('Mean F1 Score by Model Type with Best Signal Type')
        ax.set_ylim(top=1.0)
        plt.xticks(rotation=45)

        # Save figure
        plt.savefig('results/figures/model_comparison.png', dpi=300)

        # Log to W&B
        if self.wandb:
            self.wandb.log({"model_comparison": wandb.Image(fig)})

        plt.close(fig)

        return fig

    def plot_feature_importance(self):
        """Plot feature importance for tree-based models."""
        if 'feature_importance' not in self.classical_results:
            print("Feature importance not found in classical ML results.")
            return None

        feature_importance = self.classical_results['feature_importance']

        # Create figure
        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(3, 2, figure=fig)

        for i, signal_type in enumerate(self.signal_types):
            # Random Forest importance
            rf_key = f"{signal_type}_rf"
            if rf_key in feature_importance:
                ax_rf = fig.add_subplot(gs[i, 0])
                rf_importance = np.array(feature_importance[rf_key])

                # Plot importance heatmap
                im = ax_rf.imshow(rf_importance.T, aspect='auto', cmap='viridis')
                ax_rf.set_title(f'Random Forest Feature Importance - {signal_type}')
                ax_rf.set_xlabel('Time Step')
                ax_rf.set_ylabel('Neuron')
                plt.colorbar(im, ax=ax_rf)

            # XGBoost importance
            xgb_key = f"{signal_type}_xgb"
            if xgb_key in feature_importance:
                ax_xgb = fig.add_subplot(gs[i, 1])
                xgb_importance = np.array(feature_importance[xgb_key])

                # Plot importance heatmap
                im = ax_xgb.imshow(xgb_importance.T, aspect='auto', cmap='viridis')
                ax_xgb.set_title(f'XGBoost Feature Importance - {signal_type}')
                ax_xgb.set_xlabel('Time Step')
                ax_xgb.set_ylabel('Neuron')
                plt.colorbar(im, ax=ax_xgb)

        plt.tight_layout()

        # Save figure
        plt.savefig('results/figures/feature_importance.png', dpi=300)

        # Log to W&B
        if self.wandb:
            self.wandb.log({"feature_importance": wandb.Image(fig)})

        plt.close(fig)

        return fig

    def plot_temporal_importance(self):
        """Plot temporal importance for each signal type."""
        if 'feature_importance' not in self.classical_results:
            print("Feature importance not found in classical ML results.")
            return None

        feature_importance = self.classical_results['feature_importance']

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        for i, signal_type in enumerate(self.signal_types):
            # Random Forest importance
            rf_key = f"{signal_type}_rf"
            if rf_key in feature_importance:
                rf_importance = np.array(feature_importance[rf_key])

                # Calculate mean importance across neurons
                temporal_importance = np.mean(rf_importance, axis=1)

                # Plot importance
                axes[i].bar(range(len(temporal_importance)), temporal_importance)
                axes[i].set_title(f'Temporal Importance - {signal_type}')
                axes[i].set_xlabel('Time Step')
                axes[i].set_ylabel('Mean Feature Importance')

        plt.tight_layout()

        # Save figure
        plt.savefig('results/figures/temporal_importance.png', dpi=300)

        # Log to W&B
        if self.wandb:
            self.wandb.log({"temporal_importance": wandb.Image(fig)})

        plt.close(fig)

        return fig

    def plot_neuron_importance(self):
        """Plot neuron importance for each signal type."""
        if 'feature_importance' not in self.classical_results:
            print("Feature importance not found in classical ML results.")
            return None

        feature_importance = self.classical_results['feature_importance']

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        for i, signal_type in enumerate(self.signal_types):
            # Random Forest importance
            rf_key = f"{signal_type}_rf"
            if rf_key in feature_importance:
                rf_importance = np.array(feature_importance[rf_key])

                # Calculate mean importance across time
                neuron_importance = np.mean(rf_importance, axis=0)

                # Get top 50 neurons
                top_neurons = np.argsort(neuron_importance)[-50:]

                # Plot importance
                axes[i].bar(top_neurons, neuron_importance[top_neurons])
                axes[i].set_title(f'Top 50 Neuron Importance - {signal_type}')
                axes[i].set_xlabel('Neuron Index')
                axes[i].set_ylabel('Mean Feature Importance')

        plt.tight_layout()

        # Save figure
        plt.savefig('results/figures/neuron_importance.png', dpi=300)

        # Log to W&B
        if self.wandb:
            self.wandb.log({"neuron_importance": wandb.Image(fig)})

        plt.close(fig)

        return fig

    def create_summary_report(self):
        """Create a summary report of all results."""
        # Get performance table
        df = self.create_performance_comparison_table()

        # Get best models by signal type
        best_by_signal = df.loc[df.groupby('Signal Type')['F1 (Macro)'].idxmax()]

        # Get best signal type overall
        best_signal = best_by_signal.loc[best_by_signal['F1 (Macro)'].idxmax()]

        # Create summary dictionary
        summary = {
            "best_overall": {
                "signal_type": best_signal['Signal Type'],
                "model": best_signal['Model'],
                "accuracy": float(best_signal['Accuracy']),
                "precision": float(best_signal['Precision (Macro)']),
                "recall": float(best_signal['Recall (Macro)']),
                "f1": float(best_signal['F1 (Macro)'])
            },
            "best_by_signal_type": {}
        }

        # Add best model for each signal type
        for _, row in best_by_signal.iterrows():
            signal_type = row['Signal Type']
            summary["best_by_signal_type"][signal_type] = {
                "model": row['Model'],
                "accuracy": float(row['Accuracy']),
                "precision": float(row['Precision (Macro)']),
                "recall": float(row['Recall (Macro)']),
                "f1": float(row['F1 (Macro)'])
            }

        # Add signal type rankings
        signal_rankings = df.groupby('Signal Type')['F1 (Macro)'].mean().sort_values(ascending=False)
        summary["signal_type_rankings"] = {
            signal: {"mean_f1": float(score)} for signal, score in signal_rankings.items()
        }

        # Add model rankings
        model_rankings = df.groupby('Model')['F1 (Macro)'].mean().sort_values(ascending=False)
        summary["model_rankings"] = {
            model: {"mean_f1": float(score)} for model, score in model_rankings.items()
        }

        # Save summary to JSON
        os.makedirs('results/metrics', exist_ok=True)
        with open('results/metrics/summary_report.json', 'w') as f:
            json.dump(summary, f, indent=4)

        # Create summary table
        summary_table = pd.DataFrame({
            "Metric": ["Best Overall Signal Type", "Best Overall Model", "Best Overall F1 Score"],
            "Value": [
                best_signal['Signal Type'],
                best_signal['Model'],
                f"{best_signal['F1 (Macro)']:.3f}"
            ]
        })

        # Log to W&B
        if self.wandb:
            self.wandb.log({"summary_report": summary})

            # Create W&B table
            wandb_table = wandb.Table(columns=summary_table.columns.tolist())
            for _, row in summary_table.iterrows():
                wandb_table.add_data(*row.tolist())

            self.wandb.log({"summary_table": wandb_table})

        return summary

    def visualize_all(self):
        """Generate all visualizations."""
        # Create all plots
        self.plot_performance_comparison()
        self.plot_signal_type_comparison()
        self.plot_confusion_matrices()
        self.plot_model_comparison()
        self.plot_feature_importance()
        self.plot_temporal_importance()
        self.plot_neuron_importance()

        # Create summary report
        summary = self.create_summary_report()

        return summary


# Example usage
if __name__ == "__main__":
    # Initialize W&B
    wandb_run = wandb.init(project="MIND", name="results_visualization", entity="mirzaeeghazal")

    # Create visualizer
    visualizer = ResultVisualizer(
        'results/metrics/classical_ml_results.json',
        'results/metrics/deep_learning_results.json',
        wandb_run
    )

    # Generate all visualizations
    summary = visualizer.visualize_all()

    # Print summary
    print("Summary Report:")
    print(json.dumps(summary, indent=4))

    # Finish W&B run
    wandb.finish()

