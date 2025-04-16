# deep_learning_models.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import wandb
import json
import os
import math
from tqdm import tqdm
import random
from data_utils import CalciumImagingDataset, create_dataloaders


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


# Model architectures
class MLPModel(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(MLPModel, self).__init__()
        # Deeper network with batch normalization
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_len, hidden_size]
        attn_weights = self.attention(hidden_states)  # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # [batch_size, seq_len, 1]

        # Apply attention weights to hidden states
        context = torch.sum(hidden_states * attn_weights, dim=1)  # [batch_size, hidden_size]
        return context, attn_weights


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=256, n_classes=3):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.4)
        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.4)

        # Add attention layer
        self.attention = AttentionLayer(hidden_size2 * 2)

        self.fc1 = nn.Linear(hidden_size2 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # First BiLSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        # Second BiLSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)

        # Apply attention
        context, _ = self.attention(lstm2_out)

        # Dense layers with batch normalization
        x = torch.relu(self.bn1(self.fc1(context)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, input_size, window_size, n_classes):
        super(CNNModel, self).__init__()
        # In PyTorch Conv1d: input shape is (batch_size, channels, seq_length)
        # So neurons are channels and window_size is sequence length

        # Deeper CNN with residual connections and batch normalization
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        # Residual connection (1x1 conv to match dimensions)
        self.residual = nn.Conv1d(input_size, 256, kernel_size=1)
        self.bn_res = nn.BatchNorm1d(256)

        # Calculate output size after pooling layers
        conv_output_size = window_size
        conv_output_size = conv_output_size // 2  # After first pooling
        conv_output_size = conv_output_size // 2  # After second pooling

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * conv_output_size, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # Save input for residual connection
        residual_in = x

        # Transform from [batch, time_steps, features] to [batch, features, time_steps]
        x = x.permute(0, 2, 1)
        residual_in = residual_in.permute(0, 2, 1)

        # Main path
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))

        # Residual connection (with downsampling to match dimensions)
        res = self.residual(residual_in)
        res = self.bn_res(res)
        res = nn.functional.adaptive_avg_pool1d(res, x.size(2))

        # Add residual connection (if shapes match)
        if x.size() == res.size():
            x = x + res

        x = self.flatten(x)
        x = torch.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Custom learning rate scheduler for one-cycle policy
def one_cycle_scheduler(optimizer, max_lr, total_epochs):
    """Create a OneCycleLR scheduler with warmup and cosine annealing."""
    warmup_epochs = int(0.3 * total_epochs)

    # Create lambda function for learning rate changes
    def lambda_func(current_epoch):
        if current_epoch < warmup_epochs:
            # Warmup phase: linearly increase learning rate
            return current_epoch / warmup_epochs
        else:
            # Cooldown phase: cosine decay
            progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return (1 + math.cos(math.pi * progress)) / 2

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)


# Focal loss implementation for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DeepLearningModels:
    def __init__(self, data_dict, wandb_run=None):
        # Set seed for reproducibility
        set_seeds(42)

        self.data = data_dict
        self.wandb = wandb_run
        self.models = {}
        self.results = {}
        self.signal_types = ['calcium', 'deltaf', 'deconv']
        self.model_types = ['mlp', 'lstm', 'cnn']

        # Get window size and neurons for each signal type
        self.window_size = data_dict['window_size']
        self.n_neurons = {}
        for signal_type in self.signal_types:
            self.n_neurons[signal_type] = data_dict.get(
                f'n_{signal_type}_neurons',
                data_dict[f'X_train_{signal_type}'].shape[1] // self.window_size
            )

        # Get number of classes
        for signal_type in self.signal_types:
            if f'y_train_{signal_type}' in data_dict:
                y_unique = np.unique(data_dict[f'y_train_{signal_type}'])
                self.n_classes = len(np.unique(y_unique))
                break

        # Set device (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Preprocess data for deep learning
        self.preprocess_data()

    def preprocess_data(self):
        """Preprocess data for deep learning models."""
        # Reshape data for CNN and LSTM
        for signal_type in self.signal_types:
            # Extract data
            X_train = self.data[f'X_train_{signal_type}']
            X_val = self.data[f'X_val_{signal_type}']
            X_test = self.data[f'X_test_{signal_type}']

            # Get number of neurons for this signal type
            n_neurons = self.n_neurons[signal_type]

            # Reshape data: (n_samples, window_size, n_neurons)
            X_train_reshaped = X_train.reshape(-1, self.window_size, n_neurons)
            X_val_reshaped = X_val.reshape(-1, self.window_size, n_neurons)
            X_test_reshaped = X_test.reshape(-1, self.window_size, n_neurons)

            # Store reshaped data
            self.data[f'X_train_{signal_type}_reshaped'] = X_train_reshaped
            self.data[f'X_val_{signal_type}_reshaped'] = X_val_reshaped
            self.data[f'X_test_{signal_type}_reshaped'] = X_test_reshaped

            # Create PyTorch tensors
            X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val_reshaped, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)

            y_train = self.data[f'y_train_{signal_type}']
            y_val = self.data[f'y_val_{signal_type}']
            y_test = self.data[f'y_test_{signal_type}']

            # Convert labels to tensor (long for classification)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)

            # Store tensors
            self.data[f'X_train_{signal_type}_tensor'] = X_train_tensor
            self.data[f'X_val_{signal_type}_tensor'] = X_val_tensor
            self.data[f'X_test_{signal_type}_tensor'] = X_test_tensor
            self.data[f'y_train_{signal_type}_tensor'] = y_train_tensor
            self.data[f'y_val_{signal_type}_tensor'] = y_val_tensor
            self.data[f'y_test_{signal_type}_tensor'] = y_test_tensor

            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

            self.data[f'{signal_type}_train_loader'] = DataLoader(
                train_dataset, batch_size=32, shuffle=True
            )
            self.data[f'{signal_type}_val_loader'] = DataLoader(
                val_dataset, batch_size=32, shuffle=False
            )
            self.data[f'{signal_type}_test_loader'] = DataLoader(
                test_dataset, batch_size=32, shuffle=False
            )

    def build_mlp_model(self, signal_type):
        """Build Multilayer Perceptron (MLP) model."""
        input_dim = self.window_size * self.n_neurons[signal_type]
        model = MLPModel(input_dim, self.n_classes)
        return model

    def build_lstm_model(self, signal_type):
        """Build LSTM model."""
        input_size = self.n_neurons[signal_type]
        model = LSTMModel(input_size, n_classes=self.n_classes)
        return model

    def build_cnn_model(self, signal_type):
        """Build 1D CNN model."""
        input_size = self.n_neurons[signal_type]
        model = CNNModel(input_size, self.window_size, self.n_classes)
        return model

    def train_model(self, model_type, signal_type, epochs=100, batch_size=32):
        """Train a specific model on a specific signal type."""
        print(f"Training {model_type} on {signal_type} signal...")

        # Get data loaders
        train_loader = self.data[f'{signal_type}_train_loader']
        val_loader = self.data[f'{signal_type}_val_loader']

        # Build model
        if model_type == 'mlp':
            model = self.build_mlp_model(signal_type)
        elif model_type == 'lstm':
            model = self.build_lstm_model(signal_type)
        elif model_type == 'cnn':
            model = self.build_cnn_model(signal_type)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Move model to device
        model = model.to(self.device)

        # Define optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Get class weights if available
        class_weights = self.data.get('class_weights', {}).get(signal_type, None)
        if class_weights:
            # Convert to PyTorch tensor
            weight_tensor = torch.ones(self.n_classes)
            for cls, weight in class_weights.items():
                cls_idx = int(cls)
                if cls_idx < self.n_classes:
                    weight_tensor[cls_idx] = weight

            # Use weighted loss function with focal loss for imbalanced data
            criterion = FocalLoss(alpha=2.0, gamma=2.0)
        else:
            criterion = nn.CrossEntropyLoss()

        # Initialize learning rate schedulers
        reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=0.00001
        )
        one_cycle = one_cycle_scheduler(optimizer, max_lr=0.001, total_epochs=epochs)

        # Initialize early stopping parameters
        best_val_loss = float('inf')
        patience = 15  # Increased patience for better convergence
        patience_counter = 0
        best_model_state = None

        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        # Create directory for checkpoints
        os.makedirs('models/checkpoints', exist_ok=True)
        checkpoint_path = f'models/checkpoints/{signal_type}_{model_type}.pt'

        # Train model
        for epoch in range(epochs):
            # Update learning rate according to one-cycle policy BEFORE training
            one_cycle.step(epoch)
            current_lr = optimizer.param_groups[0]['lr']

            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # Use tqdm for progress tracking
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()  # Step optimizer BEFORE lr_scheduler

                # Track statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()

            # Calculate training metrics
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    # Track statistics
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()

            # Calculate validation metrics
            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total

            # Update learning rate based on validation loss AFTER completing an epoch
            reduce_lr.step(val_loss)

            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            # Print progress
            print(f'Epoch {epoch + 1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                  f'LR: {current_lr:.6f}')

            # Log to W&B
            if self.wandb:
                self.wandb.log({
                    f"{signal_type}_{model_type}_train_loss": train_loss,
                    f"{signal_type}_{model_type}_val_loss": val_loss,
                    f"{signal_type}_{model_type}_train_acc": train_acc,
                    f"{signal_type}_{model_type}_val_acc": val_acc,
                    f"{signal_type}_{model_type}_learning_rate": current_lr
                })

            # Early stopping and model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()

                # Save best model
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Store model
        self.models[f"{signal_type}_{model_type}"] = model

        # Store history
        self.results[f"{signal_type}_{model_type}_history"] = history

        # Plot training history
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # Plot loss
        ax[0].plot(history['train_loss'], label='Training Loss')
        ax[0].plot(history['val_loss'], label='Validation Loss')
        ax[0].set_title(f'Loss - {signal_type} - {model_type}')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        # Plot accuracy
        ax[1].plot(history['train_acc'], label='Training Accuracy')
        ax[1].plot(history['val_acc'], label='Validation Accuracy')
        ax[1].set_title(f'Accuracy - {signal_type} - {model_type}')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()

        # Save figure
        plt.tight_layout()
        plt.savefig(f'results/figures/{signal_type}_{model_type}_history.png', dpi=300)

        # Log to W&B
        if self.wandb:
            self.wandb.log({f"{signal_type}_{model_type}_history": wandb.Image(fig)})

        plt.close(fig)

        return model, history

    def evaluate_model(self, model, data_loader, y_true, signal_type, model_type):
        """Evaluate model performance and store metrics."""
        model.eval()
        y_pred_probs = []
        y_preds = []

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)

                # Get probabilities using softmax
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                y_pred_probs.append(probs.cpu().numpy())
                y_preds.append(predicted.cpu().numpy())

        # Concatenate batch predictions
        y_pred_prob = np.concatenate(y_pred_probs)
        y_pred = np.concatenate(y_preds)

        # Get classes
        classes = np.unique(y_true)

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
        self.results[f"{signal_type}_{model_type}_metrics"] = metrics

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

            # Save figure
            plt.savefig(f'results/figures/{signal_type}_{model_type}_confusion_matrix.png', dpi=300)

            # Log figure to W&B
            self.wandb.log({f"{signal_type}_{model_type}_confusion_matrix": wandb.Image(cm_figure)})
            plt.close(cm_figure)

            # Create ROC curves for each class
            if len(classes) > 1:
                roc_figure, ax = plt.subplots(figsize=(10, 8))
                for i, cls in enumerate(classes):
                    y_true_binary = (y_true == cls).astype(int)
                    y_score = y_pred_prob[:, i]
                    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=2, label=f'Class {cls} (AUC = {roc_auc:.2f})')

                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curve - {signal_type} - {model_type}')
                ax.legend(loc="lower right")

                # Save figure
                plt.savefig(f'results/figures/{signal_type}_{model_type}_roc_curve.png', dpi=300)

                # Log figure to W&B
                self.wandb.log({f"{signal_type}_{model_type}_roc_curve": wandb.Image(roc_figure)})
                plt.close(roc_figure)

        return metrics

    def train_and_evaluate_all(self, epochs=100, batch_size=32):
        """Train and evaluate all models on all signal types."""
        for signal_type in self.signal_types:
            # Train and evaluate MLP
            self.train_model('mlp', signal_type, epochs, batch_size)
            mlp_model = self.models[f"{signal_type}_mlp"]
            self.evaluate_model(
                mlp_model,
                self.data[f'{signal_type}_val_loader'],
                self.data[f'y_val_{signal_type}'],
                signal_type,
                'mlp'
            )

            # Train and evaluate LSTM
            self.train_model('lstm', signal_type, epochs, batch_size)
            lstm_model = self.models[f"{signal_type}_lstm"]
            self.evaluate_model(
                lstm_model,
                self.data[f'{signal_type}_val_loader'],
                self.data[f'y_val_{signal_type}'],
                signal_type,
                'lstm'
            )

            # Train and evaluate CNN
            self.train_model('cnn', signal_type, epochs, batch_size)
            cnn_model = self.models[f"{signal_type}_cnn"]
            self.evaluate_model(
                cnn_model,
                self.data[f'{signal_type}_val_loader'],
                self.data[f'y_val_{signal_type}'],
                signal_type,
                'cnn'
            )

    def test_best_models(self):
        """Evaluate best models on test set."""
        test_results = {}

        for signal_type in self.signal_types:
            signal_results = {}

            # Test each model type
            for model_type in self.model_types:
                # Get model
                model = self.models[f"{signal_type}_{model_type}"]

                # Evaluate
                metrics = self.evaluate_model(
                    model,
                    self.data[f'{signal_type}_test_loader'],
                    self.data[f'y_test_{signal_type}'],
                    f"test_{signal_type}",
                    model_type
                )

                signal_results[model_type] = metrics

            test_results[signal_type] = signal_results

        # Store test results
        self.results['test_results'] = test_results

        return test_results

    def visualize_activations(self, layer_name='conv1'):
        """Visualize activations of convolutional layers."""
        activation_results = {}

        for signal_type in self.signal_types:
            # Skip if not a CNN model
            if f"{signal_type}_cnn" not in self.models:
                continue

            # Get model
            model = self.models[f"{signal_type}_cnn"]

            # Create a hook to get activations
            activations = {}

            def get_activation(name):
                def hook(model, input, output):
                    activations[name] = output.detach().cpu().numpy()

                return hook

            # Register hooks for the specified layer
            hook = None
            if layer_name == 'conv1':
                hook = model.conv1.register_forward_hook(get_activation('conv1'))
            else:
                print(f"Layer '{layer_name}' not recognized. Using 'conv1' instead.")
                hook = model.conv1.register_forward_hook(get_activation('conv1'))

            # Get a single sample from test set
            X_test = self.data[f'X_test_{signal_type}_tensor']
            sample = X_test[0:1].to(self.device)

            # Forward pass to get activations
            model.eval()
            with torch.no_grad():
                _ = model(sample)

            # Remove the hook
            hook.remove()

            # Get activations
            act_key = 'conv1' if layer_name == 'conv1' else layer_name
            if act_key not in activations:
                print(f"No activations found for layer '{layer_name}' in {signal_type}_cnn model.")
                continue

            layer_activations = activations[act_key]

            # Store activations
            activation_results[f"{signal_type}_{layer_name}"] = layer_activations.tolist()

            # Visualize activations
            n_filters = layer_activations.shape[1]  # Number of filters
            n_cols = 8
            n_rows = (n_filters + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else axes

            for i in range(min(n_filters, len(axes))):
                # Get activation for one filter
                filter_act = layer_activations[0, i]

                # Plot as an image
                axes[i].imshow(filter_act.reshape(-1, 1), cmap='viridis', aspect='auto')
                axes[i].set_title(f'Filter {i + 1}')
                axes[i].axis('off')

            # Hide remaining axes
            for i in range(n_filters, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            plt.suptitle(f'Layer Activations - {signal_type} - {layer_name}')
            plt.subplots_adjust(top=0.9)

            # Save figure
            plt.savefig(f'results/figures/{signal_type}_{layer_name}_activations.png', dpi=300)

            # Log to W&B
            if self.wandb:
                self.wandb.log({f"{signal_type}_{layer_name}_activations": wandb.Image(fig)})

            plt.close(fig)

        # Store activation results
        self.results['activations'] = activation_results

        return activation_results

    def visualize_neuron_signals(self):
        """
        Visualize the raw signals for the 100 most important neurons
        for each signal type (calcium, deltaf, deconv).
        """
        # Check if raw signals are available in the data dictionary
        required_keys = ['raw_calcium', 'raw_deltaf', 'raw_deconv']
        for key in required_keys:
            if key not in self.data:
                print(f"Missing raw signal: {key}")
                return None

        # Check if feature importance results are available
        if 'feature_importance' not in self.results:
            print("Feature importance not calculated. Run feature_importance() first.")
            return None

        # Create figure with three subplots (one for each signal type)
        fig, axes = plt.subplots(3, 1, figsize=(18, 24))

        # Get the top 100 important neurons for each signal type
        importance_results = self.results['feature_importance']

        for i, signal_type in enumerate(self.signal_types):
            # Try to get importance rankings from different models
            if f"{signal_type}_top_neurons_rf" in importance_results:
                top_neurons = importance_results[f"{signal_type}_top_neurons_rf"]
            elif f"{signal_type}_top_neurons_mlp" in importance_results:
                top_neurons = importance_results[f"{signal_type}_top_neurons_mlp"]
            else:
                print(f"No neuron importance found for {signal_type}")
                continue

            # Get raw signal data
            raw_signal = self.data[f'raw_{signal_type}']

            # Select up to 100 top neurons
            n_neurons = min(100, len(top_neurons))
            selected_neurons = top_neurons[:n_neurons]

            # Extract signals for selected neurons
            neuron_signals = raw_signal[:, selected_neurons].T  # Transpose to get (neurons, timesteps)

            # Plot heatmap of signals
            im = axes[i].imshow(neuron_signals, aspect='auto', cmap='viridis',
                                interpolation='nearest', extent=[0, 2999, 0, n_neurons])


            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_label(f'{signal_type.capitalize()} Signal Intensity')

            # Add labels and title
            axes[i].set_title(f'Top 100 Important Neurons - {signal_type.capitalize()} Signal')
            axes[i].set_xlabel('Time Frames (2999 total)')
            axes[i].set_ylabel('Neuron ID')

            # Add grid lines
            axes[i].grid(False)

        plt.tight_layout()

        # Save figure
        plt.savefig('results/figures/important_neurons_signals.png', dpi=300)

        # Log to W&B
        if self.wandb:
            self.wandb.log({"important_neurons_signals": wandb.Image(fig)})

        plt.close(fig)

        return fig

    def save_results(self, output_dir='results/metrics'):
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'deep_learning_results.json')

        # Convert PyTorch tensors to lists for JSON serialization
        results_json = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_json[key] = value
            elif isinstance(value, np.ndarray):
                results_json[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                results_json[key] = value.cpu().numpy().tolist()
            else:
                results_json[key] = value

        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=4)

        print(f"Results saved to {output_file}")

    def save_models(self, output_dir='models'):
        """Save trained models."""
        os.makedirs(output_dir, exist_ok=True)

        for model_name, model in self.models.items():
            model_file = os.path.join(output_dir, f"{model_name}.pt")
            torch.save(model.state_dict(), model_file)

        print(f"Models saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Load processed data
    data = np.load('data/processed/processed_calcium_data.npz', allow_pickle=True)
    data_dict = dict(data)  # Convert to dictionary

    # Initialize W&B
    wandb_run = wandb.init(project="MIND", name="deep_learning_pytorch_comparison",
                           entity="mirzaeeghazal")

    # Train and evaluate models
    dl = DeepLearningModels(data_dict, wandb_run)
    dl.train_and_evaluate_all(epochs=100, batch_size=32)
    dl.test_best_models()
    dl.visualize_activations()
    dl.visualize_neuron_signals()  # Add the new visualization

    # Save results
    dl.save_results()
    dl.save_models()

    # Finish W&B run
    wandb.finish()
