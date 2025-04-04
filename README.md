# MIND: Multi-signal Interpretable Neural Decoding

## Neural Activity Classification with Classical and Deep Learning Models

![MIND Banner](https://img.shields.io/badge/MIND-Neural%20Decoding-blue)
![Python](https://img.shields.io/badge/Python-3.10-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

MIND (Multi-signal Interpretable Neural Decoding) is a comprehensive framework for analyzing and classifying neural activity patterns from calcium imaging data. The framework integrates multiple processing pipelines for different signal types, implements both classical machine learning and deep learning models, and provides extensive visualization tools for interpretability.

This project is designed to analyze calcium imaging data recorded during mouse reaching tasks, detecting and classifying footstep events based on neural activity patterns.

## Features

- **Multi-signal processing** - Analyze three types of calcium imaging signals:
  - Raw calcium signal
  - ΔF/F (normalized fluorescence)
  - Deconvolved signal (spike inference)

- **Machine learning models**:
  - **Classical ML**:
    - Random Forest
    - Support Vector Machine
    - XGBoost
    - Multilayer Perceptron

  - **Deep Learning** (PyTorch):
    - Fully Connected Neural Networks
    - Long Short-Term Memory (LSTM) Networks
    - 1D Convolutional Neural Networks

- **Performance analysis**:
  - Comprehensive metrics (accuracy, precision, recall, F1-score)
  - Confusion matrices
  - ROC curves
  - Model comparison across signal types

- **Model interpretability**:
  - Feature importance visualization
  - Temporal importance analysis
  - Neuron contribution ranking
  - CNN activation visualization

- **Experiment tracking** via Weights & Biases

## Technical Architecture

MIND uses a sliding window approach to process temporal calcium imaging data, converting it into a supervised learning problem. The pipeline consists of:

1. **Data processing** (`project_setup.py`):
   - Load MATLAB calcium imaging data
   - Load and align behavioral event data
   - Create sliding windows with a configurable size and step
   - Split data into train/validation/test sets

2. **Classical ML models** (`classical_ml_models.py`):
   - Implement and evaluate traditional ML algorithms
   - Extract feature importance data
   - Generate performance metrics

3. **Deep learning models** (`deep_learning_models.py`):
   - Implement PyTorch neural network architectures
   - Configure one-cycle learning rate scheduling
   - Handle class imbalance with weighted loss functions

4. **Visualization** (`visualize_results.py`):
   - Generate comparative performance visualizations
   - Visualize feature importance and model interpretability
   - Create summary reports of model performance

5. **Orchestration** (`main.py`):
   - Control execution flow
   - Configure experiment parameters
   - Handle experiment tracking

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for deep learning)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MIND.git
   cd MIND
   Create and activate a virtual environment:
   ```
2. Create and activate a virtual environment:
  ```bash
  python -m venv myenv
  source myenv/bin/activate  # Linux/Mac
  # or
  myenv\Scripts\activate  # Windows
 ```
3. Install dependencies:
  ```bash
  pip install -r requirements.txt
 ```

4. Configure Weights & Biases (optional):
  ```bash
  wandb login
 ```
Usage
Data Preparation
Place your calcium imaging data files and behavioral data files in the appropriate directories:
```bash
MIND/
├── data/
│   ├── raw/
│   │   ├── your_calcium_data.mat
│   │   └── your_behavior_data.xlsx
```

Running the Pipeline
Execute the main script with appropriate arguments:
```bash
python main.py --matlab_file data/raw/your_calcium_data.mat \
               --behavior_file data/raw/your_behavior_data.xlsx \
               --window_size 10 \
               --step_size 2 \
               --epochs 50 \
               --batch_size 32 \
               --project_name "MIND" \
               --experiment_name "experiment001"
```
Command Line Arguments

--matlab_file: Path to MATLAB file containing calcium imaging data
--behavior_file: Path to behavior data Excel file
--window_size: Window size for sliding window approach (default: 10)
--step_size: Step size for sliding window (default: 2)
--epochs: Number of epochs for deep learning models (default: 50)
--batch_size: Batch size for deep learning models (default: 32)
--project_name: W&B project name (default: "MIND")
--experiment_name: Name for the W&B experiment
--device: Device to use for PyTorch training (default: auto-detect)
--num_workers: Number of workers for data loading (default: 4)

Execution Flags

--skip_processing: Skip data processing step
--skip_classical: Skip classical ML models
--skip_deep_learning: Skip deep learning models
--skip_visualization: Skip results visualization

Results and Visualization
After running the pipeline, results and visualizations are saved to:

```bash
MIND/
├── results/
│   ├── figures/
│   │   ├── performance_comparison.png
│   │   ├── signal_type_comparison.png
│   │   ├── model_comparison.png
│   │   └── ...
│   ├── metrics/
│   │   ├── classical_ml_results.json
│   │   ├── deep_learning_results.json
│   │   └── summary_report.json
```


