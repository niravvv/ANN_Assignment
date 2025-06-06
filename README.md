# Feedforward ANN from Scratch – COMP2009 Assignment 3

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dpH2-OjBRBxh2fjAgX3g5RWd5Ei5n5S4?usp=sharing)

This project implements two artificial neural networks (ANNs) from first principles using only NumPy. The aim is to predict penalty scores associated with 10-task-to-employee assignment mappings. All training, evaluation, and visualisation were performed in Google Colaboratory using core Python libraries, without high-level ML frameworks such as TensorFlow or PyTorch.

## Project Overview

- **Goal**: Predict penalty values for PSO-generated task-employee mappings using structured features from task and employee profiles.
- **Model A**: One hidden layer with 256 neurons.
- **Model B**: Two hidden layers, each with 128 neurons.
- **Loss Function**: Mean Squared Error (MSE).
- **Optimisation**: Mini-batch stochastic gradient descent implemented manually.

## Files

- `ann_assignment_21497502.py` – Complete implementation script including:
  - Data loading and preprocessing
  - Manual encoding using one-hot vectors and numeric attributes
  - Neural network class definitions for Models A and B
  - Forward and backward propagation
  - Configurable training loop with grid search support
  - CSV logging and result visualisation
- `mappings_with_penalty.csv` – Input dataset containing 100 PSO-generated mappings and associated penalty scores.
- `grid_search_results.csv` – Summary of all grid search configurations and final metrics.
- `loss_log_modelA.csv` / `loss_log_modelB.csv` – Epoch-wise training and validation losses.
- `test_predictions_modelA.csv` / `test_predictions_modelB.csv` – Final predictions vs true values on the test set.

## Features

- End-to-end ANN implementation from scratch
- Supports both ReLU and Sigmoid activations
- Flexible grid search for:
  - Learning rate: 0.01, 0.001, 0.0001
  - Batch size: 8, 16, 32
  - Epochs: 100, 150, 200
  - Activation function: ReLU or Sigmoid
- Loss curves, timing plots, and comparative analysis
- Logs key metrics and outputs to CSV for reproducibility and appendix inclusion

## Running Instructions

1. Click the **"Open in Colab"** badge above to launch the notebook directly.
2. Upload:
   - `ann_assignment_21497502.py`
   - `mappings_with_penalty.csv`
3. Ensure built-in libraries (NumPy, pandas, matplotlib, time) are available—these are pre-installed in Colab.
4. Run the notebook cells top to bottom. Output files will be saved to the working directory.
5. Adjust hyperparameters at the top of the script or modify the config dictionary before execution to re-run grid search.

**Note**: `np.random.seed(0)` is used throughout to ensure reproducibility in data shuffling and weight initialisation.

## Environment

- **Platform**: Google Colaboratory
- **Python**: 3.10
- **Backend**: CPU runtime (T4 GPU was requested but unavailable due to usage limits)
- **Libraries**: NumPy, pandas, matplotlib, time

## Development Notes

- All training was conducted on a CPU runtime due to Colab GPU access limits. However, the small dataset size ensured acceptable performance.
- The project was developed and executed in a single active session, with Colab’s built-in history used during early iterations.
- Git version control was applied at later stages for recordkeeping and final submission.
- All graphs were generated with Matplotlib and embedded inline.

