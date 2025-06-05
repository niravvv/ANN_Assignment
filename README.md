# Feedforward ANN from Scratch – COMP2009 Assignment 3

This project implements two artificial neural networks (ANNs) from first principles using NumPy. The objective is to predict penalty scores associated with task-to-employee assignment mappings. All training, evaluation, and visualisation were performed in Google Colaboratory using only core Python libraries, in accordance with the assignment specifications.

## Project Overview

- Objective: Predict penalty values for 10-task assignment mappings based on structured task and employee attributes.
- Model A: One hidden layer with 256 neurons.
- Model B: Two hidden layers, each with 128 neurons.
- No high-level machine learning frameworks (e.g. TensorFlow, PyTorch) were used.

## Files

- `ann_assignment_21497502.py`: Main Python script containing all code components:
  - Data loading and preprocessing
  - Neural network class definitions
  - Manual implementation of forward and backward propagation
  - Mini-batch stochastic gradient descent
  - Hyperparameter grid search
  - Plotting and result visualisation
- `mappings_with_penalty.csv`: Dataset file with 100 ACO-generated task-employee mappings and corresponding penalty scores (to be uploaded into Colab file pane)

## Features

- Manual ANN implementation using NumPy
- Activation functions: ReLU and Sigmoid
- Mean Squared Error (MSE) as the loss function
- Mini-batch gradient descent with configurable batch size
- Support for hyperparameter tuning via grid search:
  - Learning rates: 0.01, 0.001, 0.0001
  - Batch sizes: 8, 16, 32
  - Activation functions: ReLU, Sigmoid
  - Epochs: 100, 150, 200
- Generates eight result plots:
  - Epoch vs Loss (train and validation)
  - Learning Rate vs Final Loss
  - Activation Function vs Final Loss
  - Batch Size vs Epoch Time (average)

## Running Instructions

1. Open Google Colaboratory.
2. Upload the Python script and `mappings_with_penalty.csv`.
3. Ensure NumPy, pandas, and matplotlib are available (pre-installed in Colab).
4. Run all cells in sequence to execute data preprocessing, model training, and plotting.
5. Modify hyperparameters or model selection by editing the appropriate cells in the notebook.

Note: `np.random.seed(0)` is used throughout to ensure reproducibility of results.

## Environment

- Platform: Google Colaboratory
- Python version: 3.x
- Backend: CPU runtime only (T4 GPU was requested but unavailable due to usage limits)
- Libraries used: NumPy, pandas, matplotlib, time

## Development Notes

- No hardware acceleration was used; all training was performed on CPU.
- GitHub version control was introduced after the codebase reached a functional stage. Early versions were managed using Colab’s internal history and checkpointing features.
- Code was developed in a single main session, with minimal interruption.

## Learning Objectives

- Implement a feedforward ANN from scratch
- Understand the mechanics of forward and backward propagation
- Apply gradient descent with mini-batching to real-world regression problems
- Explore the effects of activation functions, learning rates, and batch sizes on training stability and accuracy
- Build reproducible experimental pipelines for structured data
