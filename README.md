# Part - 1

# Part - 2

# Advanced Custom TensorFlow Techniques

This repository demonstrates the use of various advanced TensorFlow techniques for enhancing neural network training, including custom layers, optimizers, training loops, and more. These custom components provide flexibility and allow for detailed control and modification of the training process, which is essential for tackling complex machine learning problems.

## Project Overview

This project includes the implementation of:
1. **Custom Learning Rate Scheduler**: An implementation based on the `OneCycleScheduler`.
2. **Custom Dropout**: Specifically, `MCAlphaDropout` to retain the mean and variance of inputs.
3. **Custom Normalization**: Implementing `MaxNormDense` for constrained weights.
4. **TensorBoard Integration**: For monitoring models and training processes.
5. **Custom Loss Function**: Using `HuberLoss` for more robust error handling.
6. **Custom Activation, Initializer, Regularizer, and Constraint**:
   - `my_softplus` as an activation function.
   - `my_glorot_initializer` for initializing weights.
   - `MyL1Regularizer` for L1 regularization.
   - `my_positive_weights` for ensuring weights remain positive.
7. **Custom Metrics**: Implementing `HuberMetric` for performance evaluation.
8. **Custom Layers**: Including `exponential_layer`, `MyDense`, and `AddGaussianNoise`.
9. **Custom Models**: Building models with custom behaviors like `ResidualRegressor`.
10. **Custom Optimizer**: `MyMomentumOptimizer` to demonstrate momentum mechanics.
11. **Custom Training Loop**: Detailed control over training with live updates on metrics and validations.


## Data Description

We use the Fashion MNIST dataset, which consists of 28x28 grayscale images of 10 fashion categories. The dataset is split into training, validation, and test sets, with preprocessing steps including normalization.

## Features

- **Custom Optimizer**: A momentum optimizer that showcases how to manage internal states and update model parameters manually.
- **Custom Training Loop**: Demonstrates manual batch processing, gradient computation, and application, along with real-time updates of loss and accuracy metrics.
- **Custom Layers and Models**: Shows how to create complex models that might involve residual connections or specific computational patterns not directly supported by pre-built TensorFlow layers.



- The use of custom components allows for tailored machine learning models that can be optimized for specific tasks.
- This approach provides deep insights into the workings of neural network components and TensorFlow's capabilities.

