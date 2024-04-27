[Colab Part - 1: AugLy](https://colab.research.google.com/drive/1fwen4-scjH1cNqPEx_HS70SxFqkEs-dh?usp=sharing)

[Colab Part - 2: Custom TensorFlow Techniques](https://colab.research.google.com/drive/140cgOHWW9_OaUjKWE2rq5mGt7gYf4_9R?usp=sharing)

[Youtube Link]()

# Part - 1

# Augmentation Experimentation with AugLy

This project aims to showcase various data augmentation and generalization techniques in machine learning, with a focus on practical implementation using TensorFlow and related libraries. The provided Colab notebooks serve as illustrative examples and include A/B tests to evaluate the effectiveness of different techniques. This project explores the capabilities of AugLy, a data augmentation library developed by Facebook AI. AugLy offers a wide range of augmentation techniques for text, images, audio, and video data. In this experiment, we utilize various functionalities of AugLy to augment different types of data.

## Table of Contents

1. [Regularization Techniques](#regularization-techniques)
   - L1 and L2 Regularization
   - Dropout
   - Early Stopping
   - Monte Carlo Dropout
   - Weight Initializations
   - Batch Normalization
   - Custom Dropout and Regularization
   - Callbacks and TensorBoard
   - Keras Tuner

2. [Data Augmentation for Various Data Types](#data-augmentation-for-various-data-types)
   - Image Data
   - Video Data
   - Text Data (NLPAug)
   - Time Series Data
   - Tabular Data
   - Speech Data
   - Document Images

3. [FastAI Data Augmentation Capabilities](#fastai-data-augmentation-capabilities)

## Regularization Techniques

- **L1 and L2 Regularization:** Introduction to L1 and L2 regularization techniques for reducing overfitting in neural networks.
- **Dropout:** Explanation and implementation of dropout regularization to prevent co-adaptation of neurons.
- **Early Stopping:** Utilizing early stopping to prevent overfitting by monitoring validation loss during training.
- **Monte Carlo Dropout:** Using Monte Carlo dropout to estimate model uncertainty and improve generalization.
- **Weight Initializations:** Overview of various weight initialization methods and their suitability for different scenarios.
- **Batch Normalization:** Understanding batch normalization and its role in stabilizing and accelerating training.
- **Custom Dropout and Regularization:** Implementing custom dropout layers and regularization techniques tailored to specific use cases.
- **Callbacks and TensorBoard:** Leveraging callbacks and TensorBoard for monitoring training progress and optimizing model performance.
- **Keras Tuner:** Exploring the Keras Tuner library for hyperparameter tuning and optimization.

## Data Augmentation for Various Data Types

- **Image Data:** Demonstrating common image data augmentation techniques using TensorFlow and Keras.
- **Video Data:** Augmenting video data with AugLy library, including trimming and applying various transformations.
- **Text Data (NLPAug):** Applying text augmentation techniques using the NLPAug library for NLP tasks.
- **Time Series Data:** Illustrating data augmentation methods for time series data to improve model robustness.
- **Tabular Data:** Discussing data augmentation strategies for tabular data and their impact on model performance.
- **Speech Data:** Introduction to audio data augmentation techniques for speech recognition tasks.
- **Document Images:** Exploring data augmentation techniques for document images, including rotation and augmentation.

## FastAI Data Augmentation Capabilities

- Overview of the data augmentation capabilities provided by the FastAI library for image classification tasks.

## Conclusion

This README provides an overview of the project's objectives, key topics covered, and the structure of available resources. Each section contains references to corresponding Colab notebooks or external resources for further exploration.


#
---
#


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

