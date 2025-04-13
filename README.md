# XOR Neural Network with MLPClassifier 

This project demonstrates how a binary XOR (exclusive OR) function can be modeled using an artificial neural network built with Scikit-learn's `MLPClassifier`. It explores how different network architectures (number of hidden layers and neurons per layer) affect prediction accuracy, visualized through a heatmap.

---

## Overview

- **Dataset**: XOR truth table
- **Model**: Multi-layer Perceptron (MLP)
- **Library**: `scikit-learn`
- **Solver**: `lbfgs` (quasi-Newton optimization)
- **Architecture Tuning**: Number of hidden layers and neurons ranged from 1 to 10
- **Visualization**: Accuracy heatmap using `matplotlib`

---

## XOR Problem

The XOR function returns `True` (1) if the inputs are different:

| Input 1 | Input 2 | XOR Output |
|---------|---------|------------|
|   0     |    0    |     0      |
|   1     |    1    |     0      |
|   0     |    1    |     1      |
|   1     |    0    |     1      |

---

## Heatmap of Accuracy

The program evaluates multiple configurations of hidden layers and neurons, capturing their prediction accuracies on the XOR dataset. The results are shown in a heatmap to visualize which architectures perform best.

---

## How It Works

1. Imports MLPclassifier model from Scikit-learn 
2. Loops through combinations of 1–10 hidden layers and 1–10 neurons per layer.
3. Trains an `MLPClassifier` on the scaled XOR data.
4. Stores the accuracy for each configuration.
5. Displays a heatmap of accuracies using `matplotlib`.

---

## Requirements

- `numpy`
- `scikit-learn`
- `matplotlib`
