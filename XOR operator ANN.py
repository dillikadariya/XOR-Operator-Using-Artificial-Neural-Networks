from sklearn.neural_network import MLPClassifier
import numpy as np

# Sample data for XOR operation
X = [[0, 0], [1, 1], [0, 1], [1, 0]]
y = [0, 0, 1, 1]  # XOR truth table outputs

# Implementing Multi-Layer Perceptron
clf = MLPClassifier(
    solver='lbfgs',  # solver that performs gradient descent
    alpha=1e-5,      # Regularization parameter
    hidden_layer_sizes= (3,),  # single hidden layer with 3 neurons, 2 neurons gives 50% accuracy 
    random_state=42,  # For reproducibility
)

# Train the model
clf.fit(X, y)

# Make predictions on the training data to verify
predictions = clf.predict(X)
print("Predictions:", predictions)
print("Actual:     ", y)
print("Accuracy:   ", np.mean(predictions == y)) # element wise comparison between pred and actual

# Test with new predictions (same as training data in this case)
test_predictions = clf.predict([[1, 0], [0, 0], [1, 1], [0, 1]])
print("\nTest predictions:", test_predictions)
print("Expected:       ", [1, 0, 0, 1])
