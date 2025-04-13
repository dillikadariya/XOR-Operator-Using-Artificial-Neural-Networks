from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

# XOR data
X = [[0, 0], [1, 1], [0, 1], [1, 0]]
y = [0, 0, 1, 1]

# Prepare to store accuracy values
layer_range = range(1, 11)  # number of layers
neuron_range = range(1, 11)  # number of neurons per layer
accuracy_matrix = np.zeros((len(layer_range), len(neuron_range)))

# Loop over combinations of layer count and neurons per layer
for i, num_layers in enumerate(layer_range):
    for j, neurons_per_layer in enumerate(neuron_range):
        # Create hidden layer configuration
        hidden_layers = tuple([neurons_per_layer] * num_layers)
        
        # Define and train the model
        clf = MLPClassifier(
            solver='lbfgs',
            alpha=1e-5,
            hidden_layer_sizes=hidden_layers,
            random_state=42,
            max_iter=1000
        )
        clf.fit(X, y)
        
        # Compute accuracy
        predictions = clf.predict(X)
        accuracy = np.mean(predictions == y)
        accuracy_matrix[i, j] = accuracy

# Plotting the heatmap
plt.figure(figsize=(5, 4))
plt.imshow(accuracy_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Accuracy')
plt.xticks(np.arange(10), neuron_range)
plt.yticks(np.arange(10), layer_range)
plt.xlabel('Neurons per Layer')
plt.ylabel('Number of Layers')
plt.title('XOR Accuracy for Varying Hidden Layers and Neurons')
plt.show()
