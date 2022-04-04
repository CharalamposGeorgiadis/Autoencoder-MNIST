# Autoencoder-MNIST
***NOTE: Extract the "dataset.zip" file before running the code!***

Python project that trains and evaluates an Autoencoder on the MNIST dataset using PyTorch.

Matplotlib is used in order to display the first 15 images of the test set side-by-side with their reconstructed counterparts.

The purpose of this project was to train a model whose encoded output (reduced dimensionality) could be used to train another classifier more efficiently than the classifier itself when trained with the original dataset.

The models that were tested were the 1-Nearest Neighbor, 3-Nearest Neighbors, Nearest Class Centroid classifiers and a Multilayer Perceptron. The codes for these models can be found in:
1. KNN-NCC: [GitHub Pages](https://github.com/XarisGeorgiadis/KNN-KNC_Classifiers)
2. MLP: [GitHub Pages](https://github.com/XarisGeorgiadis/MLP_From_Scratch)

71 different hyperparameters were tested in order to find the Autoencoder that could achieve the highest possible test accuracy out of these combinations.

That model was trained with the following hyperparameters:
1. Learning rate = 0.001
2. Number of hidden layers: 3
3. Number of hidden layer neurons (144 -> 64 -> 144)
4. Batch size = 64
5. Number of epochs = 20
6. Loss = MSE
7. Optimizer = Adam
8. Activation functions: Sgimoid on the output layer, ReLU on all the others.

The performance of the Autoencoder when evaluated on the whole training set and on the test set is listed below:

| Train Loss | Test Loss | Execution Time (Avg.) |
| ------------- | ------------- | -------------|
| 0.003  | 0.003 | ~140565 ms |

The performances of the other classifiers when evaluated on the on the test set are listed below:

| Classifier | Test Accuracy | Execution Time (Avg.) |
| ------------- | ------------- | -------------|
| Original 1-Nearest Neighbor  | 96.91% | ~2200000 ms |
| Encoded 1-Nearest Neighbor | 97.07% | ~785791 ms 
| Original 3-Nearest Neighbors | 97.17% | ~2200000 ms |
| Encoded 3-Nearest Neighbors | 97.06% | ~785791 ms |
| Original Nearest Class Centroid | 82.03% | ~780 ms |
| Encoded Nearest Class Centroid | 79.32% | ~141084 ms |
| Original Multilayer Perceptron | 98.59% | ~340510 ms |
| Encoded Multilayer Perceptron | 98.73% | ~345642 ms |

The execution times for the Encoded classifiers are equal to: Autoencoder Execution Time + Classifier Execution Time
