import torch
import torch.nn as nn
from tqdm import tqdm


# Training method of the autoencoder
# param model: Current autoencoder
# param dataLoader: Split training set DataLoader
# param criterion: Loss criterion
# param optimizer: Optimization method
# param epoch: Current epoch
# return: Loss of the current epoch
def train(model, dataLoader, criterion, optimizer, epoch):
    total_loss = 0
    for data in tqdm(dataLoader, desc="Training Epoch " + str(epoch), leave=False):
        x, _ = data
        x = x.reshape(-1, 784)
        optimizer.zero_grad()
        reconstructed = model(x)
        loss = criterion(reconstructed, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataLoader)


# Evaluation method of the autoencoder
# param model: Current or final, trained autoencoder
# param dataLoader: Validation or test set DataLoader
# param criterion: Loss criterion
# return: Loss of the current epoch or of the final, trained autoencoder
def evaluate(model, dataLoader, criterion):
    with torch.no_grad():
        total_loss = 0
        for data in dataLoader:
            x, _ = data
            x = x.reshape(-1, 784)
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            total_loss += loss.item()
    return total_loss / len(dataLoader)


# Class that represents an autoencoder
class Autoencoder(nn.Module):

    # Constructor
    # param input_neurons: The number of neurons in the input and output layers
    def __init__(self, input_neurons):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_neurons, 144),
            torch.nn.ReLU(),
            torch.nn.Linear(144, 64)
        )
        self.decoder = nn.Sequential(
            torch.nn.Linear(64, 144),
            torch.nn.ReLU(),
            torch.nn.Linear(144, input_neurons),
            torch.nn.Sigmoid()
        )

    # Feed forward method of the autoencoder
    # param x: Input samples
    # return: Decoded input data
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
