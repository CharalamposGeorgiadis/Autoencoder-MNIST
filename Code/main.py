from IPython.display import clear_output
import time
from autoencoder import *
from loadDataset import load_dataset
from plotImages import plotStackedImages

# Specifying a random seed in order to have the same random environment for every run of the program
random_state = 42
torch.manual_seed(random_state)

# Initializing Autoencoder parameters
model = Autoencoder(784)
learning_rate = 0.001
epochs = 20
criterion = nn.MSELoss()
batch_size = 64
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initializing the split percentage of the training set into a training and a validation set
split = 0.9
# Loading the training, validation and test sets
whole_train_set, train_set, val_set, test_set = load_dataset(split=split, batch_size=batch_size)

# Initializing the minimum validation loss
best_val_loss = float('inf')

# Training the autoencoder
clear_output()
print("AUTOENCODER TRAINING...")
total_time = time.time()
for epoch in range(epochs):
    epoch_time = time.time()
    train_loss = train(model, train_set, criterion, optimizer, epoch + 1)
    val_loss = evaluate(model, val_set, criterion)
    epoch_time = time.time() - epoch_time
    # If the model achieved a validation loss lower than the previous best one, the current model parameters are saved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
    clear_output()
    print(f"Epoch {epoch + 1} / {epochs} | Epoch Time: {round(epoch_time * 1000)} ms")
    print(f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

total_time = time.time() - total_time
print(f"Total Training Time: {round(total_time * 1000)} ms")

# Loading the model parameters that achieved the lowest validation loss during training
model.load_state_dict(torch.load('tut1-model.pt'))

# Evaluating the model on the whole training set and on the test set
train_loss = evaluate(model, whole_train_set, criterion)
test_loss = evaluate(model, test_set, criterion)
clear_output()
print(f"Whole Training Set Loss: {train_loss:.3f} | Test Set Loss: {test_loss:.3f}")

# Passing the first 15 images of the test set through the autoencoder in order to plot the original ones next to their
# reconstructed counterparts
original_images = []
reconstructed_images = []
labels = []
for data in test_set:
    x, y = data
    for i in range(len(x)):
        original_images.append(x[i].detach().clone().numpy().reshape(28, 28))
        reconstructed_images.append(model.forward(x[i].reshape(-1, 784)).detach().clone().numpy().reshape(28, 28))
        labels.append(y[i])
    if i == 14:
        break
# Plotting the reconstructed images
figure_title = "Original and Reconstructed Test Images Side-by-Side"
plotStackedImages(original_images, reconstructed_images, labels, figure_title)
