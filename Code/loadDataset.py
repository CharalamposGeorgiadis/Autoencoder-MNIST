import copy
import torch.utils.data as data
from torchvision import datasets, transforms


# Function that loads the dataset
# param split: Float that contains the train-validation split percentage
# param batch_size: Batch size
# return: Training, split training, validation and test sets
def load_dataset(split, batch_size):
    # Initializing the transform for the dataset
    transform = transforms.Compose([transforms.ToTensor()])

    # Loading the training set
    train_set = datasets.MNIST(root='dataset',
                               train=True,
                               download=True,
                               transform=transform)

    # Loading the test set
    test_set = datasets.MNIST(root='dataset',
                              train=False,
                              download=True,
                              transform=transform)

    # Creating a copy of the training set in order to use it for evaluating/training the KNN, NCC and SVM classifiers
    # since they will not use a validation set
    whole_train_set = copy.deepcopy(train_set)

    # Splitting the whole training set into a smaller training set and a validation set
    n_train_examples = int(len(train_set) * split)
    n_val_examples = len(train_set) - n_train_examples
    train_set, val_set = data.random_split(train_set, [n_train_examples, n_val_examples])

    # Creating a DataLoader for the whole training set
    whole_train_set = data.DataLoader(whole_train_set, shuffle=True, batch_size=batch_size)

    # Creating a DataLoader for the split training set
    train_set = data.DataLoader(train_set, shuffle=True, batch_size=batch_size)

    # Creating a DataLoader for the validation set
    val_set = data.DataLoader(val_set, batch_size=batch_size)

    # Creating a DataLoader for the test set
    test_set = data.DataLoader(test_set, batch_size=batch_size)

    return whole_train_set, train_set, val_set, test_set
