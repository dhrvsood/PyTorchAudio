import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 1. download dataset
# 2. create data loader
# 3. build model
# 4. train model
# 5. save trained model

# constants
BATCH_SIZE = 128

class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),      # input layer
            nn.ReLU(),                  # Rectified linear unit
            nn.Linear(256, 10)          # output layer
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions

# download datasets
def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, test_data

if __name__ == "__main__":
    # download MNIST dataset
    train_data, _ = download_mnist_datasets()

    # create data loader for train set
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # building model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")
    feed_forward_net = FeedForwardNet().to(device)