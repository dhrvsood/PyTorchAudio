import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from urbansounddataset import UrbanSoundDataset
from cnn import CNNetwork
import config

# constants
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = getattr(config, 'ANNOTATIONS_FILE', None)
AUDIO_DIR = getattr(config, 'AUDIO_DIR', None)
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # 1. calculate loss 
        predictions  = model(inputs)
        loss = loss_fn(predictions, targets)

        # 2. backpropogate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("--------------------")
    print("Training complete.")


if __name__ == "__main__":
    # instantiate dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE, 
        AUDIO_DIR, 
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
    )

    # create data loader for train set
    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)

    # build model and assign it to device
    cnn = CNNetwork().to(device)
    print(cnn)

    # instantiate loss fn and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # training model
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    # saving model
    torch.save(cnn.state_dict(), "cnn.pth")
    print("Model trained and stored at cnn.pth")