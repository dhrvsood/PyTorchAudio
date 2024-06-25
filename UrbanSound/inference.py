import torch
import torchaudio
from cnn import CNNetwork
from urbansounddataset import UrbanSoundDataset
from train import ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

def predict(model, input, target, class_mapping):
    model.eval()    
    # using a context manager so model doesn't calculate any gradients
    with torch.no_grad():
        predictions = model(input)
        # predictions are of object Tensor (1, 10)
        # confidence levels where sums are 1 (softmax)
        # want to get the max value (most likely confidence)
        predicted_index = predictions[0].argmax()
        # map this to the relative class mapping
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

if __name__ == "__main__":
    # 1. load back the model
    cnn = CNNetwork()
    state_dict = torch.load("cnn.pth")
    cnn.load_state_dict(state_dict)

    # 2. load Urban sound dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    
    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE, 
        AUDIO_DIR, 
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        "cpu"
    )

    # 3. get sample from urban sound dataset for inference 
    input, target = usd[0][0], usd[0][1]    # currently 3d [num_channels, fr_axis, time_axis]
    # needs 4 dimension - batch_size
    input.unsqueeze_(0)

    # 4. make an inference 
    predicted, expected = predict(cnn, input, target, class_mapping)

    print(f"Predicted: {predicted}, Expected: '{expected}'")
