# creating a custom PyTorch dataset from the UrbanSound8k dataset
import os

from torch.utils.data import Dataset
import pandas as pd
import torchaudio

# constants
try:
    import config
except ImportError:
    config = None

ANNOTATIONS_FILE = getattr(config, 'ANNOTATIONS_FILE', None)
AUDIO_DIR = getattr(config, 'AUDIO_DIR', None)

print(f"Annotations Path: {ANNOTATIONS_FILE}")
print(f"Audio Directory: {AUDIO_DIR}")

class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        # using load functionality in torchaudio
        # signal - waveform/timeseries of audio file
        # sr - sample rate
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label

    def _get_audio_sample_path(self, index):
        # annotations.iloc[index, 5] - folder name
        # annotations.iloc[index, 0] - file name
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path
    
    def _get_audio_sample_label(self, index):
        # returning classID
        return self.annotations.iloc[index, 6]

if __name__ == "__main__":
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
    print(signal, label)