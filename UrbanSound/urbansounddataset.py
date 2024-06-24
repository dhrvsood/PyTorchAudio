# creating a custom PyTorch dataset from the UrbanSound8k dataset
import os
import torch

from torch.utils.data import Dataset
import pandas as pd
import torchaudio
try:
    import config
except ImportError:
    config = None

# constants
ANNOTATIONS_FILE = getattr(config, 'ANNOTATIONS_FILE', None)
AUDIO_DIR = getattr(config, 'AUDIO_DIR', None)
SAMPLE_RATE = 16000

print(f"Annotations Path: {ANNOTATIONS_FILE}")
print(f"Audio Directory: {AUDIO_DIR}")

class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, 
                 target_sample_rate):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        
        # using load functionality in torchaudio
        # signal - waveform/timeseries of audio file
        # sr - sample rate
        signal, sr = torchaudio.load(audio_sample_path)

        # mixing signal down to mono and resampling
        signal = self._mix_down_if_necessary(signal)
        signal = self._resample_if_necessary(signal, sr)
        
        # transform waveform into Mel Spectrogram
        signal = self.transformation(signal)
        return signal, label

    """
    Private methods
    """
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

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
        SAMPLE_RATE
    )
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
    print(signal)