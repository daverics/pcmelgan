import scipy.io.wavfile
import torch
import numpy
import matplotlib.pyplot as plt
import librosa
import librosa.display


def save_sample(file_path, sampling_rate, audio):
    """Helper function to save sample

    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 22050)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)

def save_spec_plot(file_path, spec):
    """
    Helper function to save mel-spectrogram plots

    Args:
        file_path (str): save file path
        spec (torch.FloatTensor) with shape [batch_size, , ] containg mel-spectrogram
    """

    spec_np = torch.squeeze(spec).cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(spec_np, x_axis='time', y_axis='mel', sr = 16000, fmax = 8000)
    fig.savefig(file_path)
    plt.close(fig)

def preprocess_spectrograms(spectrograms):
    """
    Preprocess spectrograms according to approach in WaveGAN paper
    Args:
        spectrograms (torch.FloatTensor) of size (batch_size, mel_bins, time_frames)

    """
    # Remove last time segment
    spectrograms = spectrograms[:,:,:-1]
    # Normalize to zero mean unit variance, clip above 3 std and rescale to [-1,1]
    means = torch.mean(spectrograms, dim = 2, keepdim = True)
    stds = torch.std(spectrograms, dim = 2, keepdim = True)
    normalized_spectrograms = (spectrograms - means)/(3*stds + smallValue)
    clipped_spectrograms = torch.clamp(normalized_spectrograms, -1, 1)

    return clipped_spectrograms
