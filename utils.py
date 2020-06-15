import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io.wavfile
import os

smallValue = 1e-6

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def StringFilter(string, substring, invert = False):
    if invert:
        return [str for str in string if
             not any(sub in str for sub in substring)]
    else:
        return [str for str in string if
             any(sub in str for sub in substring)]

def preprocess_spectrograms(spectrograms):
    """
    Preprocess spectrograms according to approach in WaveGAN paper
    Args:
        spectrograms (torch.FloatTensor) of size (batch_size, mel_bins, time_frames)

    """
    # Remove last time segment
    #spectrograms = spectrograms[:,:,:-1]
    # Normalize to zero mean unit variance, clip above 3 std and rescale to [-1,1]
    means = torch.mean(spectrograms, dim = (1,2), keepdim = True)
    stds = torch.std(spectrograms, dim = (1,2), keepdim = True)
    normalized_spectrograms = (spectrograms - means)/(3*stds + smallValue)
    clipped_spectrograms = torch.clamp(normalized_spectrograms, -1, 1)

    return clipped_spectrograms, means, stds


def save_spec_plot(file_path, spec, title):
    """
    Helper function to save mel-spectrogram plots

    Args:
        file_path (str): save file path
        spec (torch.FloatTensor) with shape [batch_size, , ] containg mel-spectrogram
        title (str): title text
    """

    spec_np = torch.squeeze(spec).cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(spec_np, x_axis = 'time', y_axis='mel', sr = 8000, fmax = 4000, hop_length = 256, cmap = 'magma')
    plt.title(title)
    fig.savefig(file_path)
    plt.close(fig)

def comparison_plot_filtergen(file_path, orig_spec, male_spec, female_spec, orig_title, male_title, female_title):
    orig_spec_np = torch.squeeze(orig_spec).cpu().numpy()
    male_spec_np = torch.squeeze(male_spec).cpu().numpy()
    female_spec_np = torch.squeeze(female_spec).cpu().numpy()
    fig = plt.figure(figsize=(30,8))
    ax1 = fig.add_subplot(131)
    p1 = librosa.display.specshow(orig_spec_np, x_axis = 'time', y_axis='mel', sr = 8000, fmax = 4000, hop_length = 256)
    plt.title(orig_title, fontsize = 18)
    ax2 = fig.add_subplot(132)
    p2 = librosa.display.specshow(male_spec_np, x_axis = 'time', y_axis='mel', sr = 8000, fmax = 4000, hop_length = 256)
    plt.title(male_title, fontsize = 18)
    ax3 = fig.add_subplot(133)
    p3 = librosa.display.specshow(female_spec_np, x_axis = 'time', y_axis='mel', sr = 8000, fmax = 4000, hop_length = 256)
    plt.title(female_title, fontsize = 18)
    fig.savefig(file_path)
    plt.close(fig)

def comparison_plot_pcgan(file_path, orig_spec, filtered_spec, male_spec, female_spec, orig_title, filtered_title, male_title, female_title):
    orig_spec_np = torch.squeeze(orig_spec).cpu().numpy()
    filtered_spec_np = torch.squeeze(filtered_spec).cpu().numpy()
    male_spec_np = torch.squeeze(male_spec).cpu().numpy()
    female_spec_np = torch.squeeze(female_spec).cpu().numpy()
    fig = plt.figure(figsize=(24,24)) # This has to be changed!!
    ax1 = fig.add_subplot(221)
    p1 = librosa.display.specshow(orig_spec_np, x_axis = 'time', y_axis='mel', sr = 8000, fmax = 4000, hop_length = 256, cmap = 'magma')
    plt.title(orig_title, fontsize = 20)
    ax2 = fig.add_subplot(222)
    p2 = librosa.display.specshow(filtered_spec_np, x_axis = 'time', y_axis='mel', sr = 8000, fmax = 4000, hop_length = 256, cmap = 'magma')
    plt.title(filtered_title, fontsize = 20)
    ax3 = fig.add_subplot(223)
    p3 = librosa.display.specshow(male_spec_np, x_axis = 'time', y_axis='mel', sr = 8000, fmax = 4000, hop_length = 256, cmap = 'magma')
    plt.title(male_title, fontsize = 20)
    ax4 = fig.add_subplot(224)
    p4 = librosa.display.specshow(female_spec_np, x_axis = 'time', y_axis='mel', sr = 8000, fmax = 4000, hop_length = 256, cmap = 'magma')
    plt.title(female_title, fontsize = 20)
    fig.savefig(file_path)
    plt.close(fig)



def save_sample(file_path, sampling_rate, audio):
    """Helper function to save sample

    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 22050)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)

def compute_activation_statistics(act):
    #act=act.data.cpu().numpy()
    #print(act.shape)
    mu = np.mean(act, axis=0)

    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def compute_frechet_inception_distance(acts1, acts2):
    mu1, sigma1 = compute_activation_statistics(acts1)
    mu2, sigma2 = compute_activation_statistics(acts2)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
