import pandas as pd
import json
import librosa
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
from utils import *

from librosa.core import load
from librosa.util import normalize

def build_annotation_index(training_files, annotation_file, balanced_genders = False):
    ann_df = pd.read_json(annotation_file, orient = 'index')

    num_files = len(training_files)
    num_speakers = len(ann_df.index)

    ids = [int(f.split('/')[-2]) for f in training_files]
    digits = np.array([int(f.split('/')[-1][0]) for f in training_files])

    if balanced_genders:
        min_gender = ann_df['gender'].value_counts().keys()[-1]
        min_gender_count = np.min(np.array(ann_df['gender'].value_counts()))
        sort_ascend = True
        if min_gender == 'male':
            sort_ascend = False
        ann_df_gender_sorted = ann_df.sort_values(by=['gender'], ascending = sort_ascend)
        ann_df = ann_df_gender_sorted[0:min_gender_count*2]
        data_balanced_index = ann_df.index.to_numpy()
        balanced_ids = ['_' + str(idx) + '_' for idx in data_balanced_index]

        ids = np.array(ids)
        mask = np.isin(ids, data_balanced_index)
        ids = ids[mask]
        digits = digits[0:int(min_gender_count*2*num_files/num_speakers)]

        training_files = StringFilter(training_files,balanced_ids)

    index_gender = np.array([int(ann_df.loc[i]['gender'] == 'male') for i in ids])
    index_digits = digits
    train_file_index = training_files
    index_speaker_id = ids


    return train_file_index, index_gender, index_digits, index_speaker_id


def balanced_annotation_split(training_file_index, annotation_index_gender, annotation_index_digit, annotation_index_speaker_id, split_ratio):
    # Create dataframe from indices and sort by gender then speaker id
    index_df = pd.DataFrame({'filename':training_file_index, 'gender':annotation_index_gender, 'digit':annotation_index_digit, 'speaker_id':annotation_index_speaker_id})
    index_df = index_df.sort_values(by=['gender', 'speaker_id'])
    # Extract speaker ids for respective gender
    female_speaker_ids = index_df.loc[index_df['gender'] == 0].speaker_id.unique()
    male_speaker_ids = index_df.loc[index_df['gender'] == 1].speaker_id.unique()

    #Sample speaker IDs according to split ratio
    female_test_ids = np.random.choice(female_speaker_ids, len(female_speaker_ids)//(split_ratio + 1), replace = False)
    male_test_ids = np.random.choice(male_speaker_ids, len(male_speaker_ids)//(split_ratio + 1), replace = False)
    test_ids = np.concatenate((female_test_ids, male_test_ids), axis = 0)
    train_ids = np.logical_not(index_df['speaker_id'].isin(test_ids))
    # Uniformly sampled ids for our final runs
    # test_ids = [58, 36, 39, 45]
    # train_ids = [12, 26, 28, 43, 47, 52, 56, 57, 59, 60, 34, 35, 37, 38, 40, 41, 44, 46, 48, 49]


    test_df = index_df.loc[index_df['speaker_id'].isin(test_ids)]
    train_df = index_df.loc[index_df['speaker_id'].isin(train_ids)]

    return test_df, train_df, test_ids, train_ids


class AnnotatedAudioDataset(torch.utils.data.Dataset):

    def __init__(self, annotation_index, sampling_rate, segment_length):#, segment_length, sampling_rate):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_file_index = annotation_index['filename'].to_numpy()
        self.annotation_index_gender = annotation_index['gender'].to_numpy()
        self.annotation_index_digit = annotation_index['digit'].to_numpy()
        self.annotation_index_speaker_id = annotation_index['speaker_id'].to_numpy()

        for file in self.audio_file_index:
            audio, sampling_rate = load(file, sr = self.sampling_rate)
            audio = torch.from_numpy(audio).float()
            if audio.shape[0] > 8192:
                print("Corrupted")
                print(audio.shape[0])
            if audio.shape[0] >= self.segment_length:
                audio = audio[0:self.segment_length]
            else:
                num_zeros_to_pad = self.segment_length - audio.shape[0]
                audio = F.pad(audio, (int(np.floor(num_zeros_to_pad / 2)), int(np.ceil(num_zeros_to_pad/2))),"constant")
            save_sample(file, sampling_rate, audio)


    def __getitem__(self, index):
        # Read audio
        filename = self.audio_file_index[index]

        # Some load wav to torch stuff
        audio, sampling_rate = self.load_wav_to_torch(filename)
        if audio.shape[0] >= self.segment_length:
            audio = audio[0:self.segment_length]
        else:
            audio = self.zero_pad_data(audio)

        return audio, self.annotation_index_gender[index], self.annotation_index_digit[index], self.annotation_index_speaker_id[index]

    def __len__(self):
        return len(self.audio_file_index)

    def load_wav_to_torch(self, full_path):
        data, sampling_rate = load(full_path, sr = self.sampling_rate)
        data = 0.95 * normalize(data)
        return torch.from_numpy(data).float(), sampling_rate

    def zero_pad_data(self,data):
        signal_length = data.shape[0]
        num_zeros_to_pad = self.segment_length - signal_length
        start_position = np.random.randint(num_zeros_to_pad)
        padded_signal = F.pad(data, (start_position, num_zeros_to_pad - start_position),"constant")
        return padded_signal

def format_data(file_index, sampling_rate, segment_length):
    for file in file_index:
        audio, sampling_rate = load(file, sr = sampling_rate)
        audio = torch.from_numpy(audio).float()
        if audio.shape[0] >= segment_length:
            audio = audio[0:segment_length]
        else:
            num_zeros_to_pad = segment_length - audio.shape[0]
            start_position = np.random.randint(num_zeros_to_pad)
            audio = F.pad(audio, (start_position, num_zeros_to_pad - start_position),"constant")
        save_sample(file, sampling_rate, audio)
