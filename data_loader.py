import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data.sampler import SubsetRandomSampler

from transform import audio_transform


class UrbanSoundDataset(torch.utils.data.Dataset):
    def __init__(self, paths, info_df, transform=None):
        self.transform = transform
        self.info_df = info_df
        self.file_list = []
        for path in paths:
            self.file_list.extend(glob(os.path.join(path, "*.wav")))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        _, filename = os.path.split(filepath)
        label = int(self.info_df[self.info_df["slice_file_name"] == filename]["classID"])
        waveform, sample_rate = torchaudio.load(filepath)
        if self.transform:
            waveform = self.transform(waveform) 
        return waveform, label


def load_data(data_path, batch_size, shuffle_dataset, random_seed=42):
    train_paths = [os.path.join(data_path, f"fold{i}") for i in range (1, 10)]
    test_paths = [os.path.join(data_path, "fold10")]

    info_df = pd.read_csv(os.path.join(data_path, "UrbanSound8K.csv"))
    train_data = UrbanSoundDataset(train_paths, info_df, audio_transform)
    test_data = UrbanSoundDataset(test_paths, info_df, audio_transform)

    # Creating data indices for training and validation splits:
    train_indices = list(range(len(train_data)))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(train_indices)

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader