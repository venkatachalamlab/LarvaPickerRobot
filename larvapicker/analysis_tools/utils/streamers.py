import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from .io import *


class MultiDataStreamer(Dataset):
    def __init__(self, index_path, dev):
        self.index_path = Path(index_path)
        with open(self.index_path) as json_file:
            self.index = json.load(json_file)
        self.dev = dev

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        testset_path = Path(self.index[str(idx)]) / 'test'
        seq_len = len(get_times(testset_path))
        testset = get_slices(testset_path, list(range(seq_len))).astype(float)
        testset = torch.tensor(testset, requires_grad=True).to(self.dev)

        targetset_path = Path(self.index[str(idx)]) / 'target'
        targetset = get_slices(targetset_path, list(range(seq_len))).astype(float)
        targetset = torch.tensor(targetset, requires_grad=False).to(self.dev)

        return [testset, targetset]


class DataStreamer(Dataset):
    def __init__(self, dataset_path, dev, file_name=None):
        self.dataset_path = Path(dataset_path)
        self.file_name = file_name
        self.times = get_times(dataset_path, self.file_name)
        self.dev = dev

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        data = get_slice(self.dataset_path, idx, self.file_name).astype(float)
        data = torch.tensor(data).to(self.dev)
        return data
