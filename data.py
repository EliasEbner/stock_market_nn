import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, long_series, chunk_size, output_size):
        self.data = long_series
        self.chunk_size = chunk_size
        self.output_size = output_size
        self.num_chunks = len(long_series) - (chunk_size + 11)

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        chunk = torch.tensor(self.data[idx:idx+self.chunk_size])
        target = torch.tensor(self.data[idx+self.chunk_size+10])
        return chunk, target
