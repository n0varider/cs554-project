import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class SeriesDataset(Dataset):
    def __init__(self, df, target, seq_len=7):
        self.df = df.copy()
        self.target = target
        self.df = self.df.dropna()
        self.seq_len = seq_len

        self.features = torch.tensor(self.df.drop(columns=[self.target]).values, dtype=torch.float32)
        self.targets = torch.tensor(self.df[target].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df) - self.seq_len

    def __getitem__(self, idx):
        seq_x = self.features[idx:idx + self.seq_len]  
        seq_y = self.targets[idx + self.seq_len]       # Target at time `idx + seq_len`
        return seq_x, seq_y
