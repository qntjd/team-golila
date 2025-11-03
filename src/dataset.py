import json
import glob
import torch
from torch.utils.data import Dataset

class MIDIDataset(Dataset):
    def __init__(self, token_folder, seq_len=512):
        self.files = glob.glob(f"{token_folder}/*.json")
        self.seq_len = seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = json.load(open(self.files[idx]))
        ids = data["ids"]
        x = ids[:-1][:self.seq_len]
        y = ids[1:][:self.seq_len]
        return torch.tensor(x), torch.tensor(y)
