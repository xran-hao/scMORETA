import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
   
    def __init__(self, features, labels):
        if not isinstance(features, torch.Tensor):
            self.x = torch.FloatTensor(features)
        else:
            self.x = torch.from_numpy(features).float()
            
        if not isinstance(labels, torch.Tensor):
            self.y = torch.LongTensor(labels)
        else:
            self.y = labels.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
