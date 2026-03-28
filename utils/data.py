import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, x, y):
        self.feature_size = x.shape[1]
        self.label_size = y.shape[1]
        self.len = x.shape[0]
        self.x_data = torch.as_tensor(x, dtype=torch.float)
        self.y_data = torch.as_tensor(y, dtype=torch.float)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    

class Dataset_t(Dataset):
    def __init__(self, x, y, y2):
        self.feature_size = x.shape[1]
        self.label_size = y.shape[1]
        self.len = x.shape[0]
        self.x_data = torch.as_tensor(x, dtype=torch.float)
        self.y_data = torch.as_tensor(y, dtype=torch.float)
        self.y_data2 = torch.as_tensor(y2, dtype=torch.float)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.y_data2[index]

    def __len__(self):
        return self.len

class Dataset_text(Dataset):
    def __init__(self, x, z, y):
        self.len = x.shape[0]
        self.x_data = torch.as_tensor(x, dtype=torch.float)
        self.y_data = torch.as_tensor(y, dtype=torch.float)
        self.z_data = torch.as_tensor(z, dtype=torch.float)

    def __getitem__(self, index):
        return self.x_data[index], self.z_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
class Dataset_pipline(Dataset):
    def __init__(self, x, y, texts):
        self.len = x.shape[0]
        self.x_data = torch.as_tensor(x, dtype=torch.float)
        self.y_data = torch.as_tensor(y, dtype=torch.float)
        self.texts = texts

    def __getitem__(self, index):
        return {"x_data":self.x_data[index], 
                "y_data":self.y_data[index], 
                "text":self.texts[index]}

    def __len__(self):
        return len(self.x_data)
    
class Dataset_onlytext(Dataset):
    def __init__(self, y, texts):
        self.len = y.shape[0]
        self.y_data = torch.as_tensor(y, dtype=torch.float)
        self.texts = texts

    def __getitem__(self, index):
        return {"y_data":self.y_data[index], 
                "text":self.texts[index]}

    def __len__(self):
        return len(self.y_data)

class Dataset_embed(Dataset):
    def __init__(self, x):
            self.x= x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)

