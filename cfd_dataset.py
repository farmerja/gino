import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()
        if isinstance(x, str):
            tensor_dict = torch.load(x)
            self.mean = tensor_dict['mean']
            self.std = tensor_dict['std']
        else:
            self.mean = torch.mean(x, dim=(0, 1))
            self.std = torch.std(x, dim=(0, 1))
        self.eps = eps
        self.time_last = time_last

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                std = self.std[..., sample_idx] + self.eps
                mean = self.mean[..., sample_idx]
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class CFDDataset(Dataset):
    def __init__(self, data_path, file_list, x_normalizer=None, y_normalizer=None):
        self.data_path = data_path
        self.file_list = file_list
        
        # Use provided normalizers or initialize new ones
        if x_normalizer is None or y_normalizer is None:
            self.x_normalizer, self.y_normalizer = self._init_normalizers()
        else:
            self.x_normalizer, self.y_normalizer = x_normalizer, y_normalizer

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_list[idx])
        data = torch.load(file_path)
        pressure = data['pressure']
        smoke = data['smoke']
        
        if not isinstance(pressure, torch.Tensor):
            pressure = torch.tensor(pressure, dtype=torch.float32)
        if not isinstance(smoke, torch.Tensor):
            smoke = torch.tensor(smoke, dtype=torch.float32)
        
        # Reshape pressure to (n_in, in_channels=1)
        x = pressure.view(1, -1, 1)
        # Reshape smoke to (n_in, out_channels=1)
        y = smoke.view(1, -1, 1)
        
        # Normalize x and y
        x = self.x_normalizer.encode(x)
        y = self.y_normalizer.encode(y)
        
        # Create input geometry (mesh coordinates)
        input_geom = self.create_mesh_coordinates(pressure.shape)
        input_geom = input_geom.view(1, -1, 3)  # shape (1, n_in, gno_coord_dim)
        
        # Create latent queries (regular grid in [0,1]^3)
        latent_queries = self.create_latent_queries(pressure.shape)
        latent_queries = latent_queries.unsqueeze(0)  # shape (1, n_gridpts_1, n_gridpts_2, n_gridpts_3, gno_coord_dim)
        
        # Use the same coordinates for output queries
        output_queries = input_geom.squeeze(0)  # shape (n_out, gno_coord_dim)
        
        return x, y, input_geom, latent_queries, output_queries

    def create_mesh_coordinates(self, shape):
        coords = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing='ij')
        return torch.tensor(np.stack(coords, axis=-1), dtype=torch.float32)

    def create_latent_queries(self, shape):
        return self.create_mesh_coordinates(shape)

    def _init_normalizers(self):
        # Load all data to compute mean and std
        all_x = []
        all_y = []
        for file_name in self.file_list:
            file_path = os.path.join(self.data_path, file_name)
            data = torch.load(file_path)
            pressure = data['pressure'].view(1, -1, 1)
            smoke = data['smoke'].view(1, -1, 1)
            all_x.append(pressure)
            all_y.append(smoke)
        
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        
        x_normalizer = UnitGaussianNormalizer(all_x)
        y_normalizer = UnitGaussianNormalizer(all_y)
        
        return x_normalizer, y_normalizer

class CFDDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y, _, _, output_queries = self.dataset[idx]
        return x, y, output_queries

def create_datasets(data_path, num_train_samples, num_val_samples, shuffle=False, seed=None):
    # Initialize normalizers using the entire dataset
    all_files = sorted([f for f in os.listdir(data_path) if f.endswith('.pt')])
    
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(all_files)
    
    all_x = []
    all_y = []
    for file_name in all_files:
        file_path = os.path.join(data_path, file_name)
        data = torch.load(file_path)
        pressure = data['pressure'].view(1, -1, 1)
        smoke = data['smoke'].view(1, -1, 1)
        all_x.append(pressure)
        all_y.append(smoke)
    
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    x_normalizer = UnitGaussianNormalizer(all_x)
    y_normalizer = UnitGaussianNormalizer(all_y)

    # Split the file list into train and validation
    total_samples = len(all_files)
    num_train_samples = min(num_train_samples, total_samples - num_val_samples)
    num_val_samples = min(num_val_samples, total_samples - num_train_samples)

    train_files = all_files[:num_train_samples]
    val_files = all_files[-num_val_samples:]

    # Create datasets with the same normalizers
    train_dataset = CFDDataset(data_path, train_files, x_normalizer=x_normalizer, y_normalizer=y_normalizer)
    val_dataset = CFDDataset(data_path, val_files, x_normalizer=x_normalizer, y_normalizer=y_normalizer)

    return train_dataset, val_dataset