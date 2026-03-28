import torch
import time
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

def read_image(img_path):
    max_retries = 10
    for attempt in range(max_retries):
        try:
            img = Image.open(img_path).convert('RGB')
            return img
        except IOError:
            if attempt < max_retries - 1:
                print("IOError incurred when reading '{}'. Retry {}/{}".format(
                    img_path, attempt + 1, max_retries))
                time.sleep(0.1 * (2 ** attempt))
            else:
                raise IOError("Failed to read '{}' after {} attempts".format(
                    img_path, max_retries))


class SMA_loader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, rna_temp, msi_temp, rna_neighbors, msi_neighbors, sample = self.dataset[index]
        
        img = read_image(img_path=img_path)
        img = self.transform(img)

        rna_temp = torch.Tensor(rna_temp)
        msi_temp = torch.Tensor(msi_temp)
       
        rna_neighbors = torch.Tensor(rna_neighbors)
        msi_neighbors = torch.Tensor(msi_neighbors)

        return img, rna_temp, msi_temp, rna_neighbors, msi_neighbors, sample


class Lymph_node_loader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        rna_temp, protein_temp, rna_neighbors, sample = self.dataset[index]
    
        rna_temp = torch.Tensor(rna_temp)
        protein_temp = torch.Tensor(protein_temp)
       
        rna_neighbors = torch.Tensor(rna_neighbors)

        return rna_temp, protein_temp, rna_neighbors, sample
    

class Breast_cancer_loader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        rna, protein, ct, rna_neighbors, ct_neighbor, sample = self.dataset[index]
        rna_temp = torch.Tensor(rna)
        protein_temp = torch.Tensor(protein)
       
        rna_neighbors = torch.Tensor(rna_neighbors)

        return rna_temp, protein_temp, rna_neighbors, sample


class AD_Mouse_loader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        rna, protein, cell, rna_neighbor, cell_neighbor, sample = self.dataset[index]
        rna, protein, cell, rna_neighbor, cell_neighbor = torch.Tensor(rna), torch.Tensor(protein), torch.Tensor(cell), torch.Tensor(rna_neighbor), torch.Tensor(cell_neighbor)

        return rna, protein, cell, rna_neighbor, cell_neighbor, sample


class Embryonic_mouse_brain(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        source_temp, target_temp, source_neighbors, sample = self.dataset[index]
    
        source_temp = torch.Tensor(source_temp)
        target_temp = torch.Tensor(target_temp)
       
        source_neighbors = torch.Tensor(source_neighbors)

        return source_temp, target_temp, source_neighbors, sample