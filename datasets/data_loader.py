import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


def build_graph_meta_index(dataset, sample_index=-2, graph_meta_index=-1):
    return {record[sample_index]: record[graph_meta_index] for record in dataset}


def stack_graph_meta(graph_meta_index, samples):
    if len(samples) == 0:
        return None

    keys = list(graph_meta_index[samples[0]].keys())
    graph_meta = {}
    for key in keys:
        values = [graph_meta_index[sample][key] for sample in samples]
        graph_meta[key] = torch.as_tensor(np.stack(values))
    return graph_meta

def read_image(img_path):
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class SMA_loader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.graph_meta_index = build_graph_meta_index(dataset)

    def __len__(self):
        return len(self.dataset)

    def get_graph_meta(self, samples):
        return stack_graph_meta(self.graph_meta_index, samples)

    def __getitem__(self, index):
        img_path, rna_temp, msi_temp, rna_neighbors, msi_neighbors, sample, _ = self.dataset[index]
        
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
        self.graph_meta_index = build_graph_meta_index(dataset)

    def __len__(self):
        return len(self.dataset)

    def get_graph_meta(self, samples):
        return stack_graph_meta(self.graph_meta_index, samples)

    def __getitem__(self, index):
        rna_temp, protein_temp, rna_neighbors, sample, _ = self.dataset[index]
    
        rna_temp = torch.Tensor(rna_temp)
        protein_temp = torch.Tensor(protein_temp)
       
        rna_neighbors = torch.Tensor(rna_neighbors)

        return rna_temp, protein_temp, rna_neighbors, sample
    

class Breast_cancer_loader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.graph_meta_index = build_graph_meta_index(dataset)

    def __len__(self):
        return len(self.dataset)

    def get_graph_meta(self, samples):
        return stack_graph_meta(self.graph_meta_index, samples)

    def __getitem__(self, index):
        rna, protein, ct, rna_neighbors, ct_neighbor, sample, _ = self.dataset[index]
        rna_temp = torch.Tensor(rna)
        protein_temp = torch.Tensor(protein)
       
        rna_neighbors = torch.Tensor(rna_neighbors)

        return rna_temp, protein_temp, rna_neighbors, sample


class AD_Mouse_loader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.graph_meta_index = build_graph_meta_index(dataset)

    def __len__(self):
        return len(self.dataset)

    def get_graph_meta(self, samples):
        return stack_graph_meta(self.graph_meta_index, samples)

    def __getitem__(self, index):
        
        rna, protein, cell, rna_neighbor, cell_neighbor, sample, _ = self.dataset[index]
        rna, protein, cell, rna_neighbor, cell_neighbor = torch.Tensor(rna), torch.Tensor(protein), torch.Tensor(cell), torch.Tensor(rna_neighbor), torch.Tensor(cell_neighbor)

        return rna, protein, cell, rna_neighbor, cell_neighbor, sample


class Embryonic_mouse_brain(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.graph_meta_index = build_graph_meta_index(dataset)

    def __len__(self):
        return len(self.dataset)

    def get_graph_meta(self, samples):
        return stack_graph_meta(self.graph_meta_index, samples)

    def __getitem__(self, index):
        source_temp, target_temp, source_neighbors, sample, _ = self.dataset[index]
    
        source_temp = torch.Tensor(source_temp)
        target_temp = torch.Tensor(target_temp)
       
        source_neighbors = torch.Tensor(source_neighbors)

        return source_temp, target_temp, source_neighbors, sample
