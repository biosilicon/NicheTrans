import os

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.cluster import KMeans

from scipy.sparse import issparse

from datasets.local_graph_utils import build_local_graph_metadata, build_spatial_neighbor_dict


# return the neighborhood nodes 
def Cal_Spatial_Net_row_col(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    coords = np.stack([adata.obs['array_row'], adata.obs['array_col']], axis=1)
    node_names = np.array(
        [
            f'{row}_{col}'
            for row, col in zip(adata.obs['array_row'].tolist(), adata.obs['array_col'].tolist())
        ],
        dtype=object,
    )
    return build_spatial_neighbor_dict(
        coords=coords,
        index_labels=adata.obs.index,
        node_names=node_names,
        rad_cutoff=rad_cutoff,
        k_cutoff=k_cutoff,
        model=model,
        verbose=verbose,
        adata=adata,
    )


class Lymph_node(object):
    def __init__(self, adata_path, n_top_genes=3000):
        
        rna_pathes = [adata_path + 'slice1/s1_adata_rna.h5ad']
        protein_pathes = [adata_path + 'slice1/s1_adata_adt.h5ad']
        
        #####
        rna_adata_list, protein_adata_list = [], []

        for i in range(len(rna_pathes)):
            rna_path, protein_path = rna_pathes[i], protein_pathes[i]

            adata_rna_training = sc.read_h5ad(rna_path)
            sc.pp.highly_variable_genes(adata_rna_training, flavor="seurat_v3", n_top_genes=n_top_genes)
            sc.pp.normalize_total(adata_rna_training, target_sum=1e4)
            sc.pp.log1p(adata_rna_training)

            adata_rna_training.obs['array_row'] = adata_rna_training.obsm['spatial'][:, 0]
            adata_rna_training.obs['array_col'] = adata_rna_training.obsm['spatial'][:, 1]

            adata_protein_training = sc.read_h5ad(protein_path)
            sc.pp.log1p(adata_protein_training)
            
            adata_protein_training.obs['array_row'] = adata_protein_training.obsm['spatial'][:, 0]
            adata_protein_training.obs['array_col'] = adata_protein_training.obsm['spatial'][:, 1]

            rna_adata_list.append(adata_rna_training.copy())
            protein_adata_list.append(adata_protein_training.copy())
        ######

        rna_path = adata_path + 'slice2/s2_adata_rna.h5ad'
        protein_path = adata_path + 'slice2/s2_adata_adt.h5ad'

        #####
        adata_rna_testing = sc.read_h5ad(rna_path)
        sc.pp.highly_variable_genes(adata_rna_testing, flavor="seurat_v3", n_top_genes=n_top_genes)
        sc.pp.normalize_total(adata_rna_testing, target_sum=1e4)
        sc.pp.log1p(adata_rna_testing)

        adata_rna_testing.obs['array_row'] = -adata_rna_testing.obsm['spatial'][:, 0]
        adata_rna_testing.obs['array_col'] = -adata_rna_testing.obsm['spatial'][:, 1]

        adata_protein_testing = sc.read_h5ad(protein_path)
        sc.pp.log1p(adata_protein_testing)

        adata_protein_testing.obs['array_row'] = adata_protein_testing.obsm['spatial'][:, 0]
        adata_protein_testing.obs['array_col'] = adata_protein_testing.obsm['spatial'][:, 1]

        ###
        hvg = rna_adata_list[0].var['highly_variable'] & adata_rna_testing.var['highly_variable']
        
        rna_adata_list[0] = rna_adata_list[0][:, hvg]
        adata_rna_testing = adata_rna_testing[:, hvg]

        temp = np.concatenate( [protein_adata_list[0].X.toarray(), adata_protein_testing.X.toarray()], axis=0)
        mean, std = temp.mean(axis=0), temp.std(axis=0)
        self.mean, self.std = mean, std

        protein_adata_list[0].X = (protein_adata_list[0].X.toarray() - mean[None, ]) / std[None, ]
        adata_protein_testing.X = (adata_protein_testing.X.toarray() - mean[None, ]) / std[None, ]


        self.training = self._process_data(rna_adata_list, protein_adata_list)
        self.testing = self._process_data([adata_rna_testing], [adata_protein_testing])

        self.rna_length = adata_rna_testing.shape[1]
        self.protein_length = adata_protein_testing.shape[1]
        self.target_panel = adata_protein_testing.var.index.tolist()

        num_training_spots, num_testing_spots = len(self.training), len(self.testing)

        print("=> Human lymph node loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # num | ")
        print("  ------------------------------")
        print("  train    |  After filting {:5d} spots".format(num_training_spots))
        print("  test     |  After filting {:5d} spots".format(num_testing_spots))
        print("  ------------------------------")


    def _dictionary_data(self, adata):
        dictionary = {}

        if issparse(adata.X):
            array = adata.X.toarray()
        else:
            array = adata.X

        array_row, array_col = adata.obs['array_row'].values, adata.obs['array_col'].values
        ######
        for i in range(adata.shape[0]):
            dictionary[str(int(array_row[i])) + '_' +  str(int(array_col[i])) ] = array[i]
        return dictionary

    
    def _process_data(self, rna_adata_list, protein_adata_list):

        dataset = []

        for index in range(len(rna_adata_list)):
            rna_adata = rna_adata_list[index]
            protein_adata = protein_adata_list[index]

            rna_dic = self._dictionary_data(rna_adata)
            protein_dic = self._dictionary_data(protein_adata)

            # construct the graph
            graph_1 = Cal_Spatial_Net_row_col(rna_adata,  rad_cutoff=2**(1/2), model='Radius')
            graph_2 = Cal_Spatial_Net_row_col(rna_adata,  rad_cutoff=2, model='Radius')

            rna_keys = rna_dic.keys()

            for key in rna_keys:
                rna_temp, protein_temp = rna_dic[key], protein_dic[key]
                center_coord = np.array([float(item) for item in key.split('_')], dtype=np.float32)

                rna_neighbors = []
                neighbor_coords, hop_ids, valid_neighbor_mask = [], [], []
                neighbors_1, neighbors_2 = graph_1[key], graph_2[key]
                neighbors_2 = [item for item in neighbors_2 if item not in neighbors_1]

                # connect to the first round 
                for j in neighbors_1:
                    if j not in rna_keys:
                        rna_neighbors.append(np.zeros_like(rna_temp))
                        neighbor_coords.append(center_coord.copy())
                        valid_neighbor_mask.append(False)
                    else:
                        rna_neighbors.append(rna_dic[j])
                        neighbor_coords.append(np.array([float(item) for item in j.split('_')], dtype=np.float32))
                        valid_neighbor_mask.append(True)
                    hop_ids.append(1)

                if len(neighbors_1) != 4:
                    for _ in range(4-len(neighbors_1)):
                        rna_neighbors.append(np.zeros_like(rna_temp))
                        neighbor_coords.append(center_coord.copy())
                        hop_ids.append(1)
                        valid_neighbor_mask.append(False)

                # connect to the second round
                for j in neighbors_2:
                    if j not in rna_keys:
                        rna_neighbors.append(np.zeros_like(rna_temp))
                        neighbor_coords.append(center_coord.copy())
                        valid_neighbor_mask.append(False)
                    else:
                        rna_neighbors.append(rna_dic[j])
                        neighbor_coords.append(np.array([float(item) for item in j.split('_')], dtype=np.float32))
                        valid_neighbor_mask.append(True)
                    hop_ids.append(2)

                if len(neighbors_2) != 4:
                    for _ in range(4-len(neighbors_2)):
                        rna_neighbors.append(np.zeros_like(rna_temp))
                        neighbor_coords.append(center_coord.copy())
                        hop_ids.append(2)
                        valid_neighbor_mask.append(False)

                rna_neighbors = np.stack(rna_neighbors)
                graph_meta = build_local_graph_metadata(center_coord, neighbor_coords, hop_ids, valid_neighbor_mask)
                
                dataset.append((rna_temp, protein_temp, rna_neighbors, key, graph_meta))

        return dataset


if __name__ == '__main__':
    dataset = Lymph_node()
