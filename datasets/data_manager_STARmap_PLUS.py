import os

import numpy as np
import pandas as pd
import scanpy as sc

from datasets.local_graph_utils import build_local_graph_metadata, build_spatial_neighbor_dict


# return the neighborhood nodes 
def Cal_Spatial_Net_row_col(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True, mouse=False):
    if mouse:
        coords = np.stack([adata.obs['x'], adata.obs['y']], axis=1)
        node_names = np.asarray(adata.obs.index, dtype=object)
    else:
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


class AD_Mouse(object):
    def __init__(self, AD_adata_path, Wild_type_adata_path, n_top_genes=3000, testing_control=False):

        training_slides = ['13months-disease-replicate_1_random.h5ad']
        testing_slides = ['13months-disease-replicate_2_random.h5ad']

        self.cell_type = 'ct_top'
        adata_list = []

        for slide in training_slides + testing_slides:
            path = os.path.join(AD_adata_path, slide)
            adata_temp = sc.read_h5ad(path)

            if 'highly_variable' not in adata_temp.var.columns:
                sc.pp.highly_variable_genes(adata_temp, flavor="seurat_v3", n_top_genes=n_top_genes)
            adata_list.append(adata_temp)

        self.rna_mask = adata_list[0].var['highly_variable'].values & adata_list[1].var['highly_variable'].values 
        self.cell_mask = np.array( sorted( set(adata_list[0].obs[self.cell_type].values) & set(adata_list[1].obs[self.cell_type].values)))

        val_adata = [sc.read_h5ad( os.path.join(Wild_type_adata_path, 'spatial_13months-control-replicate_1.h5ad'))]
        val_slides = ['spatial_13months-control-replicate_1.h5ad']
        
        self.training, training_tau, training_Aβ, _ = self._process_data(adata_list[0: len(training_slides)], training_slides)
        self.testing, testing_tau, testing_Aβ, graph = self._process_data(adata_list[len(training_slides): ], testing_slides)
        self.val, _, _, _ = self._process_data(val_adata, val_slides, WT=True)

        self.graph = graph

        self.rna_length = self.rna_mask.sum()
        self.target_length = 2
        self.target_panel = np.array(['tau', 'plaque'])
        self.source_panel = adata_temp.var_names[self.rna_mask]
       
        num_training_spots = len(self.training)
        num_testing_spots = len(self.testing)

        print("=> AD Mouse loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # num | ")
        print("  ------------------------------")
        print("  train    |  {:5d} spots, {} positive tao, {} positive plaque ".format(num_training_spots, training_tau, training_Aβ))
        print("  test     |  {:5d} spots, {} positive tao, {} positive plaque ".format(num_testing_spots, testing_tau, testing_Aβ))
        print("  ------------------------------")


    def _process_data(self, adata_list, slides, WT=False):

        tau, Aβ = 0, 0
        dataset = []

        for index, rna_adata in enumerate(adata_list):
            slide = slides[index]

            if WT == True:
                proteins = np.zeros((rna_adata.shape[0], 2))
            else:
                proteins = rna_adata.obs[['p-tau', 'Aβ']].values

            graph = Cal_Spatial_Net_row_col(rna_adata, k_cutoff=12, model='KNN', mouse=True)

            rna_array = rna_adata.X[:, self.rna_mask]

            indexes = rna_adata.obs.index.tolist()
            cell_array = rna_adata.obs[self.cell_type].values

            tau += proteins[:, 0].sum()
            Aβ += proteins[:, 1].sum()

            dict_rna = {}
            for i, index in enumerate(indexes):
                dict_rna[index] = rna_array[i]

            dict_cell = {}
            for i, index in enumerate(indexes):
                dict_cell[index] = (cell_array[i] == self.cell_mask) * 1

            dict_coord = {}
            coordinates = rna_adata.obs[['x', 'y']].values.astype(np.float32)
            for i, index in enumerate(indexes):
                dict_coord[index] = coordinates[i]

            for i in range(rna_adata.shape[0]):
                rna_neighbor, cell_neighbor = [], []

                cell = (cell_array[i] == self.cell_mask) * 1
                rna, protein = rna_array[i], proteins[i]
                index = indexes[i]
                center_coord = dict_coord[index]

                for j in graph[index]:
                    rna_neighbor.append(dict_rna[j])
                    cell_neighbor.append(dict_cell[j])

                rna_neighbor = np.array(rna_neighbor)
                cell_neighbor = np.array(cell_neighbor)
                neighbor_coords = np.array([dict_coord[j] for j in graph[index]], dtype=np.float32)
                hop_ids = np.ones((len(graph[index]),), dtype=np.int64)
                valid_neighbor_mask = np.ones((len(graph[index]),), dtype=bool)
                graph_meta = build_local_graph_metadata(center_coord, neighbor_coords, hop_ids, valid_neighbor_mask)

                dataset.append((rna, protein, cell, rna_neighbor, cell_neighbor, slide + '/' + index, graph_meta))

        return dataset, tau, Aβ, graph

    
if __name__ == '__main__':
    dataset = AD_Mouse()
