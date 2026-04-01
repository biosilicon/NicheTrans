import numpy as np
import pandas as pd
import scanpy as sc

from datasets.local_graph_utils import build_local_graph_metadata, build_spatial_neighbor_dict

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


class Breast_cancer(object):
    def __init__(self, adata_path, coordinate_path, ct_path):

        adata = sc.read_h5ad(adata_path)
        coordinates = pd.read_csv(coordinate_path, compression='gzip')
        ct = pd.read_excel(ct_path, sheet_name='Xenium R1 Fig1-5 (supervised)') 

        #######
        adata.obs['cell_CD20_mean'] = np.log(adata.obs['cell_CD20_mean'].values + 1)
        adata.obs['cell_HER2_mean'] = np.log(adata.obs['cell_HER2_mean'].values + 1)

        sc.pp.normalize_total(adata, target_sum=1e3)
        sc.pp.log1p(adata)
        #######

        adata.obs['x'], adata.obs['y'] = coordinates['x_centroid'].values, coordinates['y_centroid'].values
        adata.obs['ct'] = ct['Cluster'].values

        self.cell_mask = np.array(sorted(set(ct['Cluster'].values.tolist())))

        centra = adata.obs['x'].values.max()//2
        training_adata = adata[ adata.obs['x'].values <= centra ]
        testing_adata = adata[ adata.obs['x'].values >= centra ]

        ########
        self.training = self._process_data(training_adata)
        self.testing = self._process_data(testing_adata)

        self.rna_length, self.protein_length = 313, 2
        self.target_panel = np.array(['CD20', 'HER2'])
       
        num_training_spots = len(self.training)
        num_testing_spots = len(self.testing) 

        print("=> AD Mouse loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # num | ")
        print("  ------------------------------")
        print("  train    |  {:5d} spots, {} positive CD20, {} positive HER2 ".format(num_training_spots, (training_adata.obs['cell_CD20_mean'].values > 0).sum(), (training_adata.obs['cell_HER2_mean'].values > 0).sum()))
        print("  test     |  {:5d} spots, {} positive CD20, {} positive HER2 ".format(num_testing_spots, (testing_adata.obs['cell_CD20_mean'].values > 0).sum(), (testing_adata.obs['cell_HER2_mean'].values > 0).sum()))
        print("  ------------------------------")


    def _process_data(self, rna_adata):
        dataset = []
        
        ######
        rna_array = rna_adata.X.toarray()
        ct_array = rna_adata.obs['ct'].values
        indexes = rna_adata.obs.index.tolist()
        proteins = rna_adata.obs[['cell_CD20_mean', 'cell_HER2_mean']].values.astype(np.float32)
        ######
        graph = Cal_Spatial_Net_row_col(rna_adata, k_cutoff=12, model='KNN', mouse=True)

        ######
        dict_rna, dict_ct = {}, {}
        dict_coord = {}
        coordinates = rna_adata.obs[['x', 'y']].values.astype(np.float32)
        for i, index in enumerate(indexes):
            dict_rna[index] = rna_array[i]
            dict_ct[index] = (ct_array[i] == self.cell_mask) * 1
            dict_coord[index] = coordinates[i]
        #######
        
        for i in range(rna_adata.shape[0]):
            rna_neighbor, ct_neighbor = [], []

            index = indexes[i]
            rna, protein, ct = rna_array[i], proteins[i], dict_ct[index]
            center_coord = dict_coord[index]

            for j in graph[index]:
                rna_neighbor.append(dict_rna[j])
                ct_neighbor.append(dict_ct[j])

            rna_neighbor = np.array(rna_neighbor)
            ct_neighbor = np.array(ct_neighbor)
            neighbor_coords = np.array([dict_coord[j] for j in graph[index]], dtype=np.float32)
            hop_ids = np.ones((len(graph[index]),), dtype=np.int64)
            valid_neighbor_mask = np.ones((len(graph[index]),), dtype=bool)
            graph_meta = build_local_graph_metadata(center_coord, neighbor_coords, hop_ids, valid_neighbor_mask)

            dataset.append((rna, protein, ct, rna_neighbor, ct_neighbor, index, graph_meta))

        return dataset

    
if __name__ == '__main__':
    dataset = Breast_cancer()
