import sklearn.neighbors

import numpy as np
import pandas as pd
import scanpy as sc

from collections import defaultdict
from datasets.cell_type_utils import resolve_global_cell_types

def Cal_Spatial_Net_row_col(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True, mouse=False):
    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    if mouse == True:
        coor = pd.DataFrame(np.stack([adata.obs['x'], adata.obs['y']], axis=1))
    else:
        coor = pd.DataFrame(np.stack([adata.obs['array_row'], adata.obs['array_col']], axis=1))

    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            # breakpoint()
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

    temp_dic = defaultdict(list)

    if mouse == True:
        for i in range(Spatial_Net.shape[0]):
            center = Spatial_Net.iloc[i, 0]
            side = Spatial_Net.iloc[i, 1]
            temp_dic[center].append(side)
    else:
        for i in range(Spatial_Net.shape[0]):
            center = Spatial_Net.iloc[i, 0]
            side = Spatial_Net.iloc[i, 1]
            center_name = str(adata.obs['array_row'][center]) + '_' + str(adata.obs['array_col'][center])
            side_name = str(adata.obs['array_row'][side]) + '_' + str(adata.obs['array_col'][side])
            temp_dic[center_name].append(side_name)

    return temp_dic


class Breast_cancer(object):
    def __init__(
        self,
        adata_path,
        coordinate_path,
        ct_path,
        cell_type_visualize=False,
        cell_type_visualization_dir=None,
        cell_type_visualization_dpi=150,
    ):

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
        # Map cell-type string → integer ID, mirroring STARmap_PLUS convention.
        self.cell_type_to_id = {ct_name: i for i, ct_name in enumerate(self.cell_mask)}
        self.n_spot_types = len(self.cell_mask)

        centra = adata.obs['x'].values.max()//2
        training_adata = adata[ adata.obs['x'].values <= centra ]
        testing_adata = adata[ adata.obs['x'].values >= centra ]
        training_slides = ['training_slice']
        testing_slides = ['testing_slice']

        cell_type_info = resolve_global_cell_types(
            adata_list=[training_adata, testing_adata],
            slice_names=training_slides + testing_slides,
            annotation_key='ct',
            candidate_annotation_keys=['ct'],
            testing_slides=testing_slides,
            visualize=cell_type_visualize,
            visualization_dir=cell_type_visualization_dir,
            visualization_dpi=cell_type_visualization_dpi,
            verbose=True,
        )
        self.cell_type_source = cell_type_info['source']
        self.cell_type_annotation_key = cell_type_info['annotation_key']
        self.cell_mask = cell_type_info['cell_type_names']
        self.cell_type_to_id = cell_type_info['name_to_id']
        self.global_cell_type_id_to_name = cell_type_info['id_to_name']
        self.global_cell_type_ids_by_slice = cell_type_info['global_cell_type_ids_by_slice']
        self.local_cell_type_to_global_id = cell_type_info['slice_local_to_global']
        self.cell_type_alignment_info = cell_type_info['alignment_info']
        self.cell_type_visualization_paths = cell_type_info.get('visualization_paths', {})
        self.n_spot_types = cell_type_info['n_cell_types']
        self.n_cell_types = self.n_spot_types
        self.global_mapping = self.local_cell_type_to_global_id

        ########
        self.training = self._process_data(training_adata, training_slides[0])
        self.testing = self._process_data(testing_adata, testing_slides[0])

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
        print(f'  Global cell-type source: {self.cell_type_source}')
        print(f'  Total global cell types used for embedding: {self.n_spot_types}')
        if self.cell_type_visualization_paths:
            print(f'  Cell-type visualization slices: {sorted(self.cell_type_visualization_paths)}')


    def _process_data(self, rna_adata, slide_name):
        dataset = []
        
        ######
        rna_array = rna_adata.X.toarray()
        indexes = rna_adata.obs.index.tolist()
        proteins = rna_adata.obs[['cell_CD20_mean', 'cell_HER2_mean']].values.astype(np.float32)
        global_cell_type_ids = self.global_cell_type_ids_by_slice[slide_name]
        ######
        graph = Cal_Spatial_Net_row_col(rna_adata, k_cutoff=12, model='KNN', mouse=True)

        ######
        dict_rna, dict_ct, dict_spot_type = {}, {}, {}
        for i, index in enumerate(indexes):
            dict_rna[index] = rna_array[i]
            cell_onehot = np.zeros(self.n_cell_types, dtype=np.float32)
            cell_onehot[int(global_cell_type_ids[i])] = 1.0
            dict_ct[index] = cell_onehot
            dict_spot_type[index] = int(global_cell_type_ids[i])
        #######
        
        for i in range(rna_adata.shape[0]):
            rna_neighbor, ct_neighbor = [], []

            index = indexes[i]
            rna, protein, ct = rna_array[i], proteins[i], dict_ct[index]
            spot_type_id = dict_spot_type[index]
            neighbor_spot_type_ids = []

            for j in graph[index]:
                rna_neighbor.append(dict_rna[j])
                ct_neighbor.append(dict_ct[j])
                neighbor_spot_type_ids.append(dict_spot_type[j])

            rna_neighbor = np.array(rna_neighbor)
            ct_neighbor = np.array(ct_neighbor)
            spot_type_ids = np.asarray([spot_type_id] + neighbor_spot_type_ids, dtype=np.int64)

            dataset.append((rna, protein, ct, rna_neighbor, ct_neighbor, spot_type_ids, index))

        return dataset

    
if __name__ == '__main__':
    dataset = Breast_cancer()
