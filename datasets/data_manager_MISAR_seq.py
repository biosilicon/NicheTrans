from __future__ import print_function, absolute_import

import os
import numpy as np
import pandas as pd
import scanpy as sc

import sklearn.neighbors
from scipy.sparse import csr_matrix

from collections import defaultdict
import episcanpy as epi
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from datasets.cell_type_utils import resolve_global_cell_types

def tfidf3(count_mat): 
    model = TfidfTransformer(smooth_idf=False, norm="l2")
    model = model.fit(np.transpose(count_mat))
    model.idf_ -= 1
    tf_idf = np.transpose(model.transform(np.transpose(count_mat)))
    # sparse_tf_idf = csr_matrix(tf_idf)
    return tf_idf

# return the neighborhood nodes 
def Cal_Spatial_Net_row_col(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True, mouse=False):
    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    # if mouse == True:
    #     coor = pd.DataFrame(np.stack([adata.obs['x'], adata.obs['y']], axis=1))
    # else:
    #     coor = pd.DataFrame(np.stack([adata.obs['array_row'], adata.obs['array_col']], axis=1))

    coor = pd.DataFrame(np.stack([adata.obsm['spatial'][:, 1], adata.obsm['spatial'][:, 0]], axis=1))

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


class ATAC_RNA_Seq(object):
    def __init__(
        self,
        peak_threshold=0.05,
        hvg_gene=1500,
        adata_path=None,
        RNA2ATAC=False,
        knn_smoothing=False,
        cell_type_visualize=False,
        cell_type_visualization_dir=None,
        cell_type_visualization_dpi=150,
    ):
        
        self.rna2atac = RNA2ATAC
        ########################
        e13_adata_atac = sc.read_h5ad(os.path.join(adata_path, 'e13_atac.h5ad'))
        e13_adata_atac.obs['sample'] = 'e13'
        e13_adata_atac.obsm['spatial'] = e13_adata_atac.obs[['array_col', 'array_row']].values

        e13_adata_rna = sc.read_h5ad(os.path.join(adata_path, 'e13_rna.h5ad'))
        e13_adata_rna.obs['sample'] = 'e13'
        e13_adata_rna.obsm['spatial'] = e13_adata_rna.obs[['array_col', 'array_row']].values

        e15_adata_atac = sc.read_h5ad(os.path.join(adata_path, 'e15_atac.h5ad'))
        e15_adata_atac.obs['sample'] = 'e15'
        e15_adata_atac.obsm['spatial'] = e15_adata_atac.obs[['array_col', 'array_row']].values

        e15_adata_rna = sc.read_h5ad(os.path.join(adata_path, 'e15_rna.h5ad'))
        e15_adata_rna.obs['sample'] = 'e15'
        e15_adata_rna.obsm['spatial'] = e15_adata_rna.obs[['array_col', 'array_row']].values

        e18_adata_atac = sc.read_h5ad(os.path.join(adata_path, 'e18_atac.h5ad'))
        e18_adata_atac.obs['sample'] = 'e18'
        e18_adata_atac.obsm['spatial'] = e18_adata_atac.obs[['array_col', 'array_row']].values

        e18_adata_rna = sc.read_h5ad(os.path.join(adata_path, 'e18_rna.h5ad'))
        e18_adata_rna.obs['sample'] = 'e18'
        e18_adata_rna.obsm['spatial'] = e18_adata_rna.obs[['array_col', 'array_row']].values

        ########################

        atac =  sc.concat([e13_adata_atac, e15_adata_atac, e18_adata_atac])
        rna =  sc.concat([e13_adata_rna, e15_adata_rna, e18_adata_rna])

        ########################
        epi.pp.binarize(atac)
        epi.pp.filter_features(atac, min_cells=np.ceil(peak_threshold * atac.shape[0]))
        # atac.X = csr_matrix(tfidf3(atac.X.T).T).copy()

        sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=hvg_gene)
        sc.pp.log1p(rna)
        rna = rna[:, rna.var['highly_variable']]

        sc.pp.combat(rna, key='sample')
        ########################

        e13_mask = rna.obs['sample'] == 'e13'
        e15_mask = rna.obs['sample'] == 'e15'
        e18_mask = rna.obs['sample'] == 'e18'

        source_by_sample = {
            'e13': rna[e13_mask] if RNA2ATAC else atac[e13_mask],
            'e15': rna[e15_mask] if RNA2ATAC else atac[e15_mask],
            'e18': rna[e18_mask] if RNA2ATAC else atac[e18_mask],
        }
        target_by_sample = {
            'e13': atac[e13_mask] if RNA2ATAC else rna[e13_mask],
            'e15': atac[e15_mask] if RNA2ATAC else rna[e15_mask],
            'e18': atac[e18_mask] if RNA2ATAC else rna[e18_mask],
        }
        alignment_slice_names = ['e13', 'e15', 'e18']
        testing_slides = ['e15']

        cell_type_info = resolve_global_cell_types(
            adata_list=[source_by_sample[slice_name] for slice_name in alignment_slice_names],
            slice_names=alignment_slice_names,
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
        
        if RNA2ATAC == True:
            self.training = (
                self._process_data(source_by_sample['e18'], target_by_sample['e18'], 'e18')
                + self._process_data(source_by_sample['e13'], target_by_sample['e13'], 'e13')
            )
            self.testing = self._process_data(
                source_by_sample['e15'],
                target_by_sample['e15'],
                'e15',
                knn_smoothing=knn_smoothing,
            )

            self.target_panel = atac.var_names
            self.source_panel = rna.var_names
        else:
            self.training = (
                self._process_data(source_by_sample['e18'], target_by_sample['e18'], 'e18')
                + self._process_data(source_by_sample['e13'], target_by_sample['e13'], 'e13')
            )
            self.testing = self._process_data(
                source_by_sample['e15'],
                target_by_sample['e15'],
                'e15',
                knn_smoothing=knn_smoothing,
            )

            self.target_panel = rna.var_names
            self.source_panel = atac.var_names

        num_training_spots = len(self.training)
        num_testing_spots = len(self.testing)

        
        print(f'source size {len(self.source_panel)}')
        print(f'target size {len(self.target_panel)}')

        print("=> Spatial atac-rna Mouse loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # num | ")
        print("  ------------------------------")
        print("  train    |  {:5d} spots,".format(num_training_spots))
        print("  test     |  {:5d} spots,".format(num_testing_spots))
        print("  ------------------------------")
        print(f'  Global cell-type source: {self.cell_type_source}')
        print(f'  Total global cell types used for embedding: {self.n_spot_types}')
        if self.cell_type_visualization_paths:
            print(f'  Cell-type visualization slices: {sorted(self.cell_type_visualization_paths)}')

    def _process_data(self, source_adata, target_adata, slide_name, knn_smoothing=False):
        dataset = []
        
        ######
        source_array = source_adata.X.toarray()
        target_array = target_adata.X.toarray()

        if knn_smoothing and self.rna2atac:
            X = target_array.copy()
            X_tfidf = tfidf3(X.T).T
            N_SVD_COMPONENTS = 30
            svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=42)
            X_svd = svd.fit_transform(X_tfidf)
            # breakpoint()          
            dist = pairwise_distances(X_svd, metric="euclidean")
            k = 50
            nearest_indices = np.argsort(dist, axis=1)[:, :k]
            target_array = np.array([X[idx_list].mean(axis=0) for idx_list in nearest_indices])

        elif knn_smoothing and not self.rna2atac:
            scaler = StandardScaler()
            X = target_array.copy()
            X_scaled = scaler.fit_transform(X)

            k = 30
            pca = PCA(n_components=k)
            X_pca = pca.fit_transform(X_scaled)
            dist = pairwise_distances(X_pca, metric="euclidean")
            k = 50
            nearest_indices = np.argsort(dist, axis=1)[:, :k]
            target_array = np.array([X[idx_list].mean(axis=0) for idx_list in nearest_indices])

        indexes = source_adata.obs_names
        ######
        graph = Cal_Spatial_Net_row_col(source_adata, k_cutoff=8, model='KNN', mouse=True)

        ######
        dict_source = {}
        dict_spot_type = {}
        global_spot_type_ids = self.global_cell_type_ids_by_slice[slide_name]
        for i, index in enumerate(indexes):
            dict_source[index] = source_array[i]
            dict_spot_type[index] = int(global_spot_type_ids[i])
        #######
        
        for i in range(source_adata.shape[0]):
            source_neighbor = []

            index = indexes[i]
            source, target = source_array[i], target_array[i]
            spot_type_id = dict_spot_type[index]
            neighbor_spot_type_ids = []

            for j in graph[index]:
                source_neighbor.append(dict_source[j])
                neighbor_spot_type_ids.append(dict_spot_type[j])

            source_neighbor = np.array(source_neighbor)
            spot_type_ids = np.asarray([spot_type_id] + neighbor_spot_type_ids, dtype=np.int64)

            dataset.append((source, target, source_neighbor, spot_type_ids, index))

        return dataset
    
if __name__ == '__main__':
    dataset = ATAC_RNA_Seq()
