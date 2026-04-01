from __future__ import print_function, absolute_import

import os
import numpy as np
import pandas as pd
import scanpy as sc

from scipy.sparse import csr_matrix

import episcanpy as epi
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from datasets.local_graph_utils import build_local_graph_metadata, build_spatial_neighbor_dict

def tfidf3(count_mat): 
    model = TfidfTransformer(smooth_idf=False, norm="l2")
    model = model.fit(np.transpose(count_mat))
    model.idf_ -= 1
    tf_idf = np.transpose(model.transform(np.transpose(count_mat)))
    # sparse_tf_idf = csr_matrix(tf_idf)
    return tf_idf

# return the neighborhood nodes 
def Cal_Spatial_Net_row_col(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True, mouse=False):
    coords = np.stack([adata.obsm['spatial'][:, 1], adata.obsm['spatial'][:, 0]], axis=1)
    if mouse:
        node_names = np.asarray(adata.obs.index, dtype=object)
    else:
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


class ATAC_RNA_Seq(object):
    def __init__(self, peak_threshold=0.05, hvg_gene=1500, adata_path=None, RNA2ATAC=False, knn_smoothing=False):
        
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
        
        if RNA2ATAC == True:
            self.training = self._process_data(rna[e18_mask], atac[e18_mask]) + self._process_data(rna[e13_mask], atac[e13_mask])
            self.testing = self._process_data(rna[e15_mask], atac[e15_mask], knn_smoothing=knn_smoothing) 

            self.target_panel = atac.var_names
            self.source_panel = rna.var_names
        else:
            self.training = self._process_data(atac[e18_mask], rna[e18_mask]) + self._process_data(atac[e13_mask], rna[e13_mask])
            self.testing = self._process_data(atac[e15_mask], rna[e15_mask], knn_smoothing=knn_smoothing) 

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


    def _process_data(self, source_adata, target_adata, knn_smoothing=False):
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
        dict_coord = {}
        coordinates = source_adata.obsm['spatial'].astype(np.float32)
        for i, index in enumerate(indexes):
            dict_source[index] = source_array[i]
            dict_coord[index] = coordinates[i]
        #######
        
        for i in range(source_adata.shape[0]):
            source_neighbor = []

            index = indexes[i]
            source, target = source_array[i], target_array[i]
            center_coord = dict_coord[index]

            for j in graph[index]:
                source_neighbor.append(dict_source[j])

            source_neighbor = np.array(source_neighbor)
            neighbor_coords = np.array([dict_coord[j] for j in graph[index]], dtype=np.float32)
            hop_ids = np.ones((len(graph[index]),), dtype=np.int64)
            valid_neighbor_mask = np.ones((len(graph[index]),), dtype=bool)
            graph_meta = build_local_graph_metadata(center_coord, neighbor_coords, hop_ids, valid_neighbor_mask)

            dataset.append((source, target, source_neighbor, index, graph_meta))

        return dataset
    
if __name__ == '__main__':
    dataset = ATAC_RNA_Seq()
