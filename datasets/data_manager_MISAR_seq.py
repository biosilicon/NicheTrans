from __future__ import print_function, absolute_import

import os

import episcanpy as epi
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from datasets.graph_utils import build_slice_graph, count_graph_nodes, to_numpy_array


def tfidf3(count_mat):
    model = TfidfTransformer(smooth_idf=False, norm="l2")
    model = model.fit(np.transpose(count_mat))
    model.idf_ -= 1
    tf_idf = np.transpose(model.transform(np.transpose(count_mat)))
    return tf_idf


class ATAC_RNA_Seq(object):
    def __init__(
        self,
        peak_threshold=0.05,
        hvg_gene=1500,
        adata_path=None,
        RNA2ATAC=False,
        knn_smoothing=False,
        graph_k=6,
        val_ratio=0.1,
        mask_seed=0,
    ):
        self.rna2atac = RNA2ATAC
        self.graph_k = graph_k
        self.val_ratio = val_ratio
        self.mask_seed = mask_seed

        e13_adata_atac = sc.read_h5ad(os.path.join(adata_path, "e13_atac.h5ad"))
        e13_adata_atac.obs["sample"] = "e13"
        e13_adata_atac.obsm["spatial"] = e13_adata_atac.obs[["array_col", "array_row"]].values

        e13_adata_rna = sc.read_h5ad(os.path.join(adata_path, "e13_rna.h5ad"))
        e13_adata_rna.obs["sample"] = "e13"
        e13_adata_rna.obsm["spatial"] = e13_adata_rna.obs[["array_col", "array_row"]].values

        e15_adata_atac = sc.read_h5ad(os.path.join(adata_path, "e15_atac.h5ad"))
        e15_adata_atac.obs["sample"] = "e15"
        e15_adata_atac.obsm["spatial"] = e15_adata_atac.obs[["array_col", "array_row"]].values

        e15_adata_rna = sc.read_h5ad(os.path.join(adata_path, "e15_rna.h5ad"))
        e15_adata_rna.obs["sample"] = "e15"
        e15_adata_rna.obsm["spatial"] = e15_adata_rna.obs[["array_col", "array_row"]].values

        e18_adata_atac = sc.read_h5ad(os.path.join(adata_path, "e18_atac.h5ad"))
        e18_adata_atac.obs["sample"] = "e18"
        e18_adata_atac.obsm["spatial"] = e18_adata_atac.obs[["array_col", "array_row"]].values

        e18_adata_rna = sc.read_h5ad(os.path.join(adata_path, "e18_rna.h5ad"))
        e18_adata_rna.obs["sample"] = "e18"
        e18_adata_rna.obsm["spatial"] = e18_adata_rna.obs[["array_col", "array_row"]].values

        atac = sc.concat([e13_adata_atac, e15_adata_atac, e18_adata_atac])
        rna = sc.concat([e13_adata_rna, e15_adata_rna, e18_adata_rna])

        epi.pp.binarize(atac)
        epi.pp.filter_features(atac, min_cells=np.ceil(peak_threshold * atac.shape[0]))

        sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=hvg_gene)
        sc.pp.log1p(rna)
        rna = rna[:, rna.var["highly_variable"]]
        sc.pp.combat(rna, key="sample")

        e13_mask = rna.obs["sample"] == "e13"
        e15_mask = rna.obs["sample"] == "e15"
        e18_mask = rna.obs["sample"] == "e18"

        if RNA2ATAC:
            self.training = [
                self._build_graph(rna[e18_mask], atac[e18_mask], split="train", mask_seed=mask_seed, knn_smoothing=knn_smoothing),
                self._build_graph(rna[e13_mask], atac[e13_mask], split="train", mask_seed=mask_seed + 1, knn_smoothing=knn_smoothing),
            ]
            self.testing = [
                self._build_graph(rna[e15_mask], atac[e15_mask], split="test", mask_seed=mask_seed + 2, knn_smoothing=knn_smoothing)
            ]
            self.target_panel = atac.var_names
            self.source_panel = rna.var_names
        else:
            self.training = [
                self._build_graph(atac[e18_mask], rna[e18_mask], split="train", mask_seed=mask_seed, knn_smoothing=knn_smoothing),
                self._build_graph(atac[e13_mask], rna[e13_mask], split="train", mask_seed=mask_seed + 1, knn_smoothing=knn_smoothing),
            ]
            self.testing = [
                self._build_graph(atac[e15_mask], rna[e15_mask], split="test", mask_seed=mask_seed + 2, knn_smoothing=knn_smoothing)
            ]
            self.target_panel = rna.var_names
            self.source_panel = atac.var_names

        self.val = self.training

        num_training_nodes = count_graph_nodes(self.training)
        num_testing_nodes = count_graph_nodes(self.testing)

        print(f"source size {len(self.source_panel)}")
        print(f"target size {len(self.target_panel)}")
        print("=> Spatial atac-rna Mouse loaded")
        print("Dataset statistics:")
        print("  -----------------------------------------")
        print("  subset   | # graphs | # nodes")
        print("  -----------------------------------------")
        print("  train    |  {:8d} | {:7d}".format(len(self.training), num_training_nodes))
        print("  test     |  {:8d} | {:7d}".format(len(self.testing), num_testing_nodes))
        print("  -----------------------------------------")

    def _smooth_targets(self, target_array):
        if self.rna2atac:
            x_tfidf = tfidf3(target_array.T).T
            svd = TruncatedSVD(n_components=30, random_state=42)
            x_svd = svd.fit_transform(x_tfidf)
            dist = pairwise_distances(x_svd, metric="euclidean")
            nearest_indices = np.argsort(dist, axis=1)[:, :50]
            return np.array([target_array[idx_list].mean(axis=0) for idx_list in nearest_indices])

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(target_array)
        pca = PCA(n_components=30)
        x_pca = pca.fit_transform(x_scaled)
        dist = pairwise_distances(x_pca, metric="euclidean")
        nearest_indices = np.argsort(dist, axis=1)[:, :50]
        return np.array([target_array[idx_list].mean(axis=0) for idx_list in nearest_indices])

    def _build_graph(self, source_adata, target_adata, split, mask_seed, knn_smoothing=False):
        # x: [N, source_dim]
        source_array = to_numpy_array(source_adata.X)
        # y: [N, target_dim]
        target_array = to_numpy_array(target_adata.X)

        if knn_smoothing:
            target_array = self._smooth_targets(target_array)

        coordinates = np.stack(
            [
                source_adata.obsm["spatial"][:, 1],
                source_adata.obsm["spatial"][:, 0],
            ],
            axis=1,
        ).astype(np.float32)
        node_ids = source_adata.obs_names.tolist()
        slice_name = str(source_adata.obs["sample"].iloc[0])

        graph = build_slice_graph(
            node_features=source_array,
            node_targets=target_array,
            coordinates=coordinates,
            split=split,
            k=self.graph_k,
            val_ratio=self.val_ratio,
            mask_seed=mask_seed,
            node_ids=[f"{slice_name}/{node_id}" for node_id in node_ids],
            sample_id=slice_name,
            slice_name=slice_name,
        )
        return graph


if __name__ == "__main__":
    dataset = ATAC_RNA_Seq(adata_path="")
