import os

import numpy as np
import scanpy as sc

from datasets.graph_utils import build_slice_graph, count_graph_nodes, to_numpy_array


class Lymph_node(object):
    def __init__(self, adata_path, n_top_genes=3000, graph_k=6, val_ratio=0.1, mask_seed=0):
        self.graph_k = graph_k
        self.val_ratio = val_ratio
        self.mask_seed = mask_seed

        training_rna_paths = [os.path.join(adata_path, "slice1", "s1_adata_rna.h5ad")]
        training_protein_paths = [os.path.join(adata_path, "slice1", "s1_adata_adt.h5ad")]

        rna_adata_list, protein_adata_list = [], []
        for rna_path, protein_path in zip(training_rna_paths, training_protein_paths):
            adata_rna_training = sc.read_h5ad(rna_path)
            sc.pp.highly_variable_genes(adata_rna_training, flavor="seurat_v3", n_top_genes=n_top_genes)
            sc.pp.normalize_total(adata_rna_training, target_sum=1e4)
            sc.pp.log1p(adata_rna_training)
            adata_rna_training.obs["array_row"] = adata_rna_training.obsm["spatial"][:, 0]
            adata_rna_training.obs["array_col"] = adata_rna_training.obsm["spatial"][:, 1]

            adata_protein_training = sc.read_h5ad(protein_path)
            sc.pp.log1p(adata_protein_training)
            adata_protein_training.obs["array_row"] = adata_protein_training.obsm["spatial"][:, 0]
            adata_protein_training.obs["array_col"] = adata_protein_training.obsm["spatial"][:, 1]

            rna_adata_list.append(adata_rna_training.copy())
            protein_adata_list.append(adata_protein_training.copy())

        rna_path = os.path.join(adata_path, "slice2", "s2_adata_rna.h5ad")
        protein_path = os.path.join(adata_path, "slice2", "s2_adata_adt.h5ad")

        adata_rna_testing = sc.read_h5ad(rna_path)
        sc.pp.highly_variable_genes(adata_rna_testing, flavor="seurat_v3", n_top_genes=n_top_genes)
        sc.pp.normalize_total(adata_rna_testing, target_sum=1e4)
        sc.pp.log1p(adata_rna_testing)
        adata_rna_testing.obs["array_row"] = -adata_rna_testing.obsm["spatial"][:, 0]
        adata_rna_testing.obs["array_col"] = -adata_rna_testing.obsm["spatial"][:, 1]

        adata_protein_testing = sc.read_h5ad(protein_path)
        sc.pp.log1p(adata_protein_testing)
        adata_protein_testing.obs["array_row"] = adata_protein_testing.obsm["spatial"][:, 0]
        adata_protein_testing.obs["array_col"] = adata_protein_testing.obsm["spatial"][:, 1]

        hvg = rna_adata_list[0].var["highly_variable"] & adata_rna_testing.var["highly_variable"]
        rna_adata_list[0] = rna_adata_list[0][:, hvg]
        adata_rna_testing = adata_rna_testing[:, hvg]

        temp = np.concatenate(
            [
                to_numpy_array(protein_adata_list[0].X),
                to_numpy_array(adata_protein_testing.X),
            ],
            axis=0,
        )
        self.mean = temp.mean(axis=0)
        self.std = temp.std(axis=0)
        self.std[self.std == 0] = 1

        protein_adata_list[0].X = (to_numpy_array(protein_adata_list[0].X) - self.mean[None, :]) / self.std[None, :]
        adata_protein_testing.X = (to_numpy_array(adata_protein_testing.X) - self.mean[None, :]) / self.std[None, :]

        self.training = [
            self._build_graph(
                rna_adata=rna_adata_list[0],
                protein_adata=protein_adata_list[0],
                split="train",
                slice_name="slice1",
                mask_seed=mask_seed,
            )
        ]
        self.testing = [
            self._build_graph(
                rna_adata=adata_rna_testing,
                protein_adata=adata_protein_testing,
                split="test",
                slice_name="slice2",
                mask_seed=mask_seed + 1,
            )
        ]
        self.val = self.training

        self.rna_length = int(adata_rna_testing.shape[1])
        self.protein_length = int(adata_protein_testing.shape[1])
        self.source_panel = adata_rna_testing.var_names.tolist()
        self.target_panel = adata_protein_testing.var.index.tolist()

        num_training_nodes = count_graph_nodes(self.training)
        num_testing_nodes = count_graph_nodes(self.testing)

        print("=> Human lymph node loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # graphs | # nodes")
        print("  ------------------------------")
        print("  train    |  {:8d} | {:7d}".format(len(self.training), num_training_nodes))
        print("  test     |  {:8d} | {:7d}".format(len(self.testing), num_testing_nodes))
        print("  ------------------------------")

    def _coordinate_keys(self, adata):
        rows = adata.obs["array_row"].values
        cols = adata.obs["array_col"].values
        return np.array([f"{int(row)}_{int(col)}" for row, col in zip(rows, cols)], dtype=object)

    def _build_graph(self, rna_adata, protein_adata, split, slice_name, mask_seed):
        rna_keys = self._coordinate_keys(rna_adata)
        protein_keys = self._coordinate_keys(protein_adata)

        rna_key_to_index = {key: idx for idx, key in enumerate(rna_keys)}
        protein_key_to_index = {key: idx for idx, key in enumerate(protein_keys)}

        common_keys = [key for key in rna_keys if key in protein_key_to_index]
        if not common_keys:
            raise ValueError(f"No aligned nodes found for slice '{slice_name}'.")

        rna_array = to_numpy_array(rna_adata.X)
        protein_array = to_numpy_array(protein_adata.X)

        rna_indices = [rna_key_to_index[key] for key in common_keys]
        protein_indices = [protein_key_to_index[key] for key in common_keys]
        coordinates = rna_adata.obs[["array_row", "array_col"]].values[rna_indices]

        graph = build_slice_graph(
            node_features=rna_array[rna_indices],
            node_targets=protein_array[protein_indices],
            coordinates=coordinates,
            split=split,
            k=self.graph_k,
            val_ratio=self.val_ratio,
            mask_seed=mask_seed,
            node_ids=common_keys,
            sample_id=slice_name,
            slice_name=slice_name,
        )
        return graph


if __name__ == "__main__":
    dataset = Lymph_node("")
