import os

import numpy as np
import scanpy as sc

from datasets.graph_utils import build_slice_graph, count_graph_nodes, to_numpy_array


class SMA(object):
    def __init__(
        self,
        path_img,
        rna_path,
        msi_path,
        n_top_genes=3000,
        n_top_targets=50,
        graph_k=6,
        val_ratio=0.1,
        mask_seed=0,
    ):
        training_slides = ["V11L12-109_B1", "V11L12-109_C1"]
        testing_slides = ["V11L12-109_A1"]

        self.path_img = path_img
        self.graph_k = graph_k
        self.val_ratio = val_ratio
        self.mask_seed = mask_seed

        rna_adata_list, msi_adata_list = [], []
        rna_highly_variable_list, msi_highly_variable_list = [], []

        for slide in training_slides + testing_slides:
            adata_rna = sc.read_visium(os.path.join(rna_path, slide))
            adata_rna.var_names_make_unique()
            sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=n_top_genes)
            sc.pp.normalize_total(adata_rna, target_sum=1e4)
            sc.pp.log1p(adata_rna)

            rna_adata_list.append(adata_rna)
            rna_highly_variable_list.append(adata_rna.var["highly_variable"].values)

            adata_msi = sc.read_h5ad(os.path.join(msi_path, f"metabolite_{slide}.h5ad"))
            adata_msi.var_names_make_unique()
            sc.pp.highly_variable_genes(adata_msi, flavor="seurat_v3", n_top_genes=n_top_targets)
            sc.pp.log1p(adata_msi)

            msi_adata_list.append(adata_msi)
            msi_highly_variable_list.append(adata_msi.var["highly_variable"].values)

        temp = np.concatenate([to_numpy_array(adata.X) for adata in rna_adata_list], axis=0)
        self.rna_mean = temp.mean(axis=0, keepdims=True)
        self.rna_std = temp.std(axis=0, keepdims=True)
        self.rna_std[self.rna_std == 0] = 1

        self.rna_mask = rna_highly_variable_list[0] & rna_highly_variable_list[1] & rna_highly_variable_list[2]
        self.msi_mask = msi_highly_variable_list[0] & msi_highly_variable_list[1] & msi_highly_variable_list[2]

        self.training = [
            self._build_graph(
                rna_adata=rna_adata_list[index],
                msi_adata=msi_adata_list[index],
                split="train",
                slide_name=training_slides[index],
                mask_seed=mask_seed + index,
            )
            for index in range(len(training_slides))
        ]
        self.testing = [
            self._build_graph(
                rna_adata=rna_adata_list[len(training_slides) + index],
                msi_adata=msi_adata_list[len(training_slides) + index],
                split="test",
                slide_name=testing_slides[index],
                mask_seed=mask_seed + len(training_slides) + index,
            )
            for index in range(len(testing_slides))
        ]
        self.val = self.training

        self.rna_length = int(self.rna_mask.sum())
        self.msi_length = int(self.msi_mask.sum())
        self.source_panel = rna_adata_list[0].var_names[self.rna_mask]
        self.target_panel = msi_adata_list[0].var["metabolism"].values[self.msi_mask].tolist()

        num_training_nodes = count_graph_nodes(self.training)
        num_testing_nodes = count_graph_nodes(self.testing)
        num_training_slides = len(training_slides)
        num_testing_slides = len(testing_slides)

        ori_num_training_spots = int(rna_adata_list[0].shape[0] + rna_adata_list[1].shape[0])
        ori_num_testing_spots = int(rna_adata_list[2].shape[0])

        print("=> SMA loaded")
        print("Dataset statistics:")
        print("  -----------------------------------------")
        print("  subset   | # graphs | # nodes")
        print("  -----------------------------------------")
        print("  train    |  {:8d} | {:7d}".format(num_training_slides, num_training_nodes))
        print("  test     |  {:8d} | {:7d}".format(num_testing_slides, num_testing_nodes))
        print("  -----------------------------------------")
        print("  original train spots: {:7d}".format(ori_num_training_spots))
        print("  original test spots:  {:7d}".format(ori_num_testing_spots))

    def _coordinate_keys(self, adata):
        rows = adata.obs["array_row"].values
        cols = adata.obs["array_col"].values
        return np.array([f"{int(row)}_{int(col)}" for row, col in zip(rows, cols)], dtype=object)

    def _build_graph(self, rna_adata, msi_adata, split, slide_name, mask_seed):
        rna_array = to_numpy_array(rna_adata.X)[:, self.rna_mask]
        msi_array = to_numpy_array(msi_adata.X)[:, self.msi_mask]

        rna_keys = self._coordinate_keys(rna_adata)
        msi_keys = self._coordinate_keys(msi_adata)

        rna_key_to_index = {key: idx for idx, key in enumerate(rna_keys)}
        msi_key_to_index = {key: idx for idx, key in enumerate(msi_keys)}

        common_keys = []
        rna_indices = []
        msi_indices = []

        for key in rna_keys:
            if key not in msi_key_to_index:
                continue

            rna_index = rna_key_to_index[key]
            msi_index = msi_key_to_index[key]
            if rna_array[rna_index].sum() == 0 or msi_array[msi_index].sum() == 0:
                continue

            common_keys.append(key)
            rna_indices.append(rna_index)
            msi_indices.append(msi_index)

        if not common_keys:
            raise ValueError(f"No aligned SMA nodes found for slide '{slide_name}'.")

        coordinates = rna_adata.obs[["array_row", "array_col"]].values[rna_indices]

        graph = build_slice_graph(
            node_features=rna_array[rna_indices],
            node_targets=msi_array[msi_indices],
            coordinates=coordinates,
            split=split,
            k=self.graph_k,
            val_ratio=self.val_ratio,
            mask_seed=mask_seed,
            node_ids=[f"{slide_name}/{node_id}" for node_id in common_keys],
            sample_id=slide_name,
            slice_name=slide_name,
        )
        return graph


if __name__ == "__main__":
    dataset = SMA("", "", "")
