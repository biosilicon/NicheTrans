import os

import numpy as np
import scanpy as sc

from datasets.graph_utils import build_slice_graph, count_graph_nodes, to_numpy_array


class AD_Mouse(object):
    def __init__(
        self,
        AD_adata_path,
        Wild_type_adata_path,
        n_top_genes=3000,
        testing_control=False,
        graph_k=6,
        val_ratio=0.1,
        mask_seed=0,
    ):
        del testing_control

        self.graph_k = graph_k
        self.val_ratio = val_ratio
        self.mask_seed = mask_seed
        self.cell_type = "ct_top"

        training_slides = ["13months-disease-replicate_1_random.h5ad"]
        testing_slides = ["13months-disease-replicate_2_random.h5ad"]

        adata_list = []
        for slide in training_slides + testing_slides:
            path = os.path.join(AD_adata_path, slide)
            adata_temp = sc.read_h5ad(path)
            if "highly_variable" not in adata_temp.var.columns:
                sc.pp.highly_variable_genes(adata_temp, flavor="seurat_v3", n_top_genes=n_top_genes)
            adata_list.append(adata_temp)

        self.rna_mask = adata_list[0].var["highly_variable"].values & adata_list[1].var["highly_variable"].values
        self.cell_mask = np.array(sorted(set(adata_list[0].obs[self.cell_type].values) & set(adata_list[1].obs[self.cell_type].values)))
        self.target_columns = self._select_target_columns(adata_list[0].obs.columns)

        val_adata = [sc.read_h5ad(os.path.join(Wild_type_adata_path, "spatial_13months-control-replicate_1.h5ad"))]
        val_slides = ["spatial_13months-control-replicate_1.h5ad"]

        self.training = [
            self._build_graph(
                rna_adata=adata_list[0],
                split="train",
                slice_name=training_slides[0],
                mask_seed=mask_seed,
                wt=False,
            )
        ]
        self.testing = [
            self._build_graph(
                rna_adata=adata_list[1],
                split="test",
                slice_name=testing_slides[0],
                mask_seed=mask_seed + 1,
                wt=False,
            )
        ]
        self.val = [
            self._build_graph(
                rna_adata=val_adata[0],
                split="val",
                slice_name=val_slides[0],
                mask_seed=mask_seed + 2,
                wt=True,
            )
        ]

        self.rna_length = int(self.rna_mask.sum())
        self.target_length = 2
        self.num_cell_types = int(len(self.cell_mask))
        self.target_panel = np.array(["tau", "plaque"])
        self.source_panel = adata_list[0].var_names[self.rna_mask]

        num_training_nodes = count_graph_nodes(self.training)
        num_testing_nodes = count_graph_nodes(self.testing)
        num_val_nodes = count_graph_nodes(self.val)

        print("=> AD Mouse loaded")
        print("Dataset statistics:")
        print("  -----------------------------------------")
        print("  subset   | # graphs | # nodes")
        print("  -----------------------------------------")
        print("  train    |  {:8d} | {:7d}".format(len(self.training), num_training_nodes))
        print("  val      |  {:8d} | {:7d}".format(len(self.val), num_val_nodes))
        print("  test     |  {:8d} | {:7d}".format(len(self.testing), num_testing_nodes))
        print("  -----------------------------------------")

    def _select_target_columns(self, obs_columns):
        preferred_pairs = [
            ["p-tau", "Aβ"],
            ["p-tau", "Abeta"],
            ["p-tau", "Aå°¾"],
        ]
        for columns in preferred_pairs:
            if all(column in obs_columns for column in columns):
                return columns

        tau_candidates = [column for column in obs_columns if "tau" in column.lower()]
        plaque_candidates = [
            column
            for column in obs_columns
            if column not in tau_candidates and ("pla" in column.lower() or "beta" in column.lower() or column.lower().startswith("a"))
        ]
        if tau_candidates and plaque_candidates:
            return [tau_candidates[0], plaque_candidates[0]]

        raise KeyError("Could not infer the STARmap_PLUS target columns from AnnData.obs.")

    def _build_graph(self, rna_adata, split, slice_name, mask_seed, wt=False):
        if wt:
            proteins = np.zeros((rna_adata.shape[0], len(self.target_columns)), dtype=np.float32)
        else:
            proteins = rna_adata.obs[self.target_columns].values.astype(np.float32)

        # x: [N, source_dim]
        rna_array = to_numpy_array(rna_adata.X)[:, self.rna_mask]
        # pos: [N, 2]
        coordinates = rna_adata.obs[["x", "y"]].values.astype(np.float32)
        cell_types = rna_adata.obs[self.cell_type].values
        node_ids = [f"{slice_name}/{node_id}" for node_id in rna_adata.obs.index.tolist()]

        graph = build_slice_graph(
            node_features=rna_array,
            node_targets=proteins,
            coordinates=coordinates,
            split=split,
            k=self.graph_k,
            val_ratio=self.val_ratio,
            mask_seed=mask_seed,
            cell_type=cell_types,
            cell_type_vocabulary=self.cell_mask,
            node_ids=node_ids,
            sample_id=slice_name,
            slice_name=slice_name,
        )
        return graph


if __name__ == "__main__":
    dataset = AD_Mouse("", "")
