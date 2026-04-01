import numpy as np
import pandas as pd
import scanpy as sc

from datasets.graph_utils import build_slice_graph, count_graph_nodes, to_numpy_array


class Breast_cancer(object):
    def __init__(self, adata_path, coordinate_path, ct_path, graph_k=6, val_ratio=0.1, mask_seed=0):
        self.graph_k = graph_k
        self.val_ratio = val_ratio
        self.mask_seed = mask_seed

        adata = sc.read_h5ad(adata_path)
        coordinates = pd.read_csv(coordinate_path, compression="gzip")
        ct = pd.read_excel(ct_path, sheet_name="Xenium R1 Fig1-5 (supervised)")

        adata.obs["cell_CD20_mean"] = np.log(adata.obs["cell_CD20_mean"].values + 1)
        adata.obs["cell_HER2_mean"] = np.log(adata.obs["cell_HER2_mean"].values + 1)

        sc.pp.normalize_total(adata, target_sum=1e3)
        sc.pp.log1p(adata)

        adata.obs["x"] = coordinates["x_centroid"].values
        adata.obs["y"] = coordinates["y_centroid"].values
        adata.obs["ct"] = ct["Cluster"].values

        self.cell_mask = np.array(sorted(set(ct["Cluster"].values.tolist())))

        center_x = adata.obs["x"].values.max() // 2
        train_mask = adata.obs["x"].values < center_x
        test_mask = ~train_mask
        training_adata = adata[train_mask].copy()
        testing_adata = adata[test_mask].copy()

        self.training = [
            self._build_graph(
                rna_adata=training_adata,
                split="train",
                slice_name="breast_cancer_train",
                mask_seed=mask_seed,
            )
        ]
        self.testing = [
            self._build_graph(
                rna_adata=testing_adata,
                split="test",
                slice_name="breast_cancer_test",
                mask_seed=mask_seed + 1,
            )
        ]
        self.val = self.training

        self.rna_length = int(training_adata.shape[1])
        self.protein_length = 2
        self.num_cell_types = int(len(self.cell_mask))
        self.source_panel = training_adata.var_names.tolist()
        self.target_panel = np.array(["CD20", "HER2"])

        num_training_nodes = count_graph_nodes(self.training)
        num_testing_nodes = count_graph_nodes(self.testing)

        print("=> Breast cancer loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # graphs | # nodes")
        print("  ------------------------------")
        print("  train    |  {:8d} | {:7d}".format(len(self.training), num_training_nodes))
        print("  test     |  {:8d} | {:7d}".format(len(self.testing), num_testing_nodes))
        print("  ------------------------------")

    def _build_graph(self, rna_adata, split, slice_name, mask_seed):
        # x: [N, 313]
        rna_array = to_numpy_array(rna_adata.X)
        # y: [N, 2]
        proteins = rna_adata.obs[["cell_CD20_mean", "cell_HER2_mean"]].values.astype(np.float32)
        # pos: [N, 2]
        coordinates = rna_adata.obs[["x", "y"]].values.astype(np.float32)
        cell_types = rna_adata.obs["ct"].values
        node_ids = rna_adata.obs.index.tolist()

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
    dataset = Breast_cancer("", "", "")
