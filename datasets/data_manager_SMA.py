import os

import numpy as np
import pandas as pd
import scanpy as sc

import sklearn.neighbors

from collections import defaultdict
from datasets.cell_type_utils import resolve_global_cell_types


def Cal_Spatial_Net_row_col(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """
    Build a spatial neighbour dictionary from ``array_row`` and ``array_col``.

    Parameters
    ----------
    adata : AnnData
        AnnData object whose ``adata.obs`` contains ``array_row`` and
        ``array_col``.
    rad_cutoff : float, optional
        Distance threshold used when ``model='Radius'``.
    k_cutoff : int, optional
        Number of nearest neighbours used when ``model='KNN'``.
    model : {'Radius', 'KNN'}
        Strategy used to build the neighbourhood graph.
    verbose : bool
        Whether to print graph statistics.

    Returns
    -------
    temp_dic : collections.defaultdict[list]
        Maps each spot key formatted as ``"row_col"`` to a list of neighbour
        spot keys in the same format.
    """

    assert model in ['Radius', 'KNN']
    if verbose:
        print('------Calculating spatial graph...')

    # Use array-row / array-col coordinates as the spatial positions.
    coor = pd.DataFrame(np.stack([adata.obs['array_row'], adata.obs['array_col']], axis=1))

    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        # Request one extra neighbour because each spot also returns itself.
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    # Drop zero-distance self loops.
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0, ]

    # Convert internal row indices back to AnnData spot indices.
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index)))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    # Store the edge list on the AnnData object for downstream inspection.
    adata.uns['Spatial_Net'] = Spatial_Net

    # Convert spot indices into the "row_col" coordinate keys used elsewhere.
    temp_dic = defaultdict(list)
    for i in range(Spatial_Net.shape[0]):

        center = Spatial_Net.iloc[i, 0]
        side = Spatial_Net.iloc[i, 1]

        center_name = str(adata.obs['array_row'][center]) + '_' + str(adata.obs['array_col'][center])
        side_name = str(adata.obs['array_row'][side]) + '_' + str(adata.obs['array_col'][side])

        temp_dic[center_name].append(side_name)

    return temp_dic


class SMA(object):
    """Dataset manager for the SMA slides used by the project."""

    def __init__(
        self,
        path_img,
        rna_path,
        msi_path,
        n_top_genes=3000,
        n_top_targets=50,
        cell_type_visualize=False,
        cell_type_visualization_dir=None,
        cell_type_visualization_dpi=150,
    ):

        training_slides = ['V11L12-109_B1', 'V11L12-109_C1']
        testing_slides = ['V11L12-109_A1']

        self.path_img = path_img

        rna_adata_list, msi_adata_list = [], []
        rna_highly_variable_list, msi_highly_variable_list = [], []

        for slide in training_slides + testing_slides:

            adata_rna = sc.read_visium(os.path.join(rna_path, slide))
            adata_rna.var_names_make_unique()

            # Select highly variable genes on raw counts, then normalise and log-transform.
            sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=n_top_genes)
            sc.pp.normalize_total(adata_rna, target_sum=1e4)
            sc.pp.log1p(adata_rna)

            rna_adata_list.append(adata_rna)
            rna_highly_variable_list.append(adata_rna.var['highly_variable'].values)

            adata_msi = sc.read_h5ad(os.path.join(msi_path, 'metabolite_' + slide + '.h5ad'))
            adata_msi.var_names_make_unique()
            # The MSI matrix is already aligned by spot, so only HVG selection and log1p are applied.
            sc.pp.highly_variable_genes(adata_msi, flavor="seurat_v3", n_top_genes=n_top_targets)
            sc.pp.log1p(adata_msi)

            msi_adata_list.append(adata_msi)
            msi_highly_variable_list.append(adata_msi.var['highly_variable'].values)

        # Estimate RNA feature statistics across all slides for later standardisation.
        temp = np.concatenate(
            [rna_adata_list[0].X.toarray(), rna_adata_list[1].X.toarray(), rna_adata_list[2].X.toarray()],
            axis=0,
        )
        self.rna_mean, self.rna_std = temp.mean(axis=0)[None, ], temp.std(axis=0)[None, ]

        temp_mask = (self.rna_std == 0)
        # Avoid division-by-zero when downstream code applies z-score normalisation.
        self.rna_std[temp_mask] = 1

        # Keep only features that are marked as highly variable in every slide.
        self.rna_mask = rna_highly_variable_list[0] & rna_highly_variable_list[1] & rna_highly_variable_list[2]
        self.msi_mask = msi_highly_variable_list[0] & msi_highly_variable_list[1] & msi_highly_variable_list[2]

        all_slides = training_slides + testing_slides
        cell_type_info = resolve_global_cell_types(
            adata_list=rna_adata_list,
            slice_names=all_slides,
            feature_masks=self.rna_mask,
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

        print(f'  Global cell-type source: {self.cell_type_source}')
        print(f'  Total global cell types used for embedding: {self.n_spot_types}')
        if self.cell_type_visualization_paths:
            print(f'  Cell-type visualization slices: {sorted(self.cell_type_visualization_paths)}')

        self.training = self._process_data(
            rna_adata_list[0:2], msi_adata_list[0:2],
            training_slides)
        self.testing = self._process_data(
            rna_adata_list[2:], msi_adata_list[2:],
            testing_slides)

        self.rna_length = (self.rna_mask * 1).sum()
        self.msi_length = (self.msi_mask * 1).sum()
        # Aliases expected by the training script.
        self.source_length = int(self.rna_length)
        self.target_length = int(self.msi_length)
        # Use the final slide metadata to expose the selected panel names.
        self.target_panel = adata_msi.var['metabolism'].values[self.msi_mask].tolist()
        self.source_panel = adata_rna.var_names[self.rna_mask]

        num_training_spots, num_testing_spots = len(self.training), len(self.testing)
        num_training_slides, num_testing_slides = len(training_slides), len(testing_slides)

        ori_num_training_spots = rna_adata_list[0].shape[0] + rna_adata_list[1].shape[0]
        ori_num_testing_spots = rna_adata_list[2].shape[0]

        print("=> SMA loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # num | ")
        print("  ------------------------------")
        print("  train    |  Without filtering {:5d} spots from {:5d} slides ".format(ori_num_training_spots, num_training_slides))
        print("  test     |  Without filtering {:5d} spots from {:5d} slides".format(ori_num_testing_spots, num_testing_slides))
        print("  train    |  After filting {:5d} spots from {:5d} slides ".format(num_training_spots, num_training_slides))
        print("  test     |  After filting {:5d} spots from {:5d} slides".format(num_testing_spots, num_testing_slides))
        print("  ------------------------------")

    def _dictionary_data(self, adata):
        """Convert the expression matrix into a ``row_col -> feature vector`` dictionary."""
        dictionary = {}
        array_row, array_col = adata.obs['array_row'].values, adata.obs['array_col'].values

        array = adata.X.toarray()

        for i in range(adata.shape[0]):
            dictionary[str(int(array_row[i])) + '_' + str(int(array_col[i]))] = array[i]

        return dictionary

    def _process_data(self, rna_adata_list, msi_adata_list, slides):
        """Build the list of per-spot training or testing tuples.

        Each element is::

            (img_path, rna_temp, msi_temp, rna_neighbors, msi_neighbors,
             spot_type_ids, sample_id)

        ``spot_type_ids`` stores the center spot type followed by first-order
        and second-order neighbour types. Missing or padded neighbours use ``-1``.
        """

        dataset = []

        for dataset_index in range(len(rna_adata_list)):
            slide = slides[dataset_index]
            rna_temp_adata = rna_adata_list[dataset_index]
            msi_temp_adata = msi_adata_list[dataset_index]

            rna_dic = self._dictionary_data(rna_temp_adata)
            msi_dic = self._dictionary_data(msi_temp_adata)

            # Build two spatial rings around each spot using row/col coordinates.
            graph_1 = Cal_Spatial_Net_row_col(rna_temp_adata, rad_cutoff=2 ** (1 / 2), model='Radius')
            graph_2 = Cal_Spatial_Net_row_col(rna_temp_adata, rad_cutoff=2, model='Radius')

            rna_keys = list(rna_dic.keys())
            rna_key_set = set(rna_keys)
            msi_keys = set(msi_dic.keys())
            global_ids = self.global_cell_type_ids_by_slice[slide]
            global_id_by_key = {}
            for obs_index, global_id in zip(rna_keys, global_ids):
                global_id_by_key[obs_index] = int(global_id)

            for key in rna_keys:
                # Only keep spots that exist in both RNA and MSI modalities.
                if key not in msi_keys:
                    continue
                else:
                    rna_temp = rna_dic[key][self.rna_mask]
                    msi_temp = msi_dic[key][self.msi_mask]

                    # Discard spots with empty signal after masking.
                    if rna_temp.sum() == 0 or msi_temp.sum() == 0:
                        continue
                    else:
                        img_path = os.path.join(self.path_img, slide, key + '.png')

                        spot_type_id = global_id_by_key.get(key, -1)

                        rna_neighbors, msi_neighbors = [], []
                        neighbor1_spot_type_ids, neighbor2_spot_type_ids = [], []

                        neighbors_1, neighbors_2 = graph_1[key], graph_2[key]
                        # Keep second-ring neighbours distinct from the first ring.
                        neighbors_2 = [item for item in neighbors_2 if item not in neighbors_1]

                        # Collect first-ring neighbour features and neighbour spot types.
                        for j in neighbors_1:
                            if j not in rna_key_set:
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                neighbor1_spot_type_ids.append(-1)
                            else:
                                rna_neighbors.append(rna_dic[j][self.rna_mask])
                                neighbor1_spot_type_ids.append(global_id_by_key.get(j, -1))

                            if j not in msi_keys:
                                msi_neighbors.append(np.zeros_like(msi_temp))
                            else:
                                msi_neighbors.append(msi_dic[j][self.msi_mask])

                        # Pad the first ring to four slots for downstream tensor shapes.
                        if len(neighbors_1) != 4:
                            for _ in range(4 - len(neighbors_1)):
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                msi_neighbors.append(np.zeros_like(msi_temp))
                                neighbor1_spot_type_ids.append(-1)

                        # Collect second-ring neighbour features and neighbour spot types.
                        for j in neighbors_2:
                            if j not in rna_key_set:
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                neighbor2_spot_type_ids.append(-1)
                            else:
                                rna_neighbors.append(rna_dic[j][self.rna_mask])
                                neighbor2_spot_type_ids.append(global_id_by_key.get(j, -1))

                            if j not in msi_keys:
                                msi_neighbors.append(np.zeros_like(msi_temp))
                            else:
                                msi_neighbors.append(msi_dic[j][self.msi_mask])

                        # Pad the second ring to four slots as well.
                        if len(neighbors_2) != 4:
                            for _ in range(4 - len(neighbors_2)):
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                msi_neighbors.append(np.zeros_like(msi_temp))
                                neighbor2_spot_type_ids.append(-1)

                        # Final sample structure:
                        #   center RNA + center MSI + 8 RNA neighbours + 8 MSI neighbours
                        #   + 9 spot-type IDs (center + neighbours) + sample ID.
                        rna_neighbors = np.stack(rna_neighbors)
                        msi_neighbors = np.stack(msi_neighbors)
                        spot_type_ids = np.asarray(
                            [spot_type_id] + neighbor1_spot_type_ids + neighbor2_spot_type_ids,
                            dtype=np.int64,
                        )

                        dataset.append((img_path, rna_temp, msi_temp, rna_neighbors, msi_neighbors,
                                        spot_type_ids, slide + '/' + key))

        return dataset


if __name__ == '__main__':
    dataset = SMA()
