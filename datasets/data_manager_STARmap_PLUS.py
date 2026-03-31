import os

import numpy as np
import pandas as pd
import scanpy as sc

import sklearn.neighbors
from collections import defaultdict

from datasets.cell_type_utils import resolve_global_cell_types


def Cal_Spatial_Net_row_col(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True, mouse=False):
    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    if mouse:
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
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0, ]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index)))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

    temp_dic = defaultdict(list)

    if mouse:
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


def _resolve_protein_columns(adata):
    columns = list(adata.obs.columns)
    tau_key = 'p-tau' if 'p-tau' in columns else None
    if tau_key is None:
        raise KeyError('Expected a p-tau column in adata.obs for AD_Mouse.')

    preferred_secondary = ['plaque', 'abeta', 'a_beta', 'amyloid_beta', 'amyloid', 'aβ', 'aβ']
    plaque_key = None
    lower_to_original = {col.lower(): col for col in columns}
    for candidate in preferred_secondary:
        if candidate in lower_to_original:
            plaque_key = lower_to_original[candidate]
            break

    if plaque_key is None:
        remaining = [col for col in columns if col != tau_key]
        plaque_like = [col for col in remaining if ('plaque' in col.lower()) or ('beta' in col.lower()) or ('amyloid' in col.lower()) or ('β' in col.lower())]
        if plaque_like:
            plaque_key = plaque_like[0]
        else:
            raise KeyError('Could not identify the second protein target column for AD_Mouse.')

    return [tau_key, plaque_key]


class AD_Mouse(object):
    def __init__(
        self,
        AD_adata_path,
        Wild_type_adata_path,
        n_top_genes=3000,
        testing_control=False,
        cell_type_visualize=False,
        cell_type_visualization_dir=None,
        cell_type_visualization_dpi=150,
    ):

        training_slides = ['13months-disease-replicate_1_random.h5ad']
        testing_slides = ['13months-disease-replicate_2_random.h5ad']

        self.cell_type = 'ct_top'
        adata_list = []

        for slide in training_slides + testing_slides:
            path = os.path.join(AD_adata_path, slide)
            adata_temp = sc.read_h5ad(path)

            if 'highly_variable' not in adata_temp.var.columns:
                sc.pp.highly_variable_genes(adata_temp, flavor='seurat_v3', n_top_genes=n_top_genes)
            adata_list.append(adata_temp)

        val_adata = [sc.read_h5ad(os.path.join(Wild_type_adata_path, 'spatial_13months-control-replicate_1.h5ad'))]
        val_slides = ['spatial_13months-control-replicate_1.h5ad']

        self.protein_obs_columns = _resolve_protein_columns(adata_list[0])
        self.rna_mask = adata_list[0].var['highly_variable'].values & adata_list[1].var['highly_variable'].values

        all_slides = training_slides + testing_slides + val_slides
        all_adatas = adata_list + val_adata

        # Provided annotation branch: use the dataset's cell-type label column
        # directly and encode those names into one global ID space shared by all
        # slices. No Leiden clustering is run for this dataset.
        cell_type_info = resolve_global_cell_types(
            adata_list=all_adatas,
            slice_names=all_slides,
            annotation_key=self.cell_type,
            candidate_annotation_keys=[self.cell_type],
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

        self.training, training_tau, training_abeta, _ = self._process_data(
            adata_list[0:len(training_slides)], training_slides
        )
        self.testing, testing_tau, testing_abeta, graph = self._process_data(
            adata_list[len(training_slides):], testing_slides
        )
        self.val, _, _, _ = self._process_data(val_adata, val_slides, WT=True)

        self.graph = graph

        self.rna_length = self.rna_mask.sum()
        self.source_length = int(self.rna_length)
        self.target_length = 2
        self.target_panel = np.array(['tau', 'plaque'])
        self.source_panel = adata_temp.var_names[self.rna_mask]

        num_training_spots = len(self.training)
        num_testing_spots = len(self.testing)

        print('=> AD Mouse loaded')
        print('Dataset statistics:')
        print('  ------------------------------')
        print('  subset   | # num | ')
        print('  ------------------------------')
        print('  train    |  {:5d} spots, {} positive tao, {} positive plaque '.format(num_training_spots, training_tau, training_abeta))
        print('  test     |  {:5d} spots, {} positive tao, {} positive plaque '.format(num_testing_spots, testing_tau, testing_abeta))
        print('  ------------------------------')
        print(f'  Global cell-type source: {self.cell_type_source}')
        print(f'  Global cell types ({self.n_spot_types}): {self.cell_mask.tolist()}')
        if self.cell_type_visualization_paths:
            print(f'  Cell-type visualization slices: {sorted(self.cell_type_visualization_paths)}')

    def _process_data(self, adata_list, slides, WT=False):
        """Build per-spot sample tuples.

        Each element is::

            (rna, protein, cell_onehot, rna_neighbor, cell_neighbor,
             spot_type_ids, sample_id)

        ``cell_onehot`` stores the one-hot encoding of the final
        ``global_cell_type_id`` for the center spot and its neighbors.
        ``spot_type_ids`` stores the center global ID followed by one ID per
        neighbor token, in the same order as ``rna_neighbor`` / ``cell_neighbor``.
        """

        tau, abeta = 0, 0
        dataset = []

        for dataset_index, rna_adata in enumerate(adata_list):
            slide = slides[dataset_index]

            if WT:
                proteins = np.zeros((rna_adata.shape[0], 2))
            else:
                proteins = rna_adata.obs[self.protein_obs_columns].values

            graph = Cal_Spatial_Net_row_col(rna_adata, k_cutoff=12, model='KNN', mouse=True)

            rna_array = rna_adata.X[:, self.rna_mask]
            indexes = rna_adata.obs.index.tolist()
            global_cell_type_ids = self.global_cell_type_ids_by_slice[slide]

            tau += proteins[:, 0].sum()
            abeta += proteins[:, 1].sum()

            dict_rna = {}
            dict_cell = {}
            dict_spot_type = {}
            for i, obs_index in enumerate(indexes):
                dict_rna[obs_index] = rna_array[i]
                cell_onehot = np.zeros(self.n_cell_types, dtype=np.float32)
                cell_onehot[int(global_cell_type_ids[i])] = 1.0
                dict_cell[obs_index] = cell_onehot
                dict_spot_type[obs_index] = int(global_cell_type_ids[i])

            for i in range(rna_adata.shape[0]):
                rna_neighbor, cell_neighbor = [], []

                obs_index = indexes[i]
                cell_onehot = dict_cell[obs_index]
                rna, protein = rna_array[i], proteins[i]
                spot_type_id = int(global_cell_type_ids[i])
                neighbor_spot_type_ids = []

                for neighbor_index in graph[obs_index]:
                    rna_neighbor.append(dict_rna[neighbor_index])
                    cell_neighbor.append(dict_cell[neighbor_index])
                    neighbor_spot_type_ids.append(dict_spot_type[neighbor_index])

                rna_neighbor = np.array(rna_neighbor)
                cell_neighbor = np.array(cell_neighbor)
                spot_type_ids = np.asarray([spot_type_id] + neighbor_spot_type_ids, dtype=np.int64)

                dataset.append((
                    rna,
                    protein,
                    cell_onehot,
                    rna_neighbor,
                    cell_neighbor,
                    spot_type_ids,
                    slide + '/' + obs_index,
                ))

        return dataset, tau, abeta, graph


if __name__ == '__main__':
    dataset = AD_Mouse()
