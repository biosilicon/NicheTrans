import os

import numpy as np
import pandas as pd
import scanpy as sc

import sklearn.neighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse

from collections import defaultdict


def assign_spot_types(adata, n_pcs=30, n_neighbors=15, resolution=0.5, verbose=True):
    """
    Classify each spot into a discrete spot type via Leiden clustering on the
    transcriptomic PCA embedding.

    The clustering is performed on the pre-processed (normalised + log1p)
    expression matrix stored in ``adata.X``. Only HVGs should have been
    selected before calling this function so that the PCA is meaningful.

    Parameters
    ----------
    adata : AnnData
        Must contain log-normalised counts in ``adata.X``.
    n_pcs : int
        Number of principal components to compute and use for neighbour graph.
    n_neighbors : int
        Number of neighbours for the kNN graph used by Leiden.
    resolution : float
        Leiden resolution; higher values produce more (finer) clusters.
    verbose : bool
        Whether to print cluster statistics.

    Returns
    -------
    spot_type_ids : np.ndarray, shape (n_obs,), dtype int
        Integer cluster label for every spot (0-indexed).
    n_types : int
        Total number of unique spot types found.
    """
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.leiden(adata, resolution=resolution, key_added='spot_type')

    # Map string cluster labels ("0", "1", ...) to integers.
    spot_type_ids = adata.obs['spot_type'].astype(int).values
    n_types = int(spot_type_ids.max()) + 1

    if verbose:
        counts = np.bincount(spot_type_ids)
        print(f'  Spot-type clustering: {n_types} types  '
              f'(sizes: {counts.tolist()})')

    return spot_type_ids, n_types


def align_clusters_across_slices(adata_hvg_list, type_ids_list, n_types_list,
                                  similarity_threshold=0.5, verbose=True):
    """
    Align per-slice Leiden cluster IDs into a unified global cell-type ID space.

    After independent Leiden clustering on each slice, cluster 0 on slice A may
    correspond to a completely different population than cluster 0 on slice B.
    This function resolves that ambiguity by:

    1. Computing per-cluster centroid expression vectors (mean over HVGs).
    2. Designating the first slice as the reference (local IDs become global IDs).
    3. For every subsequent slice, matching its clusters to the reference using
       cosine similarity plus the Hungarian algorithm.
    4. Assigning new global IDs to unmatched clusters.

    Parameters
    ----------
    adata_hvg_list : list[AnnData]
        HVG-filtered AnnData objects, one per slice.
    type_ids_list : list[np.ndarray]
        Per-spot integer Leiden labels for each slice.
    n_types_list : list[int]
        Number of clusters found in each slice.
    similarity_threshold : float
        Minimum cosine similarity to accept a Hungarian match.
    verbose : bool
        Print alignment diagnostics.

    Returns
    -------
    global_mapping : dict[(int, int), int]
        ``(slice_idx, local_cluster_id) -> global_cell_type_id``.
    n_global_types : int
        Total number of distinct global cell-type IDs.
    alignment_info : dict
        Diagnostic information such as matched pairs for each slice.
    """
    n_slices = len(adata_hvg_list)

    # Step 1: compute one centroid expression profile per cluster per slice.
    centroids_list = []  # centroids_list[s] shape: (n_types_list[s], n_hvg)
    for s in range(n_slices):
        X = adata_hvg_list[s].X
        if issparse(X):
            X = X.toarray()
        n_clusters = n_types_list[s]
        ids = type_ids_list[s]
        centroids = np.zeros((n_clusters, X.shape[1]))
        for c in range(n_clusters):
            mask = (ids == c)
            if mask.sum() > 0:
                centroids[c] = X[mask].mean(axis=0)
        centroids_list.append(centroids)

    # Step 2: use slice 0 as the reference ID space.
    global_mapping = {}
    for c in range(n_types_list[0]):
        global_mapping[(0, c)] = c
    next_global_id = n_types_list[0]

    alignment_info = {'reference_slice': 0, 'matches': {}}

    # Step 3: align every later slice against the reference slice.
    ref_centroids = centroids_list[0]

    for s in range(1, n_slices):
        cur_centroids = centroids_list[s]
        # Cosine similarity matrix with shape (n_cur_clusters, n_ref_clusters).
        sim_matrix = cosine_similarity(cur_centroids, ref_centroids)
        cost_matrix = 1.0 - sim_matrix

        # Hungarian algorithm finds the best one-to-one cluster assignment.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        slice_matches = []
        matched_rows = set()
        for r, c_ref in zip(row_ind, col_ind):
            sim_val = sim_matrix[r, c_ref]
            if sim_val >= similarity_threshold:
                # Reuse the reference global ID for sufficiently similar clusters.
                global_mapping[(s, int(r))] = global_mapping[(0, int(c_ref))]
                matched_rows.add(r)
                slice_matches.append((int(r), int(c_ref), float(sim_val)))

        # Assign new global IDs to clusters that could not be matched.
        for c in range(n_types_list[s]):
            if c not in matched_rows:
                global_mapping[(s, c)] = next_global_id
                next_global_id += 1
                if verbose:
                    print(f'  Slice {s}, local cluster {c}: no match '
                          f'(new global ID {global_mapping[(s, c)]})')

        alignment_info['matches'][s] = slice_matches

        if verbose:
            print(f'  Slice {s} alignment: {len(slice_matches)}/{n_types_list[s]} '
                  f'clusters matched to reference')
            for r, c_ref, sim_val in slice_matches:
                print(f'    local {r} -> ref {c_ref} (global {global_mapping[(0, c_ref)]}), '
                      f'cosine sim = {sim_val:.4f}')

    n_global_types = next_global_id
    if verbose:
        print(f'  Cross-slice alignment complete: {n_global_types} global cell types')

    return global_mapping, n_global_types, alignment_info


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

    def __init__(self, path_img, rna_path, msi_path, n_top_genes=3000, n_top_targets=50):

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

        # Cluster each slide independently, then align the local cluster IDs
        # so biologically similar clusters share the same global spot-type ID.
        adata_hvg_list = []
        type_ids_list = []
        n_types_per_slide = []

        for idx, slide in enumerate(training_slides + testing_slides):
            adata_rna = rna_adata_list[idx]
            # PCA and Leiden are run only on the HVG subset used for training.
            adata_hvg = adata_rna[:, self.rna_mask].copy()
            type_ids, n_t = assign_spot_types(adata_hvg, verbose=True)
            adata_hvg_list.append(adata_hvg)
            type_ids_list.append(type_ids)
            n_types_per_slide.append(n_t)

        # Align local cluster IDs into one global ID space across slices.
        global_mapping, n_global_types, alignment_info = align_clusters_across_slices(
            adata_hvg_list, type_ids_list, n_types_per_slide, verbose=True)
        self.n_spot_types = n_global_types
        self.global_mapping = global_mapping  # (slice_idx, local_id) -> global_id

        # Build a per-spot lookup table keyed by "slide/row_col".
        all_spot_type_ids = {}  # key: "slide/row_col" -> global cell-type ID
        for idx, slide in enumerate(training_slides + testing_slides):
            adata_rna = rna_adata_list[idx]
            array_row = adata_rna.obs['array_row'].values
            array_col = adata_rna.obs['array_col'].values
            for i in range(adata_rna.shape[0]):
                coord_key = str(int(array_row[i])) + '_' + str(int(array_col[i]))
                local_id = int(type_ids_list[idx][i])
                all_spot_type_ids[slide + '/' + coord_key] = global_mapping[(idx, local_id)]

        print(f'  Total global cell types used for embedding: {self.n_spot_types}')

        self.training = self._process_data(
            rna_adata_list[0:2], msi_adata_list[0:2],
            training_slides, all_spot_type_ids)
        self.testing = self._process_data(
            rna_adata_list[2:], msi_adata_list[2:],
            testing_slides, all_spot_type_ids)

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

    def _dictionary_data(self, adata, rna=False):
        """Convert the expression matrix into a ``row_col -> feature vector`` dictionary."""
        dictionary = {}
        array_row, array_col = adata.obs['array_row'].values, adata.obs['array_col'].values

        array = adata.X.toarray()

        for i in range(adata.shape[0]):
            dictionary[str(int(array_row[i])) + '_' + str(int(array_col[i]))] = array[i]

        return dictionary

    def _process_data(self, rna_adata_list, msi_adata_list, names, all_spot_type_ids):
        """Build the list of per-spot training or testing tuples.

        Each element is::

            (img_path, rna_temp, msi_temp, rna_neighbors, msi_neighbors,
             spot_type_ids, sample_id)

        ``spot_type_ids`` stores the center spot type followed by first-order
        and second-order neighbour types. Missing or padded neighbours use ``-1``.
        """

        dataset = []

        for i in range(len(rna_adata_list)):

            rna_temp_adata = rna_adata_list[i]
            msi_temp_adata = msi_adata_list[i]

            rna_dic = self._dictionary_data(rna_temp_adata, rna=True)
            msi_dic = self._dictionary_data(msi_temp_adata)

            # Build two spatial rings around each spot using row/col coordinates.
            graph_1 = Cal_Spatial_Net_row_col(rna_temp_adata, rad_cutoff=2 ** (1 / 2), model='Radius')
            graph_2 = Cal_Spatial_Net_row_col(rna_temp_adata, rad_cutoff=2, model='Radius')

            rna_keys, msi_keys = rna_dic.keys(), msi_dic.keys()

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
                        img_path = os.path.join(self.path_img, names[i], key + '.png')

                        # Retrieve the globally aligned spot type for the center spot.
                        # Fall back to 0 only if the coordinate is missing after filtering.
                        global_key = names[i] + '/' + key
                        spot_type_id = all_spot_type_ids.get(global_key, 0)

                        rna_neighbors, msi_neighbors = [], []
                        neighbor1_spot_type_ids, neighbor2_spot_type_ids = [], []

                        neighbors_1, neighbors_2 = graph_1[key], graph_2[key]
                        # Keep second-ring neighbours distinct from the first ring.
                        neighbors_2 = [item for item in neighbors_2 if item not in neighbors_1]

                        # Collect first-ring neighbour features and neighbour spot types.
                        for j in neighbors_1:
                            if j not in rna_keys:
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                neighbor1_spot_type_ids.append(-1)
                            else:
                                rna_neighbors.append(rna_dic[j][self.rna_mask])
                                neighbor1_spot_type_ids.append(
                                    all_spot_type_ids.get(names[i] + '/' + j, -1)
                                )

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
                            if j not in rna_keys:
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                neighbor2_spot_type_ids.append(-1)
                            else:
                                rna_neighbors.append(rna_dic[j][self.rna_mask])
                                neighbor2_spot_type_ids.append(
                                    all_spot_type_ids.get(names[i] + '/' + j, -1)
                                )

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
                                        spot_type_ids, names[i] + '/' + key))

        return dataset


if __name__ == '__main__':
    dataset = SMA()
