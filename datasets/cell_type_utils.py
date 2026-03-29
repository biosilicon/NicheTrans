import numpy as np
import pandas as pd
import scanpy as sc

from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_CELL_TYPE_ANNOTATION_KEYS = (
    'cell_type',
    'celltype',
    'cell_type_name',
    'celltype_name',
    'annotation',
    'annotations',
    'ct_top',
    'ct',
)


def _sanitize_labels(labels):
    clean_labels = []
    for label in np.asarray(labels, dtype=object):
        if pd.isna(label):
            clean_labels.append('Unknown')
            continue

        label_text = str(label).strip()
        clean_labels.append(label_text if label_text else 'Unknown')

    return np.asarray(clean_labels, dtype=object)


def detect_cell_type_annotation_key(
    adata_list,
    candidate_keys=None,
    min_unique_labels=2,
):
    if candidate_keys is None:
        candidate_keys = DEFAULT_CELL_TYPE_ANNOTATION_KEYS

    for key in candidate_keys:
        if not all(key in adata.obs.columns for adata in adata_list):
            continue

        combined_labels = []
        valid = True
        for adata in adata_list:
            labels = _sanitize_labels(adata.obs[key].values)
            non_unknown = labels[labels != 'Unknown']
            if non_unknown.size == 0:
                valid = False
                break
            combined_labels.append(non_unknown)

        if not valid:
            continue

        unique_labels = np.unique(np.concatenate(combined_labels))
        if unique_labels.size >= min_unique_labels:
            return key

    return None


def encode_annotation_cell_types(adata_list, slice_names, annotation_key, verbose=True):
    global_names = sorted(
        {
            label
            for adata in adata_list
            for label in _sanitize_labels(adata.obs[annotation_key].values).tolist()
        }
    )
    name_to_id = {name: idx for idx, name in enumerate(global_names)}
    id_to_name = {idx: name for name, idx in name_to_id.items()}

    global_ids_by_slice = {}
    local_labels_by_slice = {}
    slice_local_to_global = {}

    for slice_name, adata in zip(slice_names, adata_list):
        labels = _sanitize_labels(adata.obs[annotation_key].values)
        global_ids = np.asarray([name_to_id[label] for label in labels], dtype=np.int64)

        global_ids_by_slice[slice_name] = global_ids
        local_labels_by_slice[slice_name] = labels

        for label in np.unique(labels):
            slice_local_to_global[(slice_name, label)] = name_to_id[label]

    if verbose:
        print(
            f'  Using provided cell-type annotation "{annotation_key}" '
            f'with {len(global_names)} global cell types'
        )

    return {
        'source': 'annotation',
        'annotation_key': annotation_key,
        'n_cell_types': len(global_names),
        'cell_type_names': np.asarray(global_names, dtype=object),
        'name_to_id': name_to_id,
        'id_to_name': id_to_name,
        'global_cell_type_ids_by_slice': global_ids_by_slice,
        'local_labels_by_slice': local_labels_by_slice,
        'slice_local_to_global': slice_local_to_global,
        'alignment_info': {
            'strategy': 'provided_annotation',
            'annotation_key': annotation_key,
        },
    }


def _resolve_feature_masks(adata_list, feature_masks):
    if feature_masks is None:
        return [None] * len(adata_list)

    if isinstance(feature_masks, (list, tuple)):
        return list(feature_masks)

    return [feature_masks] * len(adata_list)


def _subset_for_clustering(adata, feature_mask):
    if feature_mask is None:
        return adata.copy()

    return adata[:, feature_mask].copy()


def _cluster_centroids(adata, local_ids, n_local_types):
    X = adata.X
    if issparse(X):
        X = X.toarray()

    centroids = np.zeros((n_local_types, X.shape[1]), dtype=np.float32)
    for local_id in range(n_local_types):
        mask = local_ids == local_id
        if mask.sum() == 0:
            continue
        centroids[local_id] = X[mask].mean(axis=0)

    return centroids


def cluster_and_align_cell_types(
    adata_list,
    slice_names,
    feature_masks=None,
    n_pcs=30,
    n_neighbors=15,
    resolution=0.5,
    similarity_threshold=0.5,
    verbose=True,
):
    feature_masks = _resolve_feature_masks(adata_list, feature_masks)

    local_ids_by_slice = {}
    local_centroids_by_slice = {}
    local_type_counts = {}

    for slice_name, adata, feature_mask in zip(slice_names, adata_list, feature_masks):
        adata_hvg = _subset_for_clustering(adata, feature_mask)

        max_pcs = min(
            n_pcs,
            max(1, adata_hvg.n_vars),
            max(1, adata_hvg.n_obs - 1),
        )
        max_neighbors = min(n_neighbors, max(1, adata_hvg.n_obs - 1))

        sc.pp.pca(adata_hvg, n_comps=max_pcs)
        sc.pp.neighbors(adata_hvg, n_neighbors=max_neighbors, n_pcs=max_pcs)
        sc.tl.leiden(adata_hvg, resolution=resolution, key_added='local_cell_type')

        local_ids = adata_hvg.obs['local_cell_type'].astype(int).values
        n_local_types = int(local_ids.max()) + 1

        local_ids_by_slice[slice_name] = local_ids
        local_type_counts[slice_name] = n_local_types
        local_centroids_by_slice[slice_name] = _cluster_centroids(
            adata_hvg, local_ids, n_local_types
        )

        if verbose:
            counts = np.bincount(local_ids)
            print(
                f'  Slice {slice_name}: Leiden found {n_local_types} local clusters '
                f'(sizes: {counts.tolist()})'
            )

    slice_local_to_global = {}
    alignment_info = {
        'strategy': 'cluster_centroid_cosine',
        'reference_slice': slice_names[0],
        'similarity_threshold': similarity_threshold,
        'matches': {},
        'local_type_counts': local_type_counts,
    }

    reference_slice = slice_names[0]
    reference_centroids = local_centroids_by_slice[reference_slice]
    global_centroids = [reference_centroids[idx].copy() for idx in range(reference_centroids.shape[0])]
    global_counts = [1] * len(global_centroids)

    for local_id in range(local_type_counts[reference_slice]):
        slice_local_to_global[(reference_slice, local_id)] = local_id

    for slice_name in slice_names[1:]:
        current_centroids = local_centroids_by_slice[slice_name]
        sim_matrix = cosine_similarity(current_centroids, np.stack(global_centroids, axis=0))
        row_ind, col_ind = linear_sum_assignment(1.0 - sim_matrix)

        matched_rows = set()
        slice_matches = []

        for row_idx, global_idx in zip(row_ind.tolist(), col_ind.tolist()):
            similarity = float(sim_matrix[row_idx, global_idx])
            if similarity < similarity_threshold:
                continue

            slice_local_to_global[(slice_name, row_idx)] = global_idx
            matched_rows.add(row_idx)
            slice_matches.append(
                {
                    'local_cluster_id': int(row_idx),
                    'global_cell_type_id': int(global_idx),
                    'similarity': similarity,
                }
            )

            old_count = global_counts[global_idx]
            global_centroids[global_idx] = (
                (global_centroids[global_idx] * old_count) + current_centroids[row_idx]
            ) / (old_count + 1)
            global_counts[global_idx] = old_count + 1

        unmatched_local_ids = []
        for local_id in range(local_type_counts[slice_name]):
            if local_id in matched_rows:
                continue

            new_global_id = len(global_centroids)
            slice_local_to_global[(slice_name, local_id)] = new_global_id
            global_centroids.append(current_centroids[local_id].copy())
            global_counts.append(1)
            unmatched_local_ids.append(int(local_id))

        alignment_info['matches'][slice_name] = {
            'matched': slice_matches,
            'unmatched_local_cluster_ids': unmatched_local_ids,
        }

        if verbose:
            print(
                f'  Slice {slice_name}: matched {len(slice_matches)}/'
                f'{local_type_counts[slice_name]} local clusters into the global space'
            )

    n_cell_types = len(global_centroids)
    global_names = np.asarray(
        [f'global_cell_type_{idx}' for idx in range(n_cell_types)],
        dtype=object,
    )

    global_ids_by_slice = {}
    for slice_name in slice_names:
        local_ids = local_ids_by_slice[slice_name]
        global_ids_by_slice[slice_name] = np.asarray(
            [slice_local_to_global[(slice_name, int(local_id))] for local_id in local_ids],
            dtype=np.int64,
        )

    if verbose:
        print(f'  Cross-slice alignment complete: {n_cell_types} global cell types')

    return {
        'source': 'cluster_alignment',
        'annotation_key': None,
        'n_cell_types': n_cell_types,
        'cell_type_names': global_names,
        'name_to_id': {name: idx for idx, name in enumerate(global_names.tolist())},
        'id_to_name': {idx: name for idx, name in enumerate(global_names.tolist())},
        'global_cell_type_ids_by_slice': global_ids_by_slice,
        'local_labels_by_slice': local_ids_by_slice,
        'slice_local_to_global': slice_local_to_global,
        'alignment_info': alignment_info,
    }


def resolve_global_cell_types(
    adata_list,
    slice_names,
    feature_masks=None,
    annotation_key=None,
    candidate_annotation_keys=None,
    n_pcs=30,
    n_neighbors=15,
    resolution=0.5,
    similarity_threshold=0.5,
    verbose=True,
):
    detected_annotation_key = annotation_key
    if detected_annotation_key is None:
        detected_annotation_key = detect_cell_type_annotation_key(
            adata_list,
            candidate_keys=candidate_annotation_keys,
        )

    if detected_annotation_key is not None:
        return encode_annotation_cell_types(
            adata_list=adata_list,
            slice_names=slice_names,
            annotation_key=detected_annotation_key,
            verbose=verbose,
        )

    return cluster_and_align_cell_types(
        adata_list=adata_list,
        slice_names=slice_names,
        feature_masks=feature_masks,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        resolution=resolution,
        similarity_threshold=similarity_threshold,
        verbose=verbose,
    )
