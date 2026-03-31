from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity

##################################################################
#清洗数据
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
        # Normalize missing or empty labels to a shared placeholder.
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

    # Only consider keys that exist in every slice.
    for key in candidate_keys:
        if not all(key in adata.obs.columns for adata in adata_list):
            continue

        combined_labels = []
        valid = True
        for adata in adata_list:
            labels = _sanitize_labels(adata.obs[key].values)
            non_unknown = labels[labels != 'Unknown']
            # Skip this key if any slice has no usable labels.
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

####################################################################


def encode_annotation_cell_types(
    adata_list,
    slice_names,
    annotation_key,
    feature_masks=None,
    reference_slice=None,
    testing_slides=None,
    n_pcs=30,
    n_neighbors=15,
    visualize=False,
    visualization_dir=None,
    visualization_dpi=150,
    verbose=True,
):
    resolved_reference_slice = _resolve_reference_slice(
        slice_names,
        reference_slice=reference_slice,
        testing_slides=testing_slides,
    )
    labels_by_slice = {}
    for slice_name, adata in zip(slice_names, adata_list):
        labels_by_slice[slice_name] = _sanitize_labels(adata.obs[annotation_key].values)

    # Anchor annotation IDs to the reference slice, then append any labels that
    # appear only in the remaining slices to preserve a shared global space.
    reference_names = sorted(set(labels_by_slice[resolved_reference_slice].tolist()))
    remaining_names = sorted(
        {
            label
            for slice_name, labels in labels_by_slice.items()
            if slice_name != resolved_reference_slice
            for label in labels.tolist()
            if label not in reference_names
        }
    )
    global_names = reference_names + remaining_names
    name_to_id = {name: idx for idx, name in enumerate(global_names)}
    id_to_name = {idx: name for name, idx in name_to_id.items()}

    global_ids_by_slice = {}
    local_labels_by_slice = {}
    slice_local_to_global = {}

    for slice_name, adata in zip(slice_names, adata_list):
        labels = labels_by_slice[slice_name]
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

    visualization_paths = {}
    if visualize:
        feature_masks = _resolve_feature_masks(adata_list, feature_masks)
        resolved_visualization_dir = visualization_dir or 'scanpy_cell_type_visualizations'
        visualization_paths = _generate_visualization_outputs(
            adata_list=adata_list,
            slice_names=slice_names,
            feature_masks=feature_masks,
            local_labels_by_slice=local_labels_by_slice,
            global_ids_by_slice=global_ids_by_slice,
            global_names=np.asarray(global_names, dtype=object),
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            visualization_dir=resolved_visualization_dir,
            visualization_dpi=visualization_dpi,
            verbose=verbose,
        )

        if verbose:
            print(f'  Scanpy visualizations saved to: {resolved_visualization_dir}')

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
            'reference_slice': resolved_reference_slice,
        },
        'visualization_paths': visualization_paths,
    }


def _resolve_feature_masks(adata_list, feature_masks):
    if feature_masks is None:
        return [None] * len(adata_list)

    # Accept either one mask per slice or one shared mask.
    if isinstance(feature_masks, (list, tuple)):
        return list(feature_masks)

    return [feature_masks] * len(adata_list)


def _normalize_slice_selection(slice_selection):
    if slice_selection is None:
        return []

    if isinstance(slice_selection, str):
        return [slice_selection]

    return list(slice_selection)


def _resolve_reference_slice(slice_names, reference_slice=None, testing_slides=None):
    slice_names = list(slice_names)
    if len(slice_names) == 0:
        raise ValueError('slice_names must contain at least one slice')

    if reference_slice is not None:
        if reference_slice not in slice_names:
            raise ValueError(
                f'reference_slice "{reference_slice}" was not found in slice_names'
            )
        return reference_slice

    testing_slide_names = _normalize_slice_selection(testing_slides)
    if testing_slide_names:
        matching_testing_slides = [
            slice_name for slice_name in testing_slide_names if slice_name in slice_names
        ]
        if not matching_testing_slides:
            raise ValueError(
                'None of the provided testing_slides were found in slice_names'
            )
        return matching_testing_slides[0]

    return slice_names[0]


def _get_alignment_slice_order(slice_names, reference_slice):
    slice_names = list(slice_names)
    return [reference_slice] + [
        slice_name for slice_name in slice_names if slice_name != reference_slice
    ]


def _subset_for_clustering(adata, feature_mask):
    if feature_mask is None:
        return adata.copy()

    return adata[:, feature_mask].copy()


def _cluster_centroids(adata, local_ids, n_local_types):
    """Compute one centroid per local cluster from the expression matrix.

    Each centroid is the mean feature vector of all cells assigned to the
    corresponding local cluster. The returned array has shape
    ``(n_local_types, n_features)`` and is later used for cross-slice
    similarity matching.
    """
    X = adata.X
    if issparse(X):
        X = X.toarray()

    centroids = np.zeros((n_local_types, X.shape[1]), dtype=np.float32)
    for local_id in range(n_local_types):
        mask = local_ids == local_id
        if mask.sum() == 0:
            continue
        # Represent each local cluster by its mean expression vector.
        centroids[local_id] = X[mask].mean(axis=0)

    return centroids


def _compute_embedding(adata, n_pcs, n_neighbors):
    # Keep Scanpy parameters within the slice-specific data limits.
    max_pcs = min(
        n_pcs,
        max(1, adata.n_vars),
        max(1, adata.n_obs - 1),
    )
    max_neighbors = min(n_neighbors, max(1, adata.n_obs - 1))

    sc.pp.pca(adata, n_comps=max_pcs)
    sc.pp.neighbors(adata, n_neighbors=max_neighbors, n_pcs=max_pcs)

    return max_pcs, max_neighbors


def _ensure_spatial_basis(adata):
    if 'spatial' in adata.obsm:
        return True

    if {'array_row', 'array_col'}.issubset(adata.obs.columns):
        adata.obsm['spatial'] = adata.obs[['array_col', 'array_row']].to_numpy(dtype=np.float32)
        return True

    return False


def _save_cluster_visualizations(
    adata,
    slice_name,
    output_dir,
    color_keys,
    dpi=150,
    verbose=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    try:
        if 'X_umap' not in adata.obsm:
            sc.tl.umap(adata)

        umap_path = output_dir / f'{slice_name}_umap_cell_types.png'
        sc.pl.umap(
            adata,
            color=color_keys,
            title=[f'{slice_name}: {key}' for key in color_keys],
            frameon=False,
            show=False,
        )
        plt.savefig(umap_path, bbox_inches='tight', dpi=dpi)
        plt.close('all')
        plot_paths['umap'] = str(umap_path)

        if _ensure_spatial_basis(adata):
            spatial_path = output_dir / f'{slice_name}_spatial_cell_types.png'
            sc.pl.embedding(
                adata,
                basis='spatial',
                color=color_keys,
                title=[f'{slice_name} spatial: {key}' for key in color_keys],
                frameon=False,
                show=False,
            )
            plt.savefig(spatial_path, bbox_inches='tight', dpi=dpi)
            plt.close('all')
            plot_paths['spatial'] = str(spatial_path)
        elif verbose:
            print(f'  Slice {slice_name}: skipped spatial plot because no spatial coordinates were found')
    except Exception as exc:
        plt.close('all')
        if verbose:
            print(f'  Slice {slice_name}: failed to generate Scanpy plots ({exc})')

    return plot_paths


def _generate_visualization_outputs(
    adata_list,
    slice_names,
    feature_masks,
    local_labels_by_slice,
    global_ids_by_slice,
    global_names,
    n_pcs,
    n_neighbors,
    visualization_dir,
    visualization_dpi,
    verbose=True,
    prepared_adata_by_slice=None,
):
    visualization_paths = {}

    for slice_name, adata, feature_mask in zip(slice_names, adata_list, feature_masks):
        adata_plot = None
        if prepared_adata_by_slice is not None:
            adata_plot = prepared_adata_by_slice.get(slice_name)

        if adata_plot is None:
            adata_plot = _subset_for_clustering(adata, feature_mask)
            _compute_embedding(adata_plot, n_pcs=n_pcs, n_neighbors=n_neighbors)

        if 'X_umap' not in adata_plot.obsm:
            sc.tl.umap(adata_plot)

        local_labels = np.asarray(local_labels_by_slice[slice_name], dtype=object)
        adata_plot.obs['local_cell_type'] = pd.Categorical(local_labels.astype(str))
        adata_plot.obs['global_cell_type'] = pd.Categorical(global_names[global_ids_by_slice[slice_name]])

        visualization_paths[slice_name] = _save_cluster_visualizations(
            adata=adata_plot,
            slice_name=slice_name,
            output_dir=visualization_dir,
            color_keys=['local_cell_type', 'global_cell_type'],
            dpi=visualization_dpi,
            verbose=verbose,
        )

    return visualization_paths


def cluster_and_align_cell_types(
    adata_list,
    slice_names,
    feature_masks=None,
    reference_slice=None,
    testing_slides=None,
    n_pcs=30,
    n_neighbors=15,
    resolution=0.5,
    similarity_threshold=0.5,
    visualize=False,
    visualization_dir=None,
    visualization_dpi=150,
    verbose=True,
):
    """Cluster each slice independently and align clusters across slices.

    Parameters
    ----------
    adata_list : list
        List of AnnData objects, where each item corresponds to one slice.
    slice_names : list
        Slice names aligned one-to-one with ``adata_list``.
    feature_masks : None, array-like, or list of array-like, optional
        Feature-selection mask(s) used before clustering. If a single mask is
        provided, it is shared across all slices; if a list is provided, each
        slice uses its own mask.
    reference_slice : str, optional
        Slice name whose local clusters seed the global cell-type ID space.
        When omitted, ``testing_slides`` is used if provided; otherwise the
        first entry in ``slice_names`` is used for backward compatibility.
    testing_slides : str or sequence of str, optional
        Testing slice name(s). When ``reference_slice`` is not provided, the
        first testing slice present in ``slice_names`` becomes the reference
        slice for the global cell-type space.
    n_pcs : int, optional
        Target number of principal components used for PCA and neighborhood
        graph construction. The effective value is clipped to each slice's
        available number of observations and features.
    n_neighbors : int, optional
        Number of neighbors used to build the Scanpy neighborhood graph for
        Leiden clustering. The effective value is clipped per slice.
    resolution : float, optional
        Resolution parameter passed to Leiden clustering. Larger values usually
        yield more local clusters.
    similarity_threshold : float, optional
        Minimum cosine similarity required to match a local cluster to an
        existing global cell type across slices.
    visualize : bool, optional
        If True, generate Scanpy UMAP plots for each slice after clustering and
        alignment. Spatial plots are also generated when spatial coordinates are
        available on the AnnData object.
    visualization_dir : str or path-like, optional
        Directory used to save Scanpy figures when ``visualize=True``. If not
        provided, figures are written to ``scanpy_cell_type_visualizations``.
    visualization_dpi : int, optional
        DPI used when saving Scanpy figures.
    verbose : bool, optional
        If True, print clustering and alignment progress information.

    Returns
    -------
    dict
        A dictionary containing global cell-type ids for each slice, the local
        to global mapping, generated global names, and alignment metadata.
    """
    feature_masks = _resolve_feature_masks(adata_list, feature_masks)
    reference_slice = _resolve_reference_slice(
        slice_names,
        reference_slice=reference_slice,
        testing_slides=testing_slides,
    )
    alignment_slice_order = _get_alignment_slice_order(slice_names, reference_slice)

    local_ids_by_slice = {}
    local_centroids_by_slice = {}
    local_type_counts = {}
    clustered_adata_by_slice = {}

    for slice_name, adata, feature_mask in zip(slice_names, adata_list, feature_masks):
        adata_hvg = _subset_for_clustering(adata, feature_mask)

        _compute_embedding(adata_hvg, n_pcs=n_pcs, n_neighbors=n_neighbors)
        sc.tl.leiden(adata_hvg, resolution=resolution, key_added='local_cell_type')

        local_ids = adata_hvg.obs['local_cell_type'].astype(int).values
        n_local_types = int(local_ids.max()) + 1

        local_ids_by_slice[slice_name] = local_ids
        local_type_counts[slice_name] = n_local_types
        local_centroids_by_slice[slice_name] = _cluster_centroids(
            adata_hvg, local_ids, n_local_types
        )
        clustered_adata_by_slice[slice_name] = adata_hvg

        if verbose:
            counts = np.bincount(local_ids)
            print(
                f'  Slice {slice_name}: Leiden found {n_local_types} local clusters '
                f'(sizes: {counts.tolist()})'
            )

    slice_local_to_global = {}
    alignment_info = {
        'strategy': 'cluster_centroid_cosine',
        'reference_slice': reference_slice,
        'similarity_threshold': similarity_threshold,
        'matches': {},
        'local_type_counts': local_type_counts,
    }

    reference_centroids = local_centroids_by_slice[reference_slice]
    # Seed the global cell-type space from the resolved reference slice.
    global_centroids = [reference_centroids[idx].copy() for idx in range(reference_centroids.shape[0])]
    global_counts = [1] * len(global_centroids)

    for local_id in range(local_type_counts[reference_slice]):
        slice_local_to_global[(reference_slice, local_id)] = local_id

    for slice_name in alignment_slice_order[1:]:
        current_centroids = local_centroids_by_slice[slice_name]
        sim_matrix = cosine_similarity(current_centroids, np.stack(global_centroids, axis=0))
        # Find the best one-to-one local/global matches, then filter weak ones.
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
            # Update the global centroid with an incremental mean after a match.
            global_centroids[global_idx] = (
                (global_centroids[global_idx] * old_count) + current_centroids[row_idx]
            ) / (old_count + 1)
            global_counts[global_idx] = old_count + 1

        unmatched_local_ids = []
        for local_id in range(local_type_counts[slice_name]):
            if local_id in matched_rows:
                continue

            # Unmatched local clusters become new global cell types.
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

    visualization_paths = {}
    if visualize:
        resolved_visualization_dir = visualization_dir or 'scanpy_cell_type_visualizations'
        visualization_paths = _generate_visualization_outputs(
            adata_list=adata_list,
            slice_names=slice_names,
            feature_masks=feature_masks,
            local_labels_by_slice=local_ids_by_slice,
            global_ids_by_slice=global_ids_by_slice,
            global_names=global_names,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            visualization_dir=resolved_visualization_dir,
            visualization_dpi=visualization_dpi,
            verbose=verbose,
            prepared_adata_by_slice=clustered_adata_by_slice,
        )

        if verbose:
            print(f'  Scanpy visualizations saved to: {resolved_visualization_dir}')

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
        'visualization_paths': visualization_paths,
    }


def resolve_global_cell_types(
    adata_list,
    slice_names,
    feature_masks=None,
    annotation_key=None,
    candidate_annotation_keys=None,
    reference_slice=None,
    testing_slides=None,
    n_pcs=30,
    n_neighbors=15,
    resolution=0.5,
    similarity_threshold=0.5,
    visualize=False,
    visualization_dir=None,
    visualization_dpi=150,
    verbose=True,
):
    detected_annotation_key = annotation_key
    if detected_annotation_key is None:
        detected_annotation_key = detect_cell_type_annotation_key(
            adata_list,
            candidate_keys=candidate_annotation_keys,
        )

    # Prefer provided annotations, and fall back to clustering only when needed.
    if detected_annotation_key is not None:
        return encode_annotation_cell_types(
            adata_list=adata_list,
            slice_names=slice_names,
            annotation_key=detected_annotation_key,
            feature_masks=feature_masks,
            reference_slice=reference_slice,
            testing_slides=testing_slides,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            visualize=visualize,
            visualization_dir=visualization_dir,
            visualization_dpi=visualization_dpi,
            verbose=verbose,
        )

    return cluster_and_align_cell_types(
        adata_list=adata_list,
        slice_names=slice_names,
        feature_masks=feature_masks,
        reference_slice=reference_slice,
        testing_slides=testing_slides,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        resolution=resolution,
        similarity_threshold=similarity_threshold,
        visualize=visualize,
        visualization_dir=visualization_dir,
        visualization_dpi=visualization_dpi,
        verbose=verbose,
    )
