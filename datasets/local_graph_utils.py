from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn.neighbors


def build_local_graph_metadata(center_coord, neighbor_coords, hop_ids, valid_mask):
    center_coord = np.asarray(center_coord, dtype=np.float32)
    neighbor_coords = np.asarray(neighbor_coords, dtype=np.float32)
    hop_ids = np.asarray(hop_ids, dtype=np.int64)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    coords = np.zeros((neighbor_coords.shape[0] + 1, 2), dtype=np.float32)
    coords[1:] = neighbor_coords - center_coord[None, :]
    coords[1:][~valid_mask] = 0

    full_valid_mask = np.zeros((neighbor_coords.shape[0] + 1,), dtype=bool)
    full_valid_mask[0] = True
    full_valid_mask[1:] = valid_mask

    full_hop_ids = np.zeros((neighbor_coords.shape[0] + 1,), dtype=np.int64)
    full_hop_ids[1:] = hop_ids

    return {
        'coords': coords,
        'hop_ids': full_hop_ids,
        'valid_mask': full_valid_mask,
    }


def build_spatial_neighbor_dict(
    coords,
    index_labels,
    node_names,
    rad_cutoff=None,
    k_cutoff=None,
    model='Radius',
    verbose=True,
    adata=None,
):
    assert model in ['Radius', 'KNN']
    if verbose:
        print('------Calculating spatial graph...')

    coords = np.asarray(coords, dtype=np.float32)
    index_labels = np.asarray(index_labels)
    node_names = np.asarray(node_names)
    coor = pd.DataFrame(coords, index=index_labels, columns=['imagerow', 'imagecol'])

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        knn_list = [
            pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it]))
            for it in range(indices.shape[0])
        ]
    else:
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        knn_list = [
            pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :]))
            for it in range(indices.shape[0])
        ]

    knn_df = pd.concat(knn_list)
    knn_df.columns = ['Cell1', 'Cell2', 'Distance']

    spatial_net = knn_df.loc[knn_df['Distance'] > 0, :].copy()
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index)))
    spatial_net['Cell1'] = spatial_net['Cell1'].map(id_cell_trans)
    spatial_net['Cell2'] = spatial_net['Cell2'].map(id_cell_trans)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (spatial_net.shape[0], coor.shape[0]))
        print('%.4f neighbors per cell on average.' % (spatial_net.shape[0] / coor.shape[0]))

    if adata is not None:
        adata.uns['Spatial_Net'] = spatial_net

    node_name_map = dict(zip(index_labels, node_names))
    temp_dic = defaultdict(list)
    for _, row in spatial_net.iterrows():
        temp_dic[node_name_map[row['Cell1']]].append(node_name_map[row['Cell2']])

    return temp_dic
