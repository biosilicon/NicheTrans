import numpy as np


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
