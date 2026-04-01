from model.attention import apply_neighbor_mask_to_graph_meta, move_graph_meta_to_device


def get_batch_graph_meta(loader_dataset, samples, device=None, neighbor_keep_mask=None):
    if not hasattr(loader_dataset, 'get_graph_meta'):
        return None

    graph_meta = loader_dataset.get_graph_meta(samples)
    if graph_meta is None:
        return None

    if device is not None:
        graph_meta = move_graph_meta_to_device(graph_meta, device)

    if neighbor_keep_mask is not None:
        graph_meta = apply_neighbor_mask_to_graph_meta(graph_meta, neighbor_keep_mask)

    return graph_meta
