import math
import unittest

import numpy as np
import torch

from datasets.graph_utils import build_graph_dataloader, build_slice_graph
from model.whole_slice_graph_transformer import WholeSliceGraphTransformer
from utils.utils_training_graph import train_epoch


class WholeSliceGraphTransformerSmokeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)

        num_nodes = 12
        source_dim = 16
        target_dim = 5

        x = np.random.randn(num_nodes, source_dim).astype(np.float32)
        y = np.random.randn(num_nodes, target_dim).astype(np.float32)
        pos = np.random.randn(num_nodes, 2).astype(np.float32)

        self.graph = build_slice_graph(
            node_features=x,
            node_targets=y,
            coordinates=pos,
            split="train",
            k=4,
            val_ratio=0.25,
            mask_seed=7,
            cell_type=np.array(["a", "b", "a", "c", "b", "a", "c", "c", "b", "a", "b", "c"], dtype=object),
            cell_type_vocabulary=np.array(["a", "b", "c"], dtype=object),
            sample_id="synthetic_slice",
            slice_name="synthetic_slice",
        )

    def test_graph_construction_shapes(self):
        self.assertEqual(tuple(self.graph.x.shape), (12, 16))
        self.assertEqual(tuple(self.graph.y.shape), (12, 5))
        self.assertEqual(tuple(self.graph.pos.shape), (12, 2))
        self.assertEqual(self.graph.edge_index.shape[0], 2)
        self.assertEqual(self.graph.edge_attr.shape[1], 3)
        self.assertEqual(self.graph.edge_index.shape[1], self.graph.edge_attr.shape[0])
        self.assertEqual(int(self.graph.train_mask.sum() + self.graph.val_mask.sum() + self.graph.test_mask.sum()), 12)

    def test_forward_and_backward(self):
        model = WholeSliceGraphTransformer(
            source_dim=16,
            target_dim=5,
            hidden_dim=32,
            num_layers=2,
            heads=4,
            use_cell_type=True,
            num_cell_types=3,
        )

        pred = model(self.graph)
        self.assertEqual(tuple(pred.shape), (12, 5))

        loss = torch.nn.functional.mse_loss(pred[self.graph.train_mask], self.graph.y[self.graph.train_mask])
        loss.backward()

        grad_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_norm += float(parameter.grad.norm().item())

        self.assertTrue(math.isfinite(loss.item()))
        self.assertGreater(grad_norm, 0.0)

    def test_training_wrapper_runs(self):
        loader = build_graph_dataloader([self.graph], batch_size=1, shuffle=False)
        model = WholeSliceGraphTransformer(
            source_dim=16,
            target_dim=5,
            hidden_dim=32,
            num_layers=2,
            heads=4,
            use_cell_type=True,
            num_cell_types=3,
        )
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_loss = train_epoch(model, criterion, optimizer, loader, split="train", device=torch.device("cpu"))
        self.assertTrue(math.isfinite(train_loss))


if __name__ == "__main__":
    unittest.main()
