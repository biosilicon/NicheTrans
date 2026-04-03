import math
import unittest

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model.nicheTrans import NicheTrans
from utils.moe_analysis import analyze_moe_routing, compute_expert_usage_metrics


class TinySpatialDataset(Dataset):
    def __init__(self, num_items=4, source_length=6, target_length=3, num_neighbors=8):
        self.num_items = num_items
        self.source_length = source_length
        self.target_length = target_length
        self.num_neighbors = num_neighbors

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        source = torch.full((self.source_length,), float(index + 1))
        target = torch.full((self.target_length,), float(index))
        neighbors = torch.full((self.num_neighbors, self.source_length), float(index + 2))
        return source, target, neighbors, f"sliceA/{index}_{index + 1}"


class MoeAnalysisTests(unittest.TestCase):
    def test_forward_is_backward_compatible_and_exposes_moe_info(self):
        torch.manual_seed(0)
        model = NicheTrans(
            source_length=6,
            target_length=3,
            noise_rate=0.0,
            dropout_rate=0.0,
            num_experts=4,
            moe_gate_hidden_dim=8,
            moe_router_temperature_enable=True,
            moe_balance_loss_enable=True,
            moe_router_entropy_penalty_enable=True,
        )
        model.eval()
        model.ffn_omic.set_current_epoch(12)

        source = torch.randn(5, 6)
        neighbors = torch.randn(5, 8, 6)

        plain_output = model(source, neighbors)
        analysis_output = model(source, neighbors, return_moe_info=True)

        self.assertEqual(plain_output.shape, (5, 3))
        self.assertTrue(torch.allclose(plain_output, analysis_output["predictions"], atol=1e-6))

        moe_info = analysis_output["moe_info"]
        self.assertEqual(moe_info["gate_weights"].shape, (5, 9, 4))
        self.assertEqual(moe_info["center_gate_weights"].shape, (5, 4))
        self.assertEqual(moe_info["center_top1_expert"].shape, (5,))
        self.assertEqual(moe_info["center_gate_margin"].shape, (5,))
        self.assertTrue(
            torch.allclose(
                moe_info["center_gate_weights"].sum(dim=-1),
                torch.ones(5),
                atol=1e-6,
            )
        )
        self.assertAlmostEqual(float(moe_info["router_temperature"]), 0.5, places=6)
        self.assertGreaterEqual(float(moe_info["balance_loss"].detach()), 0.0)
        self.assertGreaterEqual(float(moe_info["router_entropy_penalty"].detach()), 0.0)
        self.assertGreaterEqual(float(moe_info["expert_output_cosine_std"].detach()), 0.0)

    def test_analysis_utilities_collect_metrics_and_regions(self):
        torch.manual_seed(1)
        model = NicheTrans(
            source_length=6,
            target_length=3,
            noise_rate=0.0,
            dropout_rate=0.0,
            num_experts=3,
            moe_gate_hidden_dim=8,
            moe_router_temperature_enable=True,
            moe_balance_loss_enable=True,
            moe_router_entropy_penalty_enable=True,
        )
        model.ffn_omic.set_current_epoch(6)
        loader = DataLoader(TinySpatialDataset(), batch_size=2, shuffle=False)

        results = analyze_moe_routing(
            model=model,
            dataloader=loader,
            device=torch.device("cpu"),
            include_predictions=False,
            include_targets=True,
            add_spatial_regions=True,
        )

        activation_frame = results["activation_frame"]
        self.assertEqual(len(activation_frame), 4)
        self.assertTrue(
            {"expert_0", "expert_1", "expert_2", "slice_id", "x", "y"}.issubset(activation_frame.columns)
        )
        self.assertTrue((activation_frame["slice_id"] == "sliceA").all())
        self.assertTrue(activation_frame["x"].notna().all())
        self.assertTrue(activation_frame["y"].notna().all())

        overall = results["overall"]
        self.assertEqual(overall["num_center_spots"], 4)
        self.assertEqual(overall["num_experts"], 3)
        self.assertGreaterEqual(overall["usage_entropy_normalised"], 0.0)
        self.assertLessEqual(overall["usage_entropy_normalised"], 1.0)
        self.assertIn("router_temperature", overall)
        self.assertIn("balance_loss", overall)
        self.assertIn("router_entropy_penalty", overall)
        self.assertIn("mean_gate_margin", overall)
        self.assertIn("expert_output_cosine_mean", overall)

        self.assertFalse(results["batch_summary"].empty)
        self.assertFalse(results["slice_summary"].empty)
        self.assertFalse(results["region_summary"].empty)

    def test_compute_expert_usage_metrics_on_synthetic_frame(self):
        frame = pd.DataFrame(
            {
                "sample_id": ["s0", "s1", "s2", "s3"],
                "slice_id": ["a", "a", "b", "b"],
                "batch_index": [0, 0, 1, 1],
                "x": [0.0, 1.0, 0.0, 1.0],
                "y": [0.0, 0.0, 1.0, 1.0],
                "expert_0": [0.9, 0.8, 0.2, 0.1],
                "expert_1": [0.1, 0.2, 0.8, 0.9],
            }
        )

        metrics = compute_expert_usage_metrics(frame, add_spatial_regions=True)

        expert_summary = metrics["expert_summary"]
        self.assertEqual(expert_summary["average_activation_weight"].tolist(), [0.5, 0.5])
        self.assertEqual(expert_summary["top1_selection_frequency"].tolist(), [0.5, 0.5])
        self.assertTrue(math.isclose(metrics["overall"]["effective_expert_count"], 2.0, rel_tol=1e-4))
        self.assertAlmostEqual(metrics["overall"]["mean_gate_margin"], 0.7, places=6)
        self.assertFalse(metrics["slice_differences"].empty)
        self.assertFalse(metrics["region_differences"].empty)

    def test_router_temperature_schedule_respects_epoch_bands(self):
        model = NicheTrans(
            source_length=6,
            target_length=3,
            noise_rate=0.0,
            dropout_rate=0.0,
            num_experts=2,
            moe_gate_hidden_dim=8,
            moe_router_temperature_enable=True,
        )

        model.ffn_omic.set_current_epoch(1)
        self.assertAlmostEqual(model.ffn_omic.gate.get_router_temperature(), 1.0, places=6)
        model.ffn_omic.set_current_epoch(7)
        self.assertAlmostEqual(model.ffn_omic.gate.get_router_temperature(), 0.7, places=6)
        model.ffn_omic.set_current_epoch(11)
        self.assertAlmostEqual(model.ffn_omic.gate.get_router_temperature(), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
