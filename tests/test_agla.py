"""Unit tests for the AGLA module (run: python -m unittest discover -s tests)."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vggt_al.attention import AGLA, AGLABlock, WindowSelfAttention  # noqa: E402
from vggt_al.model import VGGTAL, VGGTALConfig  # noqa: E402


class TestAGLA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.b, self.v, self.h, self.w, self.c = 2, 6, 12, 16, 32
        self.x = torch.randn(self.b, self.v * self.h * self.w, self.c)

    def test_gate_sums_to_one(self):
        attn = AGLA(self.c, num_heads=4, global_downsample=4, window_size=8)
        out, gate = attn(self.x, (self.h, self.w), self.v, return_gate=True)
        self.assertEqual(tuple(out.shape), tuple(self.x.shape))
        self.assertEqual(tuple(gate.shape), (self.b, self.v, 2, self.h, self.w))
        self.assertTrue(torch.allclose(gate.sum(2), torch.ones_like(gate[:, :, 0]), atol=1e-6))

    def test_all_modes_and_scopes(self):
        for mode in ("gated", "concat", "global_only", "local_only"):
            for scope in ("intra", "cross"):
                attn = AGLA(self.c, num_heads=4, global_downsample=4,
                            window_size=4, mode=mode)
                out = attn(self.x, (self.h, self.w), self.v, scope=scope)
                self.assertEqual(tuple(out.shape), tuple(self.x.shape))
                self.assertTrue(torch.isfinite(out).all())

    def test_window_padding(self):
        win = WindowSelfAttention(self.c, num_heads=4, window_size=8)
        x = torch.randn(1, 5 * 7, self.c)
        y = win(x, 5, 7)
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_block_residual(self):
        block = AGLABlock(self.c, num_heads=4, global_downsample=4, window_size=4)
        y = block(self.x, (self.h, self.w), self.v, scope="cross")
        self.assertTrue(torch.isfinite(y).all())
        y.sum().backward()
        self.assertTrue(any(p.grad is not None for p in block.parameters()))

    def test_model_output_shapes(self):
        cfg = VGGTALConfig(img_size=(32, 48), patch_size=8, embed_dim=32,
                           depth=2, num_heads=4, num_views=3,
                           global_downsample=2, window_size=4)
        model = VGGTAL(cfg)
        images = torch.rand(1, 3, 3, 32, 48)
        out = model(images)
        self.assertEqual(tuple(out["depth"].shape), (1, 3, 1, 32, 48))
        self.assertEqual(tuple(out["cameras"].shape), (1, 3, 4, 4))
        self.assertEqual(tuple(out["points"].shape), (1, 3, 32, 48, 3))


if __name__ == "__main__":
    unittest.main()
