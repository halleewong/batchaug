import math

import torch
import monai.transforms

import batchaug


class TestDivisiblePadBasic:
    """Basic padding functionality."""

    def test_end_method(self, device):
        vol = torch.rand(2, 1, 13, 17, 21, device=device)
        t = batchaug.DivisiblePad(k=8, method="end")
        out = t(vol)
        assert out.shape[2] % 8 == 0
        assert out.shape[3] % 8 == 0
        assert out.shape[4] % 8 == 0
        assert out.shape == (2, 1, 16, 24, 24)
        # Original data preserved at start
        assert torch.equal(out[:, :, :13, :17, :21], vol)

    def test_symmetric_method(self, device):
        vol = torch.rand(2, 1, 13, 17, 21, device=device)
        t = batchaug.DivisiblePad(k=8, method="symmetric")
        out = t(vol)
        assert out.shape[2] % 8 == 0
        assert out.shape[3] % 8 == 0
        assert out.shape[4] % 8 == 0
        assert out.shape == (2, 1, 16, 24, 24)

    def test_already_divisible(self, vol, device):
        t = batchaug.DivisiblePad(k=8)
        out = t(vol)
        # vol is 16x16x16, already divisible by 8
        assert torch.equal(out, vol)

    def test_per_axis_k(self, device):
        vol = torch.rand(1, 1, 10, 10, 10, device=device)
        t = batchaug.DivisiblePad(k=(8, 16, 32), method="end")
        out = t(vol)
        assert out.shape[2] % 8 == 0  # 10 → 16
        assert out.shape[3] % 16 == 0  # 10 → 16
        assert out.shape[4] % 32 == 0  # 10 → 32
        assert out.shape == (1, 1, 16, 16, 32)


class TestDivisiblePadMatchesMONAI:
    """B=1: batchaug matches MONAI's DivisiblePad."""

    def test_match_monai_symmetric(self, device):
        torch.manual_seed(0)
        input_4d = torch.rand(1, 13, 17, 21, device=device)
        input_5d = input_4d.unsqueeze(0)  # (1, 1, 13, 17, 21)

        monai_t = monai.transforms.DivisiblePad(k=8, mode="constant", method="symmetric")
        monai_out = monai_t(input_4d)  # (1, H', W', D')

        ba_t = batchaug.DivisiblePad(k=8, mode="constant", method="symmetric")
        ba_out = ba_t(input_5d)  # (1, 1, H', W', D')

        assert ba_out.shape[2:] == monai_out.shape[1:]
        assert torch.equal(ba_out[0], monai_out)

    def test_match_monai_end(self, device):
        torch.manual_seed(0)
        input_4d = torch.rand(1, 13, 17, 21, device=device)
        input_5d = input_4d.unsqueeze(0)

        monai_t = monai.transforms.DivisiblePad(k=8, mode="constant", method="end")
        monai_out = monai_t(input_4d)

        ba_t = batchaug.DivisiblePad(k=8, mode="constant", method="end")
        ba_out = ba_t(input_5d)

        assert ba_out.shape[2:] == monai_out.shape[1:]
        assert torch.equal(ba_out[0], monai_out)

    def test_match_monai_per_axis(self, device):
        torch.manual_seed(0)
        input_4d = torch.rand(2, 10, 10, 10, device=device)
        input_5d = input_4d.unsqueeze(0)

        monai_t = monai.transforms.DivisiblePad(k=(8, 16, 32), mode="constant", method="symmetric")
        monai_out = monai_t(input_4d)

        ba_t = batchaug.DivisiblePad(k=(8, 16, 32), mode="constant", method="symmetric")
        ba_out = ba_t(input_5d)

        assert ba_out.shape[2:] == monai_out.shape[1:]
        assert torch.equal(ba_out[0], monai_out)


class TestDivisiblePadBatched:
    """All batch elements are padded identically."""

    def test_batched_uniform(self, device):
        vol = torch.rand(4, 2, 13, 17, 21, device=device)
        t = batchaug.DivisiblePad(k=8)
        out = t(vol)
        # All elements should have same shape (they always do since it's one tensor)
        assert out.shape == (4, 2, 16, 24, 24)


class TestDivisiblePadModes:
    """Different padding modes."""

    def test_reflect_mode(self, device):
        vol = torch.rand(1, 1, 13, 13, 13, device=device)
        t = batchaug.DivisiblePad(k=16, mode="reflect")
        out = t(vol)
        assert out.shape == (1, 1, 16, 16, 16)
        assert not torch.isnan(out).any()

    def test_replicate_mode(self, device):
        vol = torch.rand(1, 1, 13, 13, 13, device=device)
        t = batchaug.DivisiblePad(k=16, mode="replicate")
        out = t(vol)
        assert out.shape == (1, 1, 16, 16, 16)
        assert not torch.isnan(out).any()

    def test_circular_mode(self, device):
        vol = torch.rand(1, 1, 13, 13, 13, device=device)
        t = batchaug.DivisiblePad(k=16, mode="circular")
        out = t(vol)
        assert out.shape == (1, 1, 16, 16, 16)
        assert not torch.isnan(out).any()


class TestDivisiblePadd:
    """Dictionary wrapper tests."""

    def test_dict_both_keys(self, device):
        vol = torch.rand(2, 1, 13, 17, 21, device=device)
        seg = torch.randint(0, 5, (2, 1, 13, 17, 21), device=device).float()
        t = batchaug.DivisiblePadd(keys=["vol", "seg"], k=8)
        data = {"vol": vol, "seg": seg}
        result = t(data)
        assert result["vol"].shape == result["seg"].shape
        assert result["vol"].shape[2] % 8 == 0
        assert result["vol"].shape[3] % 8 == 0
        assert result["vol"].shape[4] % 8 == 0


class TestDivisiblePadDtype:
    """dtype preservation."""

    def test_bfloat16(self, device):
        vol = torch.rand(2, 1, 13, 13, 13, device=device, dtype=torch.bfloat16)
        t = batchaug.DivisiblePad(k=8)
        out = t(vol)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()
