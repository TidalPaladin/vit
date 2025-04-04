import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from vit.tokens import apply_mask, create_mask, generate_non_overlapping_mask, mask_is_ragged, unapply_mask


@pytest.mark.parametrize(
    "mask, exp",
    [
        (([True, True], [True, True]), False),
        (([True, False], [True, True]), True),
        (([False, True], [True, True]), True),
        (([False, False], [True, True]), True),
        (([False, False], [False, False]), False),
    ],
)
def test_is_ragged(mask, exp):
    assert mask_is_ragged(torch.tensor(mask, dtype=torch.bool)) == exp


@pytest.mark.parametrize(
    "mask, fill_value, padding_value, exp",
    [
        # Non-ragged
        (([True, True], [True, True]), None, 0, torch.tensor([[0, 1], [0, 1]])),
        (([True, False], [False, True]), None, 0, torch.tensor([[0], [1]])),
        (([False, True], [True, False]), None, 0, torch.tensor([[1], [0]])),
        (([False, False], [False, False]), None, 0, torch.tensor([[], []])),
        (([True, False], [False, True]), -1.0, 0, torch.tensor([[0, -1.0], [-1.0, 1]])),
        (([True, False], [False, True]), torch.tensor(-1.0), 0, torch.tensor([[0, -1.0], [-1.0, 1]])),
        # Ragged
        (([True, False], [True, True]), None, 0, torch.tensor([[0, 0], [0, 1]])),
        (([False, True], [True, True]), None, 0, torch.tensor([[1, 0], [0, 1]])),
        (([True, False], [True, True]), None, -1.0, torch.tensor([[0, -1.0], [0, 1]])),
        (([False, True], [True, True]), None, -1.0, torch.tensor([[1, -1.0], [0, 1]])),
        (([False, True], [True, True]), -1.0, 0, torch.tensor([[-1.0, 1], [0, 1]])),
    ],
)
def test_apply(mask, fill_value, padding_value, exp):
    mask = torch.tensor(mask, dtype=torch.bool)
    N, L = mask.shape
    x = torch.arange(L).view(1, L, 1).expand(N, L, 1)
    o = apply_mask(mask, x, fill_value, padding_value)
    assert_close(o, exp.view_as(o).type_as(o))


def test_unapply():
    B, L, D = 2, 10, 32
    x = torch.randn((B, L, D))
    mask = create_mask((L,), 0.5, batch_size=B)
    y = apply_mask(mask, x)
    x2 = unapply_mask(mask, y)
    assert_close(x2[mask], x[mask])
    assert_close(x2[~mask], x2.new_zeros(x2.shape)[~mask])


class TestCreateMask:

    @pytest.mark.parametrize(
        "size, exp",
        [
            ((32,), 32),
            ((8, 8), 64),
        ],
    )
    def test_create_size(self, size, exp):
        assert create_mask(size, 0.5).numel() == exp

    @pytest.mark.parametrize(
        "size, ratio, exp",
        [
            ((1000,), 0.5, 500),
            ((100, 10), 0.25, 750),
        ],
    )
    def test_create_ratio(self, size, ratio, exp):
        assert create_mask(size, ratio).sum() == exp

    @pytest.mark.parametrize(
        "batch_size, size",
        [
            (2, (1000,)),
            (4, (100, 10)),
        ],
    )
    def test_create_batch_size(self, batch_size, size):
        mask = create_mask(size, 0.5, batch_size)
        assert mask.shape[0] == batch_size
        assert (mask.sum(-1) == mask.sum(-1)[0, None]).all()

    def test_create_scale(self):
        ratio = 0.5
        torch.random.manual_seed(0)
        mask = create_mask((8, 8), ratio, scale=2)
        mask_grid = mask.view(1, 1, 8, 8)
        target_size = (4, 4)
        # Average pool the mask as a a float. Mask should be all 1.0 or 0.0 within a block,
        # so pooled entries should be all 1.0 or 0.0
        pooled = F.adaptive_avg_pool2d(mask_grid.float(), target_size).view(*target_size)
        assert ((pooled == 1.0) | (pooled == 0.0)).all()

    def test_create_device(self):
        size = (16, 16)
        ratio = 0.25
        scale = 2
        batch_size = 2
        mask = create_mask(size, ratio, batch_size, scale=scale)
        assert mask.device.type == "cpu"


def test_generate_non_overlapping_mask_no_overlap():
    B, L = 5, 10
    p1, p2 = 0.3, 0.2
    # Create an initial mask1 with exactly n= int(L*p1) True values per row
    mask1 = torch.zeros((B, L), dtype=torch.bool)
    n = int(L * p1)
    for i in range(B):
        idx = torch.randperm(L)[:n]
        mask1[i, idx] = True

    mask2 = generate_non_overlapping_mask(mask1, p1, p2)
    assert mask1.shape == mask2.shape
    # Ensure no overlap
    assert not torch.logical_and(mask1, mask2).any()
    # Ensure mask2 has correct number of True values
    m = int(L * p2)
    assert torch.all(mask2.sum(-1) == m)


def test_generate_non_overlapping_mask_invalid_ratio():
    B, L = 5, 10
    p1, p2 = 0.7, 0.4  # n + m = 7 + 4 > 10
    mask1 = torch.zeros((B, L), dtype=torch.bool)
    # Fill mask1 with int(L * p1) Trues
    n = int(L * p1)
    for i in range(B):
        idx = torch.randperm(L)[:n]
        mask1[i, idx] = True
    # Should raise ValueError
    with pytest.raises(ValueError):
        _ = generate_non_overlapping_mask(mask1, p1, p2)
