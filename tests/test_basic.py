import torch

import simpler_flash
from simpler_flash import FlashTransformer


def test_package_has_version():
    assert simpler_flash.__version__ is not None


def test_base():
    transformer = FlashTransformer(
        d_model=64,
        nhead=2,
        dropout=0.1,
        # attn_dropout=0.1,
        nlayers=2,
        attn_type="normal",
    )
    encoding = torch.randn(1, 16, 64)
    ret = transformer(
        encoding,
        # return_qkv=[False, True],
        bias=None,
        bias_layer=list(range(1)),
        mask_zeros=None,
    )
    assert ret.shape == (1, 16, 64)
    assert torch.is_tensor(ret)
    # no nans
    assert not torch.isnan(ret).any()
