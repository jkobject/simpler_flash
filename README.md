# A simpler flash attention package

[![codecov](https://codecov.io/gh/jkobject/simpler_flash/graph/badge.svg?token=3W3Y1VKWBH)](https://codecov.io/gh/jkobject/simpler_flash)
[![CI](https://github.com/jkobject/simpler_flash/actions/workflows/main.yml/badge.svg)](https://github.com/jkobject/simpler_flash/actions/workflows/main.yml)
[![PyPI version](https://badge.fury.io/py/simpler_flash.svg)](https://badge.fury.io/py/simpler_flash)
[![Downloads](https://pepy.tech/badge/simpler_flash)](https://pepy.tech/project/simpler_flash)
[![Downloads](https://pepy.tech/badge/simpler_flash/month)](https://pepy.tech/project/simpler_flash)
[![Downloads](https://pepy.tech/badge/simpler_flash/week)](https://pepy.tech/project/simpler_flash)
[![GitHub issues](https://img.shields.io/github/issues/jkobject/simpler_flash)](https://img.shields.io/github/issues/jkobject/simpler_flash)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14749466.svg)](https://doi.org/10.5281/zenodo.14749466)

a full suite of fast:
- flash attention mechanisms with triton,
- flashattention with pytorch,
- flash-hyperattention,
- flash-softpick attention,
- flash-adasplash,
- flash criss-cross attention.

Just pip install it with conda or with uv, have a compatible GPU, and it should work.

installs in 1 sec with no complex steps.

## Installation

You need to use Python 3.10+.

There are several alternative options to install simpler_flash:

1. Install the latest release of `simpler_flash` from [PyPI][]:

```bash
pip install simpler_flash
```

In some old GPUs, you might need to use a lower block dimension. For now, it needs to be updated directly in the source code, e.g., by setting MAX_BLOCK_SIZE=64 instead of 128. It will reduce your max head size to 64

## Usage

```python

from simpler_flash import FlashTransformer


self.transformer = FlashTransformer(
    d_model=1024,
    nhead=16,
    nlayers=12,
    dropout=0.1,
    attn_dropout=0.1 # only works with regular flash attn
    cross_attn=False, # else does cross attn too
    cross_dim=512, # if the cross attention is on another emb dim
    mlp_ratio=4, # classic
    attn_type="normal",
        - "flash": Use flash attention's v2 triton's implementation (older)
        - "normal": Use pytorch's MHA attention (can be flash in some cases) (newer).
        - "hyper": Use HyperAttention.
        - "criss-cross": Use Criss-Cross attention (the fastest attention available), !! Please cite the scPRINT-2 paper if using it !!
        - "softpick": Use SoftPick attention.
        - "flash-softpick": Use efficient softpick attention
    num_heads_kv=4, # option to do Grouped Attention
    checkpointing=True, # option to use checkpointing
    prenorm=True, # option to use prenorm
    drop_path_rate=0.1, # option to use drop path
    sketcher_size=32, # for criss-cross, the number of sketching embeddings
    sketcher_dim=256, # for criss-cross, the dimension of the sketching embeddings
)

transformer_output = self.transformer(
    encoding,
    return_qkv=get_attention_layer, #option to get the q,k,v matrices (to extract attention scores for example)
    bias=bias if do_bias else None, # option to add attention bias
    bias_layer=list(range(self.nlayers - 1)), # option to add attention bias to specific layers

)
```
