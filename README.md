# simpler_flash

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/jkobject/simpler_flash/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/simpler_flash

a full on suite of fast flash attention mechanisms, flashattention, flash-hyperattention, flash-softpick attention, flash-adasplash, flash criss-cross attention. just pip install it with conda or with uv, have a compatible gpu and it should work.

installs in 1 sec without anything complex.

## Installation

You need to use Python 3.10+.

There are several alternative options to install simpler_flash:

1. Install the latest release of `simpler_flash` from [PyPI][]:

```bash
pip install simpler_flash
```

1. Install the latest development version:

```bash
pip install git+https://github.com/jkobject/simpler_flash.git@main
```

in some old GPUs, you might need to use a lower block dim, for now it just has to be updated in the source code directly e.g. setting MAX_BLOCK_SIZE=64 instead of 128. it will reduce your max head size to 64

## Usage

```python

from simpler_flash import FlashTransformer


self.transformer = FlashTransformer(
    d_model=1024,
    nhead=16,
    nlayers=12,
    dropout=0.1,
    use_flash_attn=True,
    num_heads_kv=4, # option to do Grouped Attention
    checkpointing=True, # option to use checkpointing
    prenorm=True, # option to use prenorm
    drop_path_rate=0.1, # option to use drop path
)

transformer_output = self.transformer(
    encoding,
    return_qkv=get_attention_layer, #option to get the q,k,v matrices (to extract attention scores for example)
    bias=bias if do_bias else None, # option to add attention bias
    bias_layer=list(range(self.nlayers - 1)), # option to add attention bias to specific layers

)
```
