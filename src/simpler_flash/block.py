# Copyright (c) 2024, Tri Dao.

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import StochasticDepth

from .mha import MHA
from .mlp import Mlp

try:
    from .layer_norm import RMSNorm, layer_norm_fn
except ModuleNotFoundError:
    layer_norm_fn = None
    RMSNorm = None


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        mixer_cls: Callable | None = None,
        mlp_cls: Callable | None = None,
        norm_cls: Callable = partial(nn.LayerNorm, eps=1e-6),
        dropout_cls: type[nn.Dropout] = nn.Dropout,
        resid_dropout1: float = 0.0,
        resid_dropout2: float = 0.0,
        drop_path1: float = 0.0,
        drop_path2: float = 0.0,
        fused_dropout_add_ln: bool = False,
        return_residual: bool = False,
        residual_in_fp32: bool = False,
        sequence_parallel: bool = False,
        mark_shared_params: bool = False,
        cross_attn: Callable | None = None,
        sketcher_dim: int = 128,
    ):
        """
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).

        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.

        Args:
            dim (int): the number of features in the input.
            mixer_cls (Optional[Callable], optional): the class to use for the mixer layer. Defaults to None.
            mlp_cls (Optional[Callable], optional): the class to use for the mlp layer. Defaults to None.
            norm_cls (Callable, optional): the class to use for the layer norm. Defaults to partial(nn.LayerNorm, eps=1e-6).
            dropout_cls (Type[nn.Dropout], optional): the class to use for the dropout. Defaults to nn.Dropout.
            resid_dropout1 (float, optional): the dropout probability for the first dropout layer. Defaults to 0.0.
            resid_dropout2 (float, optional): the dropout probability for the second dropout layer. Defaults to 0.0.
            drop_path1 (float, optional): the drop path probability for the first drop path layer. Defaults to 0.0.
            drop_path2 (float, optional): the drop path probability for the second drop path layer. Defaults to 0.0.
            fused_dropout_add_ln (bool, optional): whether to fuse the dropout, add and layer norm. Defaults to False.
            return_residual (bool, optional): whether each of the sub-layers (mixer and mlp) will return the residual.
                This is for performance reason: for post-norm architecture, returning the input allows us
                to fuse the backward of nn.Linear with the residual connection.
                Defaults to False.
            residual_in_fp32 (bool, optional): whether to keep the residual in fp32. This is for performance reason:
                for post-norm architecture, keeping the residual in fp32 allows us to fuse the backward of nn.Linear
                with the residual connection. Defaults to False.
            sequence_parallel (bool, optional): whether to use sequence parallelism. Defaults to False.
            mark_shared_params (bool, optional): whether to mark the norm parameters as "shared_params".
                This is useful when we want to sync the norm parameters across workers. Defaults to False.
            cross_attn (Optional[Callable], optional): an additional attention layer for cross-attention. Defaults to None.
            sketcher_dim (int, optional): the dimension of the sketcher. Defaults to 128.
        """
        super().__init__()
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        if cross_attn:
            if self.mixer.attn_type == "criss-cross":
                print(
                    "Criss-cross attention only works for self-attention, using regular attn for this..."
                )
            self.cross_attn = cross_attn(dim)
            self.dropout3 = dropout_cls(resid_dropout1)
            self.drop_path3 = StochasticDepth(drop_path1, mode="row")
            self.norm3 = norm_cls(dim)
        else:
            self.cross_attn = None
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode="row")
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode="row")
            self.norm2 = norm_cls(dim)
            if self.mixer.attn_type == "criss-cross":
                self.norm_cc = norm_cls(sketcher_dim)

        if self.fused_dropout_add_ln:
            assert layer_norm_fn is not None, "Triton is not installed"
            assert isinstance(self.norm1, (nn.LayerNorm, RMSNorm)) and isinstance(
                self.dropout1, nn.Dropout
            )

        # TD [2023-01-07]: TODO: During training, if sequence_parallel is False and dropout != 0.0,
        # then the input to each worker in the tensor parallel group will be different.
        # This would produce wrong outputs? Somehow we'd need to sync the RNG state across workers.
        # For now this is not an issue because we always use sequence_parallel=True during training
        # and only use sequence_parallel=False during inference.

        # Mark the norm parameters as "sequence_parallel" so that we run all-reduce on their grads.
        if sequence_parallel:
            for p in self.norm1.parameters():
                p._sequence_parallel = True
            if hasattr(self, "norm2"):
                for p in self.norm2.parameters():
                    p._sequence_parallel = True
            if hasattr(self, "norm_cc"):
                for p in self.norm_cc.parameters():
                    p._sequence_parallel = True
        # Mark the norm parameters as "shared_params" so that we sync their values at init.
        if mark_shared_params:
            for p in self.norm1.parameters():
                p._shared_params = True
            if hasattr(self, "norm2"):
                for p in self.norm2.parameters():
                    p._shared_params = True
            if hasattr(self, "norm_cc"):
                for p in self.norm_cc.parameters():
                    p._shared_params = True

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def set_seq_parallel(self, val: bool):
        for p in self.norm1.parameters():
            p._sequence_parallel = val
        if hasattr(self, "norm2"):
            for p in self.norm2.parameters():
                p._sequence_parallel = val
        if hasattr(self, "norm_cc"):
            for p in self.norm_cc.parameters():
                p._sequence_parallel = val

    def forward(
        self,
        hidden_states: Tensor,
        x_kv: Tensor | None = None,
        latent_kv: Tensor | None = None,
        residual: Tensor | None = None,
        bias: Tensor | None = None,
        is_causal: bool | None = None,
        src_key_padding_mask: Tensor | None = None,
        mixer_subset: Tensor | None = None,
        mixer_kwargs: dict[str, Any] | None = None,
        return_qkv: bool = False,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states (Tensor): The sequence to be passed to the encoder layer. This is a required argument.
            residual (Optional[Tensor]): This argument is used differently based on the normalization method.
            mixer_subset: This argument is used only for cross-attention.
                If not None, a subset of the input sequence 'x' is taken before applying the query projection.
                This is particularly useful for models like ViT where only the CLS token in the last layer is of interest.
            mixer_kwargs: This argument is used only for cross-attention.
                It is a dictionary of additional arguments to be passed to the mixer.
            return_qkv: If True, the function will return the query, key, and value tensors.

        Returns
        -------
            Tensor or Tuple[Tensor, Tensor]: The output tensor of the encoder layer.
            If return_qkv is True, the function will return a tuple of the output tensor and the query, key, and value tensors.
        """
        if not self.fused_dropout_add_ln:
            dropped = self.drop_path1(self.dropout1(hidden_states))
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.mixer.attn_type == "criss-cross":
                latent_kv = self.norm_cc(latent_kv.to(dtype=self.norm_cc.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            if self.drop_path1.p == 0 or not self.training:
                rowscale1 = None
            else:
                rowscale1 = self.drop_path1(
                    torch.ones(
                        hidden_states.shape[:-1],
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                )
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm1.weight,
                self.norm1.bias,
                residual=residual,
                eps=self.norm1.eps,
                dropout_p=self.dropout1.p if self.training else 0.0,
                rowscale=rowscale1,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm1, RMSNorm),
            )
            if self.mixer.attn_type == "criss-cross":
                latent_kv = layer_norm_fn(
                    latent_kv,
                    self.norm_cc.weight,
                    self.norm_cc.bias,
                    residual=None,
                    eps=self.norm_cc.eps,
                    dropout_p=0.0,
                    rowscale=None,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                    is_rms_norm=isinstance(self.norm_cc, RMSNorm),
                )
        if mixer_kwargs is None:
            mixer_kwargs = {}
        if mixer_subset is not None:
            mixer_kwargs["mixer_subset"] = mixer_subset
        if self.cross_attn and x_kv is not None:
            hidden_states = self.cross_attn(
                hidden_states,
                x_kv=x_kv,
                return_qkv=False,
                key_padding_mask=src_key_padding_mask,
                **mixer_kwargs,
            )
            if not self.fused_dropout_add_ln:
                dropped = self.drop_path3(self.dropout3(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm3(residual.to(dtype=self.norm3.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                if self.drop_path3.p == 0 or not self.training:
                    rowscale3 = None
                else:
                    rowscale3 = self.drop_path3(
                        torch.ones(
                            hidden_states.shape[:-1],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    )
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm3.weight,
                    self.norm3.bias,
                    residual=residual,
                    eps=self.norm3.eps,
                    dropout_p=self.dropout3.p if self.training else 0.0,
                    rowscale=rowscale3,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    is_rms_norm=isinstance(self.norm3, RMSNorm),
                )
        hidden_states = self.mixer(
            hidden_states,
            x_kv=latent_kv,
            return_qkv=return_qkv,
            bias=bias,
            key_padding_mask=src_key_padding_mask,
            **mixer_kwargs,
        )
        if return_qkv:
            qkv = hidden_states[1]
            hidden_states = hidden_states[0]
        if mixer_subset is not None:
            residual = residual[:, mixer_subset]
        if self.mixer.attn_type == "criss-cross":
            latent_kv = hidden_states[1]
            hidden_states = hidden_states[0]
        if not isinstance(self.mlp, nn.Identity):
            if not self.fused_dropout_add_ln:
                dropped = self.drop_path2(self.dropout2(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

            else:
                if self.drop_path2.p == 0 or not self.training:
                    rowscale2 = None
                else:
                    rowscale2 = self.drop_path2(
                        torch.ones(
                            hidden_states.shape[:-1],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    )
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    eps=self.norm2.eps,
                    dropout_p=self.dropout2.p if self.training else 0.0,
                    rowscale=rowscale2,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    is_rms_norm=isinstance(self.norm2, RMSNorm),
                )
            # maybe in the future add a criss-cross mlp and don't forget to add the residual connection
            hidden_states = self.mlp(hidden_states)
            if self.mixer.attn_type == "criss-cross":
                hidden_states = (hidden_states, latent_kv)
        return (
            (hidden_states, residual)
            if not return_qkv
            else (
                hidden_states,
                residual,
                qkv,
            )
        )
