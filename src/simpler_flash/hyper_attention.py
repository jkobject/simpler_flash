import torch

from .flashattention import flash_attn_func
from .hyper.angular_lsh_triton import AngularLSHTriton
from .hyper.hyper_attn_triton import hyper_attn_func


class HyperAttention(torch.nn.Module):
    def __init__(
        self,
        input_dim=64,
        lsh_num_projs=8,
        block_size=128,
        sample_size=128,
        min_seq_len=2048,
        smooth_block=False,
        **kwargs,
    ):
        """
        - block_size and sample_size must be divisible by 128
        """
        super().__init__()
        self.input_dim = input_dim
        self.lsh_num_projs = lsh_num_projs
        self.block_size = block_size
        self.sample_size = sample_size
        self.min_seq_len = min_seq_len
        self.smooth_block = smooth_block
        self.lsh = AngularLSHTriton(num_projs=self.lsh_num_projs, dim=(1, 1, input_dim))

    def forward(
        self,
        query: torch.tensor,
        key: torch.tensor,
        value: torch.tensor,
        scale=None,
        causal=False,
        return_lse=False,
    ):
        """
        Forward function for HyperAttention. If no causal masking, simply invokes forward_no_causal_mask method.
        If there is causal masking, it partitions the attention matrix and recurses on the partitions.
        inputs:
            - query, key, and valu: must have same sequence lengths but dimension of values vectors can be different
            from that of query or key
            - sequence lengths must be divisible by block_size
        output:
            - attn: (approximation of) the final attention output tensor
            - lse: (approximation of) log sum exp of the qk matrix
        """
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        n_query = query.shape[2]
        batch_size, n_heads, n_key, dim = key.shape
        scale = scale or dim ** (-0.5)
        assert n_query == n_key

        # without causal masking
        if causal is False:
            attn, lse = self.forward_no_causal_mask(query, key, value, scale)

        else:  # with causal masking
            print("not causal")
        if not return_lse:
            return attn
        else:
            return attn, lse

    def forward_no_causal_mask(self, query, key, value, scale):
        """
        - sequence lengths must be divisible by block_size
        """
        batch_size, n_query, head_size, dim = query.shape
        if self.min_seq_len > n_query:
            attn, lse = flash_attn_func(
                query,
                key,
                value,
                None,
                False,
                None,
            )
        else:
            # Hash keys and queries via SortLSH and obtain buckets
            _, query_sort_idx = torch.sort(
                self.lsh.hash_triton(query.transpose(1, 2)), dim=2, stable=True
            )  # batch_size x head_size x n
            _, key_sort_idx = torch.sort(
                self.lsh.hash_triton(key.transpose(1, 2)), dim=2, stable=True
            )

            # Now run hyper attention function on q,k,v and the permutations
            attn, lse = hyper_attn_func(
                query,
                key,
                value,
                query_sort_idx.transpose(1, 2),
                key_sort_idx.transpose(1, 2),
                self.block_size,
                self.sample_size,
                scale,
                self.smooth_block,
            )

        return attn, lse.unsqueeze(-1)
