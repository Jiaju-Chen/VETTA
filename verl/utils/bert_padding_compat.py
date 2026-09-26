"""Compatibility wrapper for FlashAttention bert padding helpers.

Some FlashAttention wheels can be importable as a package but fail while loading
their CUDA extension because of a PyTorch C++ ABI mismatch. The padding helpers
used by the FSDP actor/critic path are pure PyTorch operations, so keep training
usable by falling back to local implementations when the extension cannot load.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except Exception:

    class IndexFirstAxis(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, indices):
            ctx.save_for_backward(indices)
            assert input.ndim >= 2
            ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
            second_dim = other_shape.numel()
            return torch.gather(
                rearrange(input, "b ... -> b (...)"),
                0,
                repeat(indices, "z -> z d", d=second_dim),
            ).reshape(-1, *other_shape)

        @staticmethod
        def backward(ctx, grad_output):
            (indices,) = ctx.saved_tensors
            assert grad_output.ndim >= 2
            other_shape = grad_output.shape[1:]
            grad_output = rearrange(grad_output, "b ... -> b (...)")
            grad_input = torch.zeros(
                [ctx.first_axis_dim, grad_output.shape[1]],
                device=grad_output.device,
                dtype=grad_output.dtype,
            )
            grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
            return grad_input.reshape(ctx.first_axis_dim, *other_shape), None

    index_first_axis = IndexFirstAxis.apply

    class IndexPutFirstAxis(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values, indices, first_axis_dim):
            ctx.save_for_backward(indices)
            assert indices.ndim == 1
            assert values.ndim >= 2
            output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
            output[indices] = values
            return output

        @staticmethod
        def backward(ctx, grad_output):
            (indices,) = ctx.saved_tensors
            return grad_output[indices], None, None

    index_put_first_axis = IndexPutFirstAxis.apply

    def unpad_input(hidden_states, attention_mask, unused_mask=None):
        all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
        seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
        used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        return (
            index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
            used_seqlens_in_batch,
        )

    def pad_input(hidden_states, indices, batch, seqlen):
        output = index_put_first_axis(hidden_states, indices, batch * seqlen)
        return rearrange(output, "(b s) ... -> b s ...", b=batch)
