#!/usr/bin/env python3
"""
Test script for dynamic block size detection in FlashAttention.
"""

import torch
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from simpler_flash.flashattention import (
        get_max_block_size, 
        set_max_block_size, 
        clear_block_size_cache,
        get_optimal_block_size,
        flash_attn_func
    )
    
    print("Testing FlashAttention dynamic block size detection...")
    print("=" * 60)
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}")
        print(f"Shared memory per block: {props.shared_memory_per_block} bytes")
        print(f"Current device: {device}")
        print()
        
        # Test optimal block size detection
        optimal_size = get_optimal_block_size()
        print(f"Detected optimal block size: {optimal_size}")
        
        # Test cached access
        cached_size = get_max_block_size()
        print(f"Cached block size: {cached_size}")
        
        # Test manual override
        print("\nTesting manual override...")
        set_max_block_size(32)
        override_size = get_max_block_size()
        print(f"Override block size: {override_size}")
        
        # Reset to optimal
        clear_block_size_cache()
        reset_size = get_max_block_size()
        print(f"Reset to optimal: {reset_size}")
        
        # Test actual FlashAttention execution
        print("\nTesting FlashAttention execution...")
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64
        
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                       dtype=torch.float16, device='cuda', requires_grad=True)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                       dtype=torch.float16, device='cuda', requires_grad=True)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                       dtype=torch.float16, device='cuda', requires_grad=True)
        
        # Forward pass
        out, lse = flash_attn_func(q, k, v)
        print(f"Forward pass successful with shape: {out.shape}")
        
        # Backward pass
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        print(f"Backward pass successful")
        print(f"Q grad shape: {q.grad.shape}")
        
        print("\n✅ All tests passed!")
        
    else:
        print("CUDA not available. Testing fallback behavior...")
        block_size = get_max_block_size()
        print(f"Fallback block size: {block_size}")
        
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
