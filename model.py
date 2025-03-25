import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from flax.training import train_state, checkpoints
import optax
import math
import os
import time
import glob
import string
import random
import pickle
import argparse
import numpy as np # only for data loads
from contextlib import nullcontext
from typing import Any

config = {
    "n_embd": 256,
    "n_head": 16,
    "n_layer": 4,
    "n_experts": 32,
    "dropout": 0.2,
    "vocab_size": 65,
    "ctx_len": 2048,
    "init_moe_scaling": 1.25,
    "type": ['mlp', 'moe', 'mlp', 'moe'],
    "device": 'gpu' if jax.default_backend() == 'gpu' else 'cpu',  # Jax-style device detection
    "use_expert_bias" : True,
    "rms_norm_eps" : 1e-6,
    "block_size" : 16, #for NSA,
    "window_size" : 128, # for NSA,
    "num_tokens_to_keep" : 2048 // 4,
}

# RoPE/RoFormer by Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position Embedding.

class RoPE(nn.Module):
    d: int
    base: int = 10000

    @nn.compact
    def __call__(self, x):
        head_dim = x.shape[-1]
        theta = 1.0 / (self.base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / self.d))
        seq_idx = jnp.arange(x.shape[-2], dtype=jnp.float32)  # Use -2 for sequence length
        idx_theta = jnp.einsum('n,d->nd', seq_idx, theta)

        cos_cache = jnp.cos(idx_theta)
        sin_cache = jnp.sin(idx_theta)

        # Concatenate and add batch and head dims
        cos_cache = jnp.concatenate([cos_cache, cos_cache], axis=-1)[None, None, :, :]
        sin_cache = jnp.concatenate([sin_cache, sin_cache], axis=-1)[None, None, :, :]

        # do RoPE
        x_real, x_imag = jnp.split(x, 2, axis=-1)
        x_rotated_real = x_real * cos_cache - x_imag * sin_cache
        x_rotated_imag = x_real * sin_cache + x_imag * cos_cache
        x_rotated = jnp.concatenate([x_rotated_real, x_rotated_imag], axis=-1)

        return x_rotated

# rope helpers

def precomp_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    return jnp.cos(freqs), jnp.sin(freqs)

def apply_rope(x: jnp.ndarray, y: jnp.ndarray, freqs_cis) -> tuple[jnp.ndarray, jnp.ndarray]:
    cos_freqs, sin_freqs = freqs_cis
    seq_len = x.shape[-2]

    cos_seq = cos_freqs[:seq_len]
    sin_seq = sin_freqs[:seq_len]
    cos_seq = cos_seq[None, None, :, :]  # Add batch and head dimensions
    sin_seq = sin_seq[None, None, :, :]  # Add batch and head dimensions

    x_real, x_imag = jnp.split(x, 2, axis=-1)
    y_real, y_imag = jnp.split(y, 2, axis=-1)

    x_rotated_real = x_real * cos_seq - x_imag * sin_seq
    x_rotated_imag = x_real * sin_seq + x_imag * cos_seq
    y_rotated_real = y_real * cos_seq - y_imag * sin_seq
    y_rotated_imag = y_real * sin_seq + y_imag * cos_seq

    x_rotated = jnp.concatenate([x_rotated_real, x_rotated_imag], axis=-1)
    y_rotated = jnp.concatenate([y_rotated_real, y_rotated_imag], axis=-1)
    return x_rotated, y_rotated

# RMSNorm, efficient free lunch layernorm

class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return x / rms * self.variable('params', 'scale', lambda: jnp.ones(self.dim)).value

# Attn (MLA-NSA Hybrid/Amalgamation Attention)
# Citations:
# - Vaswani et al., 2017 (for the core attention mechanism)
# - Yuan et al., 2025 (for Native Sparse Attention)
# - DeepSeek-AI, 2024 (for DeepSeek-V3, source for MLA)

# Mixing MLA and NSA is kind of a three prong approach, strange, but also keep in mind I took NSA for better loss due to sparsity, etc, not the hardware specific optim
# replace NSA compression stage with MLA projection, use a linear layer for the tok_selector, and have a sliding window section, full during training, but centered at imp pos at inference, works well

class Attn(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, is_training: bool, current_pos=None, use_cache=False):
        B, T, C = x.shape
        n_embd = self.config['n_embd']
        n_head = self.config['n_head']
        dropout = self.config['dropout']
        ctx_len = self.config['ctx_len']
        rms_norm_eps = self.config['rms_norm_eps']
        v_head_dim = 96
        kv_lora_rank = 32
        q_lora_rank = 3 * kv_lora_rank
        rope_head_dim = 64
        nope_head_dim = 32
        value_dim = n_head * v_head_dim
        nope_dim = n_head * nope_head_dim
        rope_dim = n_head * rope_head_dim
        block_size = self.config['block_size']
        num_blocks = ctx_len // block_size
        window_size = self.config['window_size']
        num_tokens_to_keep = self.config['num_tokens_to_keep']


        # Branch 1: Coarse-grained compression with MLA
        compress_q_linear = nn.Dense(features=q_lora_rank, use_bias=False, kernel_init=initializers.xavier_uniform())
        q_norm = RMSNorm(dim=q_lora_rank, eps=rms_norm_eps)
        decompress_q_nope = nn.Dense(features=nope_dim, use_bias=False, kernel_init=initializers.xavier_uniform())
        decompress_q_rope = nn.Dense(features=rope_dim, use_bias=False, kernel_init=initializers.xavier_uniform())

        compress_kv_linear = nn.Dense(features=kv_lora_rank, use_bias=False, kernel_init=initializers.xavier_uniform())
        kv_norm = RMSNorm(dim=kv_lora_rank, eps=rms_norm_eps)
        decompress_k_nope = nn.Dense(features=nope_dim, use_bias=False, kernel_init=initializers.xavier_uniform())
        decompress_v_linear = nn.Dense(features=value_dim, use_bias=False, kernel_init=initializers.xavier_uniform())
        
        k_rope_linear = nn.Dense(features=rope_dim, use_bias=False, kernel_init=initializers.xavier_uniform())

        # Branch 2: Token Selection
        importance_scorer = nn.Dense(features=1, kernel_init=initializers.xavier_uniform())
        selection_k = nn.Dense(features=n_head * (rope_head_dim + nope_head_dim), use_bias=False, kernel_init=initializers.xavier_uniform())
        selection_v = nn.Dense(features=value_dim, use_bias=False, kernel_init=initializers.xavier_uniform())

        # Branch 3: Sliding Window
        window_k = nn.Dense(features=n_head * (rope_head_dim + nope_head_dim), use_bias=False, kernel_init=initializers.xavier_uniform())
        window_v = nn.Dense(features=value_dim, use_bias=False, kernel_init=initializers.xavier_uniform())

        branch_gate = nn.Dense(features=3, kernel_init=initializers.xavier_uniform())

        proj = nn.Dense(features=n_embd, use_bias=False, kernel_init=initializers.xavier_uniform())
        res_dropout = nn.Dropout(rate=dropout)

        freqs_cis = precomp_freqs_cis(rope_head_dim, ctx_len)

        compressed_q = compress_q_linear(x)
        norm_q = q_norm(compressed_q)
        query_nope = decompress_q_nope(norm_q)
        query_rope = decompress_q_rope(norm_q)

        query_nope = query_nope.reshape(B, T, n_head, nope_head_dim).transpose(0, 2, 1, 3)
        query_rope = query_rope.reshape(B, T, n_head, rope_head_dim).transpose(0, 2, 1, 3)

        q_rope, _ = apply_rope(query_rope, query_rope, freqs_cis)

        q_recombined = jnp.concatenate([query_nope, q_rope], axis=-1)

        branch_weights = nn.softmax(branch_gate(x).mean(axis=1))  # [B, 3]

        compressed_kv = compress_kv_linear(x)
        norm_kv = kv_norm(compressed_kv)
        key_nope_1 = decompress_k_nope(norm_kv)
        value_1 = decompress_v_linear(norm_kv)
        key_rope_1 = k_rope_linear(norm_kv)

        key_nope_1 = key_nope_1.reshape(B, T, n_head, nope_head_dim).transpose(0, 2, 1, 3)
        key_rope_1 = key_rope_1.reshape(B, T, n_head, rope_head_dim).transpose(0, 2, 1, 3)
        value_1 = value_1.reshape(B, T, n_head, v_head_dim).transpose(0, 2, 1, 3)

        _, k_rope_1 = apply_rope(key_rope_1, key_rope_1, freqs_cis) # MLA seems to be working okay for jax without the head div, keep it this way and see ehat happens
        k_recombined_1 = jnp.concatenate([key_nope_1, k_rope_1], axis=-1)

        importance_scores = importance_scorer(x)

        _, indices = jax.lax.top_k(importance_scores.squeeze(-1), min(num_tokens_to_keep, T)) # use the linear layers probs on the topk
        indices = jnp.sort(indices, axis=1)
        batch_indices = jnp.arange(B)[:, None]
        selected_tokens = x[batch_indices, indices]

        B, S, _ = selected_tokens.shape 
        k_selected = selection_k(selected_tokens).reshape(B, S, n_head, rope_head_dim + nope_head_dim).transpose(0, 2, 1, 3)
        v_selected = selection_v(selected_tokens).reshape(B, S, n_head, v_head_dim).transpose(0, 2, 1, 3)

        k_selected_rope = k_selected[:, :, :, nope_head_dim:]
        k_selected_nope = k_selected[:, :, :, :nope_head_dim]
        _, k_selected_rope = apply_rope(k_selected_rope, k_selected_rope, freqs_cis)
        k_selected = k_selected.at[:, :, :, nope_head_dim:].set(k_selected_rope)
        k_selected = k_selected.at[:, :, :, :nope_head_dim].set(k_selected_nope)

        if is_training or current_pos is None:
            window_tokens = x
            W = T # window size is just T during train

        else:
            # During inf, get a window centered around the current pos

            window_start = jnp.maximum(0, current_pos - window_size // 2)
            window_end = jnp.minimum(T, window_start + window_size)
            
            # Use dynamic slicing, works fine for jax, one of the few upsides
            window_tokens = jax.lax.dynamic_slice_in_dim(x, window_start, window_end - window_start, axis=1)
            W = window_end - window_start

        k_window = window_k(window_tokens).reshape(B, W, n_head, rope_head_dim + nope_head_dim).transpose(0, 2, 1, 3)
        v_window = window_v(window_tokens).reshape(B, W, n_head, v_head_dim).transpose(0, 2, 1, 3)

        k_window_rope = k_window[:, :, :, nope_head_dim:]
        k_window_nope = k_window[:, :, :, :nope_head_dim]
        _, k_window_rope = apply_rope(k_window_rope, k_window_rope, freqs_cis)

        k_window = k_window.at[:, :, :, nope_head_dim:].set(k_window_rope)
        k_window = k_window.at[:, :, :, :nope_head_dim].set(k_window_nope)

        # KV Caching
        if not is_training:

            if use_cache:
                
                cache = self.variable('cache', 'kv_cache', lambda: {
                    'k_recombined_1': jnp.zeros((B, n_head, ctx_len, rope_head_dim + nope_head_dim), dtype=x.dtype),
                    'v_1': jnp.zeros((B, n_head, ctx_len, v_head_dim), dtype=x.dtype),
                    'cache_filled': 0,
                })

                k_cache = cache.value['k_recombined_1']
                v_cache = cache.value['v_1']
                cache_filled = cache.value['cache_filled']

                new_cache_filled = min(cache_filled + T, ctx_len)
                k_cache = jax.lax.dynamic_update_slice(k_cache, k_recombined_1[:,:,:new_cache_filled - cache_filled], (0, 0, cache_filled, 0))
                v_cache = jax.lax.dynamic_update_slice(v_cache, value_1[:,:,:new_cache_filled - cache_filled], (0, 0, cache_filled, 0))

                cache.value['k_recombined_1'] = k_cache
                cache.value['v_1'] = v_cache
                cache.value['cache_filled'] = new_cache_filled

                k_recombined_1 = k_cache[:, :, :new_cache_filled] # use updated cache
                value_1 = v_cache[:,:,:new_cache_filled]

        # Branches
        output_1 = jax.nn.dot_product_attention(q_recombined, k_recombined_1, value_1, is_causal=True)
        output_2 = jax.nn.dot_product_attention(q_recombined, k_selected, v_selected, is_causal=False)
        output_3 = jax.nn.dot_product_attention(q_recombined, k_window, v_window, is_causal=True)

        blended_output = (
            output_1 * branch_weights[:, 0, None, None, None] +
            output_2 * branch_weights[:, 1, None, None, None] +
            output_3 * branch_weights[:, 2, None, None, None]
        )

        output = blended_output.transpose(0, 2, 1, 3).reshape(B, T, value_dim)
        output = proj(output)
        output = res_dropout(output, deterministic=not is_training)
        return output

class MLP(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, is_training: bool):
        n_embd = self.config['n_embd']
        dropout_rate = self.config['dropout']

        x = nn.Dense(features=4 * n_embd, kernel_init=initializers.xavier_uniform())(x)
        x = nn.gelu(x)
        x = nn.Dense(features=n_embd, kernel_init=initializers.xavier_uniform())(x)
        x = nn.Dropout(rate=dropout_rate)(x, deterministic=not is_training)
        return x

# DS-MoE Layer
# Citations:
# - DeepSeek-AI, 2024 (for DeepSeek-MoE)
# - 1a3orn, 2025 (for regular MoE)

class UnitCenteredNoise(nn.Module):
    scaling: float = 0.02
    
    @nn.compact
    def __call__(self, x, is_training): # mutability and inheritance hurt here for a bit sometimes
        base = 1.0 - (self.scaling * 0.5)
        if is_training:
            noise = jax.random.uniform(self.make_rng('noise'), x.shape, dtype=x.dtype)
            noise_centered = (noise * self.scaling) + base
            return x * noise_centered
        else:
            return x

class DSMoE(nn.Module):
    config: dict
    index: int
    num_exp: int = 4

    @nn.compact
    def __call__(self, x, is_training: bool):
        hidden_dim = self.config['n_embd'] * 2
        num_experts = self.config["n_experts"]
        moe_scaling = self.config["init_moe_scaling"]

        experts = [MLP(config=self.config) for _ in range(num_experts)]

        # Explicitly pass is_training to each layer in Sequential, inheritance goofing
        gate = nn.Sequential([
            lambda x: nn.Dense(features=num_experts - 1, kernel_init=initializers.xavier_uniform())(x),
            lambda x : UnitCenteredNoise(scaling=0.02)(x, is_training=is_training),
            nn.softmax
        ])

        # Initialize expert bias (excluding the shared exp)
        expert_bias = self.variable('params', 'expert_bias', lambda: jnp.zeros(num_experts - 1))

        b, t, c = x.shape
        x_flat = x.reshape(b * t, c)

        gate_val_continuous = gate(x_flat)
        biased_gate_vals = gate_val_continuous + expert_bias.value

        # get top-(num_exp-1) experts
        gate_vals, gate_val_indices = jax.lax.top_k(biased_gate_vals, self.num_exp - 1)
        gate_vals = gate_vals / jnp.sum(gate_vals, axis=-1, keepdims=True)  # normalize

        # prepend the shared expert
        shared_expert_weight = jnp.ones_like(gate_vals[:, :1]) / self.num_exp
        gate_vals = jnp.concatenate([shared_expert_weight, gate_vals * (self.num_exp - 1) / self.num_exp], axis=-1)
        gate_val_indices = jnp.concatenate([jnp.zeros_like(gate_val_indices[:, :1]), gate_val_indices + 1], axis=-1)

        expert_outputs = jnp.stack([expert(x_flat, is_training=is_training) for expert in experts], axis=0)
        router_weights = jnp.zeros((x_flat.shape[0], num_experts), dtype=x.dtype)

        # efficient accum
        def update_router_weights(router_weights, i):
            idx = jax.lax.dynamic_slice_in_dim(gate_val_indices, i, 1, axis=1)
            val = jax.lax.dynamic_slice_in_dim(gate_vals, i, 1, axis=1)
            router_weights = router_weights.at[jnp.arange(x_flat.shape[0])[:, None], idx].add(val)
            return router_weights, None

        router_weights, _ = jax.lax.scan(update_router_weights, router_weights, jnp.arange(self.num_exp))

        weighted_outputs = expert_outputs * router_weights.T[:, :, None]
        output = jnp.sum(weighted_outputs, axis=0)

        return output.reshape(b, t, c), router_weights

class Block(nn.Module):
    config: dict
    index: int

    @nn.compact
    def __call__(self, x, is_training: bool, current_pos = None, use_cache=False):
        attn = Attn(config=self.config, name="attn") # jax naming??
        ffn_type = self.config['type'][self.index]

        if ffn_type == "mlp":
            ffn = MLP(config=self.config, name="mlp") 
        elif ffn_type == "moe":
            ffn = DSMoE(config=self.config, index=self.index, name="moe")
        else:
            raise ValueError(f"Invalid layer type: {ffn_type}")

        rm1 = RMSNorm(dim=self.config['n_embd'], name="rm1") 
        rm2 = RMSNorm(dim=self.config['n_embd'], name="rm2")

        x = x + attn(rm1(x), is_training=is_training, current_pos=current_pos, use_cache=use_cache)

        if ffn_type == "moe":
            x_ffn, router_weights = ffn(rm2(x), is_training=is_training)
            x = x + x_ffn
            return x, router_weights
        else:
            x_ffn = ffn(rm2(x), is_training=is_training)
            x = x + x_ffn
            return x, None

class Transformer(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, idx, targets=None, is_training=True, current_pos=None, use_cache=False):

        if idx.ndim == 1:
            idx = idx[None, :]  # Add batch dim
        B, T = idx.shape
        
        tok_emb = nn.Embed(num_embeddings=self.config['vocab_size'], features=self.config['n_embd'],
                           embedding_init=initializers.normal(stddev=0.02))(idx)
        pos_emb = nn.Embed(num_embeddings=self.config['ctx_len'], features=self.config['n_embd'],
                           embedding_init=initializers.normal(stddev=0.02))(jnp.arange(T)[None, :])  # Add batch dim
        x = tok_emb + pos_emb

        all_router_weights = []
        for i in range(self.config['n_layer']):
            block = Block(config=self.config, index=i)
            x, router_weights = block(x, is_training=is_training, current_pos=current_pos, use_cache=use_cache)
            if router_weights is not None:
                all_router_weights.append(router_weights)

        x = RMSNorm(dim=self.config['n_embd'])(x)
        logits = nn.Dense(features=self.config['vocab_size'],
                          kernel_init=initializers.normal(stddev=0.02),
                          bias_init=initializers.zeros,
                          name='lm_head')(x)

        if targets is None:
            loss = None
        else:
            logits_flat = logits.reshape(-1, self.config['vocab_size'])
            targets_flat = targets.reshape(-1)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat).mean()

        return logits, loss, all_router_weights
    
    def generate(self, idx, max_new_tokens, key):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['ctx_len']:]
            current_pos = idx_cond.shape[1] -1

            # Forward pass, updating the cache.
            logits, _, _ = self(idx_cond, is_training=False, current_pos=current_pos)
            logits = logits[:, -1, :]
            probs = nn.softmax(logits)

            # Sample
            next_token_key, key = jax.random.split(key)  # Generate new key again
            idx_next = jax.random.categorical(next_token_key, logits, axis=-1) # weird choice for multinomial, but works
            idx_next = idx_next.reshape(-1, 1)

            idx = jnp.concatenate([idx, idx_next], axis=1)

        total_size_gb = 0

        def get_size(variables):
            nonlocal total_size_gb

            if 'cache' in variables and 'attn_kv_cache' in variables['cache']:

                k_cache = variables['cache']['attn_kv_cache']['k_recombined_1']
                v_cache = variables['cache']['attn_kv_cache']['v_1']

                size_bytes = jnp.size(k_cache) * k_cache.dtype.itemsize
                size_gb = size_bytes / (1024 ** 3)
                total_size_gb += size_gb

                size_bytes = jnp.size(v_cache) * v_cache.dtype.itemsize
                size_gb = size_bytes / (1024 ** 3)
                total_size_gb += size_gb
                
        dummy_input = jnp.zeros((1, self.config['ctx_len']), dtype=jnp.int32)
        variables = self.apply({}, dummy_input, is_training=False, rngs={'params': key, 'cache' : key, 'noise': key}, mutable=['cache']) # make sure cache is mutable

        get_size(variables)

        return idx, total_size_gb

    def update_expert_biases(self, all_router_weights, update_rate, params):
      """Updates expert biases based on accumulated router weights."""
      with jax.disable_jit():  # IMPORTANT: modifying parameters directly.
        updated_params = params.copy() # use .copy() instead of .unfreeze(), jax is weirding

        j = 0
        for i in range(self.config['n_layer']):
            if self.config['type'][i] == "moe":
                block_router_weights = all_router_weights[j]
                j+=1
                c_i = block_router_weights[:, 1:].sum(axis=0)
                total_routed_tokens = c_i.sum()
                c_i_bar = total_routed_tokens / (self.config['n_experts'] - 1)
                e_i = c_i - c_i_bar  
                
                current_bias = updated_params[f'Block_{i}']['moe']['expert_bias']

                updated_bias = current_bias + update_rate * jnp.sign(e_i)
                updated_params[f'Block_{i}']['moe']['expert_bias'] = updated_bias

        return updated_params

    def estimate_mfu(self, params, fwdbwd_per_iter, dt):
        """Model Flops Utilization calculation (JAX version)"""
        N = sum(p.size for p in jax.tree_util.tree_leaves(params))
        L, H = self.config['n_layer'], self.config['n_head']
        Q = self.config['n_embd'] // self.config['n_head']
        T = self.config['ctx_len']
        
        # Base transformer FLOPs calculation, Adjust for MoE eventually
        flops_per_token = 6*N + 12*L*H*Q*T
         
        if 'moe' in self.config['type']:
            flops_per_token += 2 * self.config['n_experts'] * self.config['n_embd'] * 4
        
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 65e12  # 65 TFLOPS for a T4
        mfu = flops_achieved / flops_promised
        return mfu, flops_achieved
