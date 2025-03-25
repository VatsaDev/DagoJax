import os
import time
import pickle
import argparse
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import model
from model import Transformer

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--ckpath', type=str, required=True, help='checkpoint path')
parser.add_argument('--data_dir', type=str, required=True, help='data directory')
parser.add_argument('--n_embd', type=int, default=32, help='Embedding dimension')
parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
parser.add_argument('--n_layer', type=int, default=2, help='Number of layers')
parser.add_argument('--n_experts', type=int, default=8, help='Number of experts per MoE layer')
parser.add_argument('--ctx_len', type=int, default=1024, help='Context length')
parser.add_argument('--max_tok', type=int, default=2048, help='Maximum number of tokens to generate')
parser.add_argument('--types', nargs='*', type=str, default=['mlp', 'moe'])
parser.add_argument('--temp', type=float, default=1.0, help='Sampling temperature')
parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
args = parser.parse_args()

# --- Model Configuration ---
model.config.update({
    'device': 'cpu',
    'ctx_len': args.ctx_len,
    'n_embd': args.n_embd,
    'n_head': args.n_head,
    'n_layer': args.n_layer,
    'n_experts': args.n_experts,
    'type': args.types,
    'temp': args.temp,
    'top_k': args.top_k
})

# Convert to absolute path
abs_path = os.path.abspath(args.ckpath)

# --- Load Tokenizer ---
with open(os.path.join(args.data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
model.config['vocab_size'] = len(itos)

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# --- Initialize Model ---
model = Transformer(model.config)
rng = jax.random.PRNGKey(0)

# --- Load Checkpoint ---
def load_checkpoint(ckpt_path):
    dummy_input = jnp.ones((model.config['ctx_len'],), dtype=jnp.int32)
    variables = model.init({'params': rng}, dummy_input)
    checkpoint = checkpoints.restore_checkpoint(
        ckpt_dir=os.path.dirname(ckpt_path),
        target=variables,
        prefix=os.path.basename(ckpt_path).split('_')[0] + '_',
        parallel=True
    )
    return checkpoint['params']

params = load_checkpoint(abs_path)

# --- Optimized Generation Function ---

def generate(params, rng, input_ids, max_tokens):
    tokens = input_ids.copy()
    total_cache_size = 0
    vocab_size = model.config['vocab_size']
    
    for i in range(max_tokens):
        rng, step_rng = jax.random.split(rng)
        outputs, mutable_vars = model.apply(
            {'params': params},
            tokens[None, :],  # Add batch dimension
            is_training=False,
            mutable=['cache']
        )
        
        # Get logits from outputs tuple
        logits = outputs[0][0, -1, :]  # [batch, seq, vocab] -> [vocab]
        logits = logits / args.temp
        
        # Handle top-k filtering safely
        if args.top_k > 0 and args.top_k < vocab_size:
            top_k = min(args.top_k, vocab_size)
            top_logits, _ = jax.lax.top_k(logits, top_k)
            min_top_logit = top_logits[-1]
            logits = jnp.where(logits < min_top_logit, -jnp.inf, logits)
        
        # Sample next token
        next_token = jax.random.categorical(step_rng, logits)
        
        # Update tokens - handle both cases where tokens is 1D or 2D
        if tokens.ndim == 1:
            tokens = jnp.concatenate([tokens[1:], jnp.array([next_token])])
        else:  # If tokens is 2D [1, seq]
            tokens = jnp.concatenate([tokens[:, 1:], jnp.array([[next_token]])], axis=1)
        
        # Calculate KV cache size (only on last step for efficiency)
        if i == max_tokens - 1 and 'cache' in mutable_vars:
            cache_size = sum(
                x.size * x.dtype.itemsize 
                for x in jax.tree_util.tree_leaves(mutable_vars['cache'])
            )
            total_cache_size = cache_size
    
    return tokens, total_cache_size

# --- Run Generation ---
start_ids = encode(" ")
input_ids = jnp.array(start_ids, dtype=jnp.int32)

print("Starting generation...")
start_time = time.time()

# Warmup run (no output)
_, _ = generate(params, rng, input_ids, 2)

# Actual timed run
start_time = time.time()
output_tokens, cache_size = generate(params, rng, input_ids, args.max_tok)
elapsed = time.time() - start_time

generated_text = decode(output_tokens.tolist())

print("\n" + generated_text)
print(f"\nTotal KV cache size: {cache_size/1024:.4f} KB")
print(f"Generated {args.max_tok} tokens in {elapsed:.2f}s ({args.max_tok/elapsed:.2f} tok/s)")
