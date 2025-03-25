import os
import time
import glob
import string
import pickle
import random
import argparse
import numpy as np
from pathlib import Path
from contextlib import nullcontext
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from flax.training import train_state, checkpoints
import optax

import model
from model import Transformer

from plot import plot_loss

# Grabs all nesc hyperparameters, slightly duplicative across files, but relativly okay

parser = argparse.ArgumentParser(description='Train an LM')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size per device')
parser.add_argument('--ctx_len', type=int, default=1024, help='Context length')
parser.add_argument('--eval_interval', type=int, default=20, help='Evaluation interval')
parser.add_argument('--grad_accum', type=int, default=4, help='Gradient accumulation steps')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate')
parser.add_argument('--dropout', type=float, default=0.02, help='Dropout rate')
parser.add_argument('--max_iters', type=int, default=200, help='Maximum training iterations')
parser.add_argument('--eval_iters', type=int, default=20, help='Iterations per evaluation')
parser.add_argument('--warmup_iters', type=int, default=10, help='Warmup iterations')
parser.add_argument('--resume', action='store_true', help='Resume training')
parser.add_argument('--res_path', type=str, default="", help='Path to resume checkpoint')
parser.add_argument('--data_dir', type=str, default="shakespeare", help='Dataset directory')
parser.add_argument('--n_embd', type=int, default=16, help='Embedding dimension')
parser.add_argument('--n_head', type=int, default=2, help='Number of attention heads')
parser.add_argument('--n_layer', type=int, default=2, help='Number of layers')
parser.add_argument('--n_experts', type=int, default=32, help='Number of experts in MoE layers')
parser.add_argument('--use_expert_bias', action='store_true', help='Use expert bias in MoE')
parser.add_argument('--types', nargs='*', type=str, default=['mlp','moe','mlp','moe'], help='Layer types sequence')
parser.add_argument('--device', type=str, default="cpu", help='Device to use')
parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
args = parser.parse_args()

# Update model config
model.config.update({
    'ctx_len': args.ctx_len,
    'device': args.device,
    'n_embd': args.n_embd,
    'n_head': args.n_head,
    'n_layer': args.n_layer,
    'n_experts': args.n_experts,
    'type': args.types,
    'use_expert_bias': args.use_expert_bias,
    'dropout': args.dropout,
    'vocab_size': 50304,  # just set it to GPT2 num, meta.pkl updates this
})

# Hyperparameters as vars
batch_size = args.batch_size
block_size = args.ctx_len # technically two words for the same thing, but its convenient 
eval_interval = args.eval_interval
grad_accum_steps = args.grad_accum
lr = args.lr
min_lr = args.min_lr
max_iters = args.max_iters
eval_iters = args.eval_iters
warmup_iters = args.warmup_iters
beta1 = args.beta1
beta2 = args.beta2
weight_decay = args.weight_decay
max_grad_norm = args.max_grad_norm
resume = args.resume
data_dir = args.data_dir
resume_checkpoint = args.res_path

# Data Loading, fixed keys and used vmap for batching

def get_batch(split, key):

    split_filenames = glob.glob(os.path.join("data", f"{data_dir}", f"{data_dir}_{split}_*.bin"))
    if not split_filenames:
        raise FileNotFoundError(f"No {split} shard files found in {data_dir}")

    shard_file = np.random.choice(split_filenames)
    data = np.memmap(shard_file, dtype=np.uint16, mode='r', offset=256*4)
    num_tokens_in_shard = len(data)

    if num_tokens_in_shard <= block_size + 1:
        return get_batch(split, jax.random.split(key, 1)[0])

    ix = jax.random.randint(key, (1,), 0, num_tokens_in_shard - block_size - 1)[0]
    x = jax.lax.dynamic_slice(data, (ix,), (block_size,))
    y = jax.lax.dynamic_slice(data, (ix + 1,), (block_size,))

    return x, y

get_batch_vmapped = jax.vmap(get_batch, in_axes=(None, 0), out_axes=(0, 0))

# Loss Estimation

@jax.jit
def estimate_loss(state, eval_rng_key):

    out = {}
    for split in ['train', 'val']:
        losses = []
        data_keys = jax.random.split(eval_rng_key, eval_iters)
        X, Y = get_batch_vmapped(split, data_keys)

        for k in range(eval_iters):
            eval_rng_key, dropout_rng, noise_rng = jax.random.split(eval_rng_key, 3)
            (logits, loss, _), _ = model.apply(
                {'params': state.params},
                X[k],
                Y[k],
                is_training=False,
                use_cache=False,
                rngs={'dropout': dropout_rng, 'noise': noise_rng},
                mutable=['cache']
            )
            losses.append(loss)

        out[split] = jnp.mean(jnp.array(losses))
    return out

# Model Init 

meta_path = f'data/{data_dir}/meta.pkl'
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    model.config["vocab_size"] = meta['vocab_size']
    print(f"Found vocab_size = {meta['vocab_size']} in {meta_path}")

model = Transformer(model.config)

# Optimizer Setup, AdamW, switch to Muon later

scheduler = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=lr,
    warmup_steps=warmup_iters,
    decay_steps=max_iters - warmup_iters,
    end_value=min_lr
)

optimizer = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adamw(learning_rate=scheduler, b1=beta1, b2=beta2, weight_decay=weight_decay)
)

# Checkpoint Handling

ckpt_dir = os.path.abspath("checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

def save_checkpoint(state, iter_num, train_hist, val_hist, run_name, args):
    """Save checkpoint with all training state."""
    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir,
        target={
            'params': state.params,
            'opt_state': state.opt_state,
            'step': iter_num,
            'train_losses_history': train_hist,
            'val_losses_history': val_hist,
            'args': vars(args)  # Save all arguments as dict
        },
        step=iter_num,
        prefix=f'{run_name}_',
        overwrite=True,
        keep=3
    )

if resume:

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.ones([block_size], jnp.int32)
    variables = model.init({'params': init_rng}, dummy_input)
    
    temp_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )
    
    checkpoint = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=temp_state,  # Provides structure for loading
        prefix=os.path.basename(resume_checkpoint).split('_')[0] + '_',
        parallel=True
    )
    
    if checkpoint is None:
        raise ValueError(f"Failed to load checkpoint from {resume_checkpoint}")
    
    if hasattr(checkpoint, 'train_losses_history') and hasattr(checkpoint, 'val_losses_history'):
        train_losses_history = checkpoint.train_losses_history
        val_losses_history = checkpoint.val_losses_history
    else:
        train_losses_history = []
        val_losses_history = []
        print("No loss histories")
    
    state = checkpoint
    start_iter = state.step + 1 if hasattr(state, 'step') else 0
    run_name = os.path.basename(resume_checkpoint).split('_')[0]
    
    print(f"Resumed training from iteration {start_iter}")

    # rare debug uncomment

    #print(f"Loaded train history length: {len(train_losses_history)}")
    #print(f"Loaded val history length: {len(val_losses_history)}")
    
else:
    rng = jax.random.PRNGKey(0)
    rng, init_rng, dropout_init_rng, noise_init_rng = jax.random.split(rng, 4)
    
    variables = model.init(
        {'params': init_rng, 'dropout': dropout_init_rng, 'noise': noise_init_rng},
        jnp.ones([block_size], jnp.int32)
    )
    
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(variables['params']))
    print(f"Initializing new model with {total_params/1e6:.2f}M parameters")

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )
    
    start_iter = 0
    train_losses_history = []
    val_losses_history = []
    run_name = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    print(f"Starting new training run {run_name}")

@jax.jit
def train_step(state, batch, dropout_rng, noise_rng):
    x, y = batch

    def loss_fn(params):
        (logits, loss, all_router_weights), mutated = state.apply_fn(
            {'params': params},
            x,
            y,
            is_training=True,
            use_cache=False,
            rngs={'dropout': dropout_rng, 'noise': noise_rng},
            mutable=['cache']
        )
        return loss, (all_router_weights, mutated)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (all_router_weights, mutated)), grads = grad_fn(state.params)
    grads = jax.tree_util.tree_map(lambda g: g / grad_accum_steps, grads)
    new_state = state.apply_gradients(grads=grads)
    
    # Update expert biases if enabled
    if model.config['use_expert_bias'] and any(layer_type == "moe" for layer_type in model.config["type"]):
        updated_params = model.update_expert_biases(all_router_weights, 1e-3, new_state.params)
        new_state = new_state.replace(params=updated_params)
    
    return new_state, loss, all_router_weights

# Training Loop
time_s = time.time()
prev_time = time_s

for iter_num in range(start_iter, max_iters + 1):
    
    # jax has a weird obession with keygen ngl

    rng, data_rng = jax.random.split(rng)
    data_keys = jax.random.split(data_rng, batch_size)
    
    loss_accum = 0.0
    all_router_weights_accum = []

    for _ in range(grad_accum_steps):
        rng, dropout_rng, noise_rng = jax.random.split(rng, 3)
        batch = get_batch_vmapped('train', data_keys)
        state, loss, router_weights = train_step(state, batch, dropout_rng, noise_rng)
        loss_accum += loss.item()
        all_router_weights_accum.extend(router_weights)

    train_losses_history.append(loss_accum / grad_accum_steps)

    if iter_num % eval_interval == 0:

        eval_rng, rng = jax.random.split(rng)
        losses = estimate_loss(state, eval_rng)
        val_losses_history.append(losses['val'].item())

        time_n = time.time()
        elapsed = time_n - time_s
        dt = time_n - prev_time
        prev_time = time_n

        mfu, flops_achieved = model.estimate_mfu(
            params=state.params,
            fwdbwd_per_iter=batch_size * grad_accum_steps,
            dt=dt
        )
    
        print(f"step {iter_num}: train loss {losses['train']:.4f}, "
              f"val loss {losses['val']:.4f}, time {elapsed/60:.2f}min, "
              f"mfu {mfu:.2%}, flops {flops_achieved/1e12:.2f} TFLOPS")

        plot_loss(
            train_hist=train_losses_history,
            val_hist=val_losses_history,
            i_eval=eval_interval,
            iter=iter_num,
            run_name=run_name
        )
        
        save_checkpoint(state, iter_num, train_losses_history, val_losses_history, run_name, args)

print('Training completed')
