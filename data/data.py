import os
import glob
import pickle
import numpy as np # no jnp, immutability and tqdm annoying af
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

shard_size = 1000000 # 1m tokens per shard may push to 10m, must test out performance
data_dir = "data"
batch_size = 64

dataset_folder = "shakespeare"
dataset_path = dataset_folder
DATA_CACHE_DIR = dataset_path # reuse real quick

if not os.path.exists(dataset_path):
    print(f"Error: Dataset folder not found at {dataset_path}")
    exit()

input_files = glob.glob(os.path.join(dataset_path, "*.txt"))
if not input_files:
    print(f"Error: No .txt files found in {dataset_path}")
    exit()

# chars in test, BPE support soon
chars = set()
total_tokens = 0
for file_path in input_files:
    with open(file_path, 'r') as f:
        for line in f:
            chars.update(line)
            total_tokens += len(line)
chars = sorted(list(chars))
vocab_size = len(chars)

# mappings
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return ''.join([itos[i] for i in l])

# Sharding logic with mp
def write_shard(shard_data, split_name, shard_index, dataset_folder, data_cache_dir):
    filename = os.path.join(data_cache_dir, f"{dataset_folder}_{split_name}_{shard_index:06d}.bin")
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # header num
    header[1] = 1 # data format version
    header[2] = len(shard_data)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(shard_data.tobytes())
    return filename # might use for better progress tracks

def process_file_and_shard(file_path, dataset_folder, data_cache_dir, shard_size, stoi, split_ratio=0.9):
    train_shard_buffer = np.empty(shard_size, dtype=np.uint16)
    val_shard_buffer = np.empty(shard_size, dtype=np.uint16)
    train_shard_count = 0
    val_shard_count = 0
    train_token_count = 0
    val_token_count = 0

    file_tokens = []
    with open(file_path, 'r') as f:
        for line in f:
            file_tokens.extend(encode(line))

    file_token_array = np.array(file_tokens, dtype=np.uint16)
    num_file_tokens = len(file_token_array)
    train_split_index = int(num_file_tokens * split_ratio)

    train_ids_file = file_token_array[:train_split_index]
    val_ids_file = file_token_array[train_split_index:]

    for token in train_ids_file:
        if train_token_count < shard_size:
            train_shard_buffer[train_token_count] = token
            train_token_count += 1
        else:
            filename = write_shard(train_shard_buffer, "train", train_shard_count, dataset_folder, data_cache_dir)
            train_shard_count += 1
            train_token_count = 0
            train_shard_buffer = np.empty(shard_size, dtype=np.uint16)
            train_shard_buffer[train_token_count] = token
            train_token_count += 1

    for token in val_ids_file:
        if val_token_count < shard_size:
            val_shard_buffer[val_token_count] = token
            val_token_count += 1
        else:
            filename = write_shard(val_shard_buffer, "val", val_shard_count, dataset_folder, data_cache_dir)
            val_shard_count += 1
            val_token_count = 0
            val_shard_buffer = np.empty(shard_size, dtype=np.uint16)
            val_shard_buffer[val_token_count] = token
            val_token_count += 1

    # straggler shard
    
    if train_token_count > 0:
        write_shard(train_shard_buffer[:train_token_count], "train", train_shard_count, dataset_folder, data_cache_dir)

    if val_token_count > 0:
        write_shard(val_shard_buffer[:val_token_count], "val", val_shard_count, dataset_folder, data_cache_dir)

    return True


if __name__ == '__main__': # Required for multiprocessing on Windows

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    print("Processing and sharding dataset using multiprocessing...")
    num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes.")

    process_file_partial = partial(process_file_and_shard,
                                    dataset_folder=dataset_folder,
                                    data_cache_dir=DATA_CACHE_DIR,
                                    shard_size=shard_size,
                                    stoi=stoi)

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_file_partial, input_files), total=len(input_files), desc="Processing files"))

    train_shard_files = glob.glob(os.path.join(DATA_CACHE_DIR, f"{dataset_folder}_train_*.bin"))
    val_shard_files = glob.glob(os.path.join(DATA_CACHE_DIR, f"{dataset_folder}_val_*.bin"))

    print(f"Sharded training data into {len(train_shard_files)} files.")
    print(f"Sharded validation data into {len(val_shard_files)} files.")

    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'vocab_source': 'char' # or 'bpe' or 'word' when added, also maybe monster_tok
    }

    meta_pkl_path = os.path.join(DATA_CACHE_DIR, 'meta.pkl')
    with open(meta_pkl_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"Dataset files and meta.pkl created in: {DATA_CACHE_DIR}")

    # auto dump to gitignore

    gitignore_path = '../../.gitignore'
    dataset_gitignore_entry = f"{data_dir}/{dataset_folder}/\n"

    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.readlines()

        found_dataset_entry = False
        for line in gitignore_content:
            if line.strip() == dataset_gitignore_entry.strip():
                found_dataset_entry = True
                break

        if not found_dataset_entry:
            with open(gitignore_path, 'a') as f:
                f.write(dataset_gitignore_entry)
                print(f"Added '{dataset_gitignore_entry.strip()}' to .gitignore")
        else:
            print(f"'{dataset_gitignore_entry.strip()}' already in .gitignore")

    else:
        with open(gitignore_path, 'w') as f:
            f.write(dataset_gitignore_entry)
            print(f".gitignore created and added '{dataset_gitignore_entry.strip()}'")
