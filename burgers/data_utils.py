import os
import scipy.io
import numpy as np
import time

from function_diffusion.utils.data_utils import BaseDataset

# Cache to avoid reloading data in the same process
_data_cache = {}

def create_dataset(config):
    path = config.dataset.data_path
    num_train = config.dataset.num_train_samples
    downsample_factor = config.dataset.downsample_factor
    
    # Create cache key based on path, num_train, and downsample_factor
    cache_key = f"{path}_{num_train}_{downsample_factor}"
    
    # Check if data is in cache
    if cache_key in _data_cache:
        print("Loading data from cache (skipping file I/O)...")
        return _data_cache[cache_key]
    
    # Load data (only once per process)
    print(f"Loading data from {path} (this may take a few seconds)...")
    start_time = time.time()
    
    # Load .mat file using scipy (optimized loading)
    data = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    usols = data['output']
    
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f}s, processing...")
    
    # Process data (optimized: ensure contiguous arrays for faster access)
    process_start = time.time()
    
    # Slice and add channel dimension, ensure contiguous for faster indexing
    usols = np.ascontiguousarray(usols[:, :-1, :, None])
    
    # Split train/test (these are views, not copies, but ensure contiguous)
    u_train = np.ascontiguousarray(usols[:num_train])
    u_test = np.ascontiguousarray(usols[num_train:])
    
    # Create datasets with optimized storage
    train_dataset = BaseDataset(u_train, downsample_factor)
    test_dataset = BaseDataset(u_test, downsample_factor)
    
    process_time = time.time() - process_start
    print(f"Data processed in {process_time:.2f}s")
    
    # Cache the datasets to avoid reloading
    _data_cache[cache_key] = (train_dataset, test_dataset)
    
    return train_dataset, test_dataset



