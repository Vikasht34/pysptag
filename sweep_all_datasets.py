"""
Comprehensive parameter sweep for all datasets
Run in parallel with separate log files
"""
import numpy as np
import time
import sys
import os
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

# Dataset configurations
DATASETS = {
    'cohere': {
        'data_dir': '/Users/viktari/cohere_data',
        'base_file': 'cohere_base.bin',
        'query_file': 'cohere_query.bin',
        'gt_file': 'cohere_groundtruth.bin',
        'dim': 768,
        'metric': 'IP',
        'gt_skip': 2,  # Skip first 2 values (metadata)
        'gt_k': 100,
        'dtype': np.float32
    },
    'gist': {
        'data_dir': '/Users/viktari/pysptag/data/gist',
        'base_file': 'base.bin',
        'query_file': 'query.bin',
        'gt_file': 'groundtruth.bin',
        'dim': 960,
        'metric': 'L2',
        'gt_skip': 2,
        'gt_k': 100,
        'dtype': np.float32
    },
    'sift': {
        'data_dir': '/Users/viktari/pysptag/data/sift',
        'base_file': 'sift_base.fvecs',
        'query_file': 'sift_query.fvecs',
        'gt_file': 'sift_groundtruth.ivecs',
        'dim': 128,
        'metric': 'L2',
        'gt_skip': 0,
        'gt_k': 100,
        'dtype': np.float32
    },
    'glove': {
        'data_file': '/Users/viktari/Downloads/glove-200-angular.hdf5',
        'dim': 200,
        'metric': 'Cosine',  # Use 'Cosine' not 'COSINE' for compatibility
        'use_hdf5': True
    }
}

def load_fvecs(filename):
    """Load .fvecs format"""
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        vec_size = data
        data = np.fromfile(f, dtype=np.float32)
        dim = vec_size
        return data.reshape(-1, dim + 1)[:, 1:]

def load_ivecs(filename):
    """Load .ivecs format"""
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        vec_size = data
        data = np.fromfile(f, dtype=np.int32)
        dim = vec_size
        return data.reshape(-1, dim + 1)[:, 1:]

def run_sweep(dataset_name):
    config = DATASETS[dataset_name]
    log_file = f'sweep_{dataset_name}.log'
    
    with open(log_file, 'w') as log:
        log.write(f"{'='*80}\n")
        log.write(f"{dataset_name.upper()} - Parameter Sweep - Recall@100\n")
        log.write(f"{'='*80}\n\n")
        log.flush()
        
        # Load data
        log.write("Loading data...\n")
        log.flush()
        
        if config.get('use_hdf5'):
            import h5py
            with h5py.File(config['data_file'], 'r') as f:
                base = f['train'][:]
                queries = f['test'][:]
                groundtruth = f['neighbors'][:]
        elif config['base_file'].endswith('.fvecs'):
            base = load_fvecs(f"{config['data_dir']}/{config['base_file']}")
            queries = load_fvecs(f"{config['data_dir']}/{config['query_file']}")
        else:
            base = np.fromfile(f"{config['data_dir']}/{config['base_file']}", dtype=config['dtype'])
            base = base[:len(base) // config['dim'] * config['dim']].reshape(-1, config['dim'])
            queries = np.fromfile(f"{config['data_dir']}/{config['query_file']}", dtype=config['dtype'])
            queries = queries[:len(queries) // config['dim'] * config['dim']].reshape(-1, config['dim'])
        
        if not config.get('use_hdf5'):
            if config['gt_file'].endswith('.ivecs'):
                groundtruth = load_ivecs(f"{config['data_dir']}/{config['gt_file']}")
            else:
                gt_raw = np.fromfile(f"{config['data_dir']}/{config['gt_file']}", dtype=np.int32)
                if config['gt_skip'] > 0:
                    gt_raw = gt_raw[config['gt_skip']:]
                groundtruth = gt_raw[:len(gt_raw) // config['gt_k'] * config['gt_k']].reshape(-1, config['gt_k'])
            
            # Only use queries that have groundtruth
            num_queries_with_gt = len(groundtruth)
            queries = queries[:num_queries_with_gt]
        
        log.write(f"✓ Base: {base.shape}, Queries: {queries.shape}, GT: {groundtruth.shape}\n\n")
        log.flush()
        
        # Normalize for COSINE metric (must match what was done during build)
        if config['metric'] in ('COSINE', 'Cosine'):
            log.write("Normalizing vectors for COSINE metric...\n")
            base = base / np.linalg.norm(base, axis=1, keepdims=True)
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
            log.flush()
        
        # Build or load index
        index_dir = f"/tmp/{dataset_name}_index"
        metadata_file = os.path.join(index_dir, 'metadata.pkl')
        
        if os.path.exists(metadata_file):
            log.write(f"Loading existing index from {index_dir}...\n")
            log.flush()
            index = SPANNDiskOptimized(
                dim=config['dim'],
                use_rabitq=False,
                metric=config['metric'],
                disk_path=index_dir,
                cache_size=2000
            )
            import pickle
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                for k, v in metadata.items():
                    if k not in ['use_faiss_centroids', '_centroid_index', '_shared_rabitq', '_hnsw_index']:
                        setattr(index, k, v)
            
            # Load HNSW index if it was used
            if metadata.get('use_hnsw_centroids', False):
                import hnswlib
                hnsw_path = os.path.join(index_dir, 'hnsw_centroids.bin')
                if os.path.exists(hnsw_path):
                    if index.metric == 'L2':
                        space = 'l2'
                    elif index.metric in ('IP', 'Cosine'):
                        space = 'ip'
                    else:
                        space = 'l2'
                    
                    index._hnsw_index = hnswlib.Index(space=space, dim=index.dim)
                    index._hnsw_index.load_index(hnsw_path, max_elements=index.num_clusters)
                    log.write(f"  Loaded HNSW index (M={metadata.get('hnsw_m', 16)})\n")
            
            log.write(f"✓ Loaded: {index.num_clusters} clusters\n\n")
        else:
            log.write(f"Building index...\n")
            log.flush()
            t0 = time.time()
            index = SPANNDiskOptimized(
                dim=config['dim'],
                target_posting_size=500,
                replica_count=8,
                use_rabitq=False,
                metric=config['metric'],
                use_hnsw_centroids=True,
                hnsw_m=16,
                hnsw_ef_construction=200,
                clustering='hierarchical',
                use_rng_filtering=True,
                use_faiss_centroids=False,
                disk_path=index_dir,
                cache_size=2000
            )
            index.build(base)
            log.write(f"✓ Built in {time.time()-t0:.1f}s ({index.num_clusters} clusters)\n\n")
        log.flush()
        
        # Sample queries - only from queries that have groundtruth
        num_queries = min(len(queries), len(groundtruth))  # Use min of queries and GT
        sample_size = min(1000, num_queries)
        np.random.seed(42)
        sample_indices = np.random.choice(num_queries, sample_size, replace=False)
        sample_queries = queries[sample_indices]
        sample_gt = groundtruth[sample_indices]
        
        log.write(f"Using {sample_size} sampled queries\n\n")
        log.flush()
        
        # Parameter sweep
        centroid_counts = [32, 64, 128, 256, 512]
        max_checks = [2048, 4096, 8192, 16384, 32768, 49152, 1000000]
        
        log.write(f"{'='*80}\n")
        log.write("Results\n")
        log.write(f"{'='*80}\n")
        log.write(f"{'Centroids':>10} {'MaxCheck':>10} {'Recall@100':>12} {'P50(ms)':>10} {'P90(ms)':>10} {'Avg(ms)':>10}\n")
        log.write(f"{'-'*80}\n")
        log.flush()
        
        for num_centroids in centroid_counts:
            for max_check in max_checks:
                recalls = []
                latencies = []
                
                for i, query in enumerate(sample_queries):
                    t0 = time.perf_counter()
                    indices, dists = index.search(
                        query, base, k=100,
                        search_internal_result_num=num_centroids,
                        max_check=max_check
                    )
                    latencies.append((time.perf_counter() - t0) * 1000)
                    
                    gt = set(sample_gt[i][:100])
                    found = set(indices[:100])
                    recalls.append(len(gt & found) / 100)
                
                avg_recall = np.mean(recalls) * 100
                p50 = np.percentile(latencies, 50)
                p90 = np.percentile(latencies, 90)
                avg = np.mean(latencies)
                
                result = f"{num_centroids:>10} {max_check:>10} {avg_recall:>11.2f}% {p50:>10.2f} {p90:>10.2f} {avg:>10.2f}\n"
                log.write(result)
                log.flush()
        
        log.write(f"{'='*80}\n")
        log.write("DONE\n")
        log.flush()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python sweep_all_datasets.py <dataset_name>")
        print(f"Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)
    
    dataset = sys.argv[1]
    if dataset not in DATASETS:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)
    
    run_sweep(dataset)
