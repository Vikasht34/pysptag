"""Profile search to identify bottlenecks"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from src.index.spann_rabitq_replica import SPANNRaBitQReplica

# Small test
np.random.seed(42)
data = np.random.randn(100000, 128).astype(np.float32)
queries = np.random.randn(100, 128).astype(np.float32)

print("Building index...")
index = SPANNRaBitQReplica(dim=128, target_posting_size=500, replica_count=4, bq=2, metric='L2')
index.build(data)

print("\nProfiling search components...")

# Profile individual components
import cProfile
import pstats
from io import StringIO

pr = cProfile.Profile()
pr.enable()

for query in queries:
    dists, indices = index.search(query, data, k=10, search_internal_result_num=10, max_check=200)

pr.disable()

# Print stats
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())
