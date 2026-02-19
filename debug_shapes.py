"""Debug shapes"""
import numpy as np
import sys
sys.path.insert(0, '.')
from src.quantization.rabitq_numba import RaBitQNumba

# Create sample data
data = np.random.randn(100, 768).astype(np.float32)

rabitq = RaBitQNumba(dim=768, bq=1, metric='IP')
codes = rabitq.build(data)

print(f"codes shape: {codes.shape}")
print(f"centroid shape: {rabitq.centroid.shape}")
print(f"f_add shape: {rabitq.f_add.shape}")
print(f"f_rescale shape: {rabitq.f_rescale.shape}")
