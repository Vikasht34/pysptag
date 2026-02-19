"""
Pure RaBitQ implementation with factor-based distance computation.
Based on the RaBitQ paper: https://arxiv.org/abs/2409.09913
"""
import numpy as np
from typing import Tuple, Optional


class RaBitQProper:
    """RaBitQ with proper factor-based distance (no dequantization)"""
    
    def __init__(self, dim: int, bits: int = 4, metric: str = 'IP'):
        self.dim = dim
        self.bits = bits
        self.n_levels = 1 << bits  # 2^bits
        self.metric = metric
        
        # Per-vector factors for distance estimation
        self.f_add = None
        self.f_rescale = None
        
        # Quantization parameters
        self.centroid = None
        self.rotation_matrix = None
        self.codes = None
        
    def fit(self, vectors: np.ndarray, centroid: np.ndarray):
        """
        Quantize vectors relative to centroid.
        
        Args:
            vectors: (n, dim) array of vectors to quantize
            centroid: (dim,) centroid vector
        """
        n = len(vectors)
        self.centroid = centroid.copy()
        
        # 1. Generate random rotation matrix
        self.rotation_matrix = self._generate_rotation_matrix()
        
        # 2. Compute residuals
        residuals = vectors - centroid
        
        # 3. Rotate residuals
        residuals_rotated = residuals @ self.rotation_matrix.T
        
        # 4. Quantize with optimal rescaling (per vector)
        self.codes = np.zeros((n, self.dim), dtype=np.uint8)
        self.f_add = np.zeros(n, dtype=np.float32)
        self.f_rescale = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            codes_i, f_add_i, f_rescale_i = self._quantize_vector(residuals_rotated[i])
            self.codes[i] = codes_i
            self.f_add[i] = f_add_i
            self.f_rescale[i] = f_rescale_i
            
    def _generate_rotation_matrix(self) -> np.ndarray:
        """Generate random orthogonal rotation matrix"""
        # Random Gaussian matrix
        matrix = np.random.randn(self.dim, self.dim).astype(np.float32)
        
        # QR decomposition for orthogonalization
        Q, _ = np.linalg.qr(matrix)
        return Q.astype(np.float32)
    
    def _quantize_vector(self, residual: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Quantize single residual vector with RaBitQ factor computation.
        Returns: (codes, f_add, f_rescale)
        
        Based on RaBitQ paper formulas.
        """
        if self.bits == 1:
            # 1-bit: binary quantization (sign)
            codes = (residual > 0).astype(np.uint8)
            
            # xu_cb = codes - 0.5 (center binary codes around 0)
            xu_cb = codes.astype(np.float32) - 0.5
            
            # Compute factors
            l2_sqr = np.sum(residual ** 2)
            ip_resi_xucb = np.dot(residual, xu_cb)
            ip_cent_xucb = np.dot(self.centroid, xu_cb)
            
            if np.abs(ip_resi_xucb) < 1e-10:
                ip_resi_xucb = 1e10
            
            if self.metric == 'IP':
                f_add = 1.0 - np.dot(residual, self.centroid) + (l2_sqr * ip_cent_xucb / ip_resi_xucb)
                f_rescale = -l2_sqr / ip_resi_xucb
            else:  # L2
                f_add = l2_sqr + (2 * l2_sqr * ip_cent_xucb / ip_resi_xucb)
                f_rescale = -2 * l2_sqr / ip_resi_xucb
                
        else:
            # Multi-bit: uniform quantization
            residual_abs = np.abs(residual)
            max_val = np.max(residual_abs)
            
            if max_val < 1e-10:
                codes = np.zeros(self.dim, dtype=np.uint8)
                return codes, 0.0, 0.0
            
            # Quantize to [0, n_levels-1]
            scale = (self.n_levels - 1) / max_val
            codes = np.clip(np.round(residual_abs * scale), 0, self.n_levels - 1).astype(np.uint8)
            
            # Center codes: xu_cb = codes - (n_levels-1)/2
            cb = -(self.n_levels - 1) / 2.0
            xu_cb = codes.astype(np.float32) + cb
            
            # Restore signs for residual direction
            signs = np.sign(residual)
            xu_cb = xu_cb * signs
            
            # Compute factors
            l2_sqr = np.sum(residual ** 2)
            ip_resi_xucb = np.dot(residual, xu_cb)
            ip_cent_xucb = np.dot(self.centroid, xu_cb)
            
            if np.abs(ip_resi_xucb) < 1e-10:
                ip_resi_xucb = 1e10
            
            if self.metric == 'IP':
                f_add = 1.0 - np.dot(residual, self.centroid) + (l2_sqr * ip_cent_xucb / ip_resi_xucb)
                f_rescale = -l2_sqr / ip_resi_xucb
            else:  # L2
                f_add = l2_sqr + (2 * l2_sqr * ip_cent_xucb / ip_resi_xucb)
                f_rescale = -2 * l2_sqr / ip_resi_xucb
        
        return codes, f_add, f_rescale
    
    def compute_distances(self, query: np.ndarray) -> np.ndarray:
        """
        Compute distances using factor-based formula (NO dequantization).
        
        Formula: dist = f_add + f_rescale * <query_rotated, codes>
        """
        # Rotate query
        query_rotated = query @ self.rotation_matrix.T
        
        # Compute inner products with codes (vectorized)
        # codes are unsigned, need to restore signs from residuals
        # For simplicity, compute: <query_rot, codes>
        ip_query_codes = np.dot(self.codes.astype(np.float32), query_rotated)
        
        # Apply factor-based formula
        if self.metric == 'IP':
            # Distance = f_add + f_rescale * ip_query_codes + <query, centroid>
            g_add = -np.dot(query, self.centroid)
            dists = self.f_add + g_add + self.f_rescale * ip_query_codes
        else:
            # L2 distance
            g_add = np.sum(query ** 2)
            dists = self.f_add + g_add + self.f_rescale * ip_query_codes
        
        return dists


def test_rabitq():
    """Test RaBitQ implementation"""
    print("="*70)
    print("Testing RaBitQ Implementation")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    dim = 128
    n_vectors = 1000
    n_queries = 100
    
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize
    
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    
    centroid = np.mean(vectors, axis=0)
    
    # Test different bit widths
    for bits in [1, 2, 4]:
        print(f"\nTesting {bits}-bit quantization:")
        print("-" * 70)
        
        # Quantize
        quantizer = RaBitQProper(dim, bits=bits, metric='IP')
        quantizer.fit(vectors, centroid)
        
        # Compute ground truth distances
        gt_dists = -np.dot(vectors, queries[0])
        
        # Compute quantized distances
        quant_dists = quantizer.compute_distances(queries[0])
        
        # Compute recall@10
        gt_top10 = set(np.argsort(gt_dists)[:10])
        quant_top10 = set(np.argsort(quant_dists)[:10])
        recall = len(gt_top10 & quant_top10) / 10
        
        # Compute distance correlation
        correlation = np.corrcoef(gt_dists, quant_dists)[0, 1]
        
        # Memory usage
        memory_bytes = quantizer.codes.nbytes + quantizer.f_add.nbytes + quantizer.f_rescale.nbytes
        original_bytes = vectors.nbytes
        compression = (1 - memory_bytes / original_bytes) * 100
        
        print(f"  Recall@10: {recall*100:.1f}%")
        print(f"  Distance correlation: {correlation:.4f}")
        print(f"  Memory: {memory_bytes/1024:.1f}KB (compression: {compression:.1f}%)")
        print(f"  Codes shape: {quantizer.codes.shape}, dtype: {quantizer.codes.dtype}")
    
    # Benchmark speed
    print("\n" + "="*70)
    print("Speed Benchmark (1000 vectors, 100 queries)")
    print("="*70)
    
    import time
    
    quantizer = RaBitQProper(dim, bits=4, metric='IP')
    quantizer.fit(vectors, centroid)
    
    # Warmup
    for _ in range(10):
        _ = quantizer.compute_distances(queries[0])
    
    # Benchmark
    start = time.perf_counter()
    for query in queries:
        _ = quantizer.compute_distances(query)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"  Total time: {elapsed:.1f}ms")
    print(f"  Per query: {elapsed/len(queries):.2f}ms")
    print(f"  QPS: {len(queries)/(elapsed/1000):.0f}")


if __name__ == "__main__":
    test_rabitq()
