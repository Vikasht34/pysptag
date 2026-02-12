"""
RaBitQ - Randomized Binary Quantization
Exact implementation from paper: https://arxiv.org/abs/2405.12497
"""
import numpy as np
from typing import Tuple
from scipy.stats import ortho_group


class RaBitQ:
    """
    RaBitQ quantization - exact paper implementation
    
    Key features:
    - D-dimensional vectors → D-bit codes
    - Unbiased distance estimation
    - O(1/√D) error bound
    - 32× compression
    """
    
    def __init__(self, dim: int, bq: int = 4):
        """
        Args:
            dim: Vector dimensionality
            bq: Query quantization bits (default 4, paper uses 4)
        """
        self.dim = dim
        self.bq = bq
        
        # Random orthogonal matrix P
        self.P = ortho_group.rvs(dim).astype(np.float32)
        
        # Pre-computed values
        self.centroid = None
        self.norms = None  # ||o_r - c||
        self.inner_products = None  # <o_bar, o>
        
    def build(self, data: np.ndarray):
        """
        Build index (quantize data vectors)
        
        Args:
            data: (N, D) raw vectors
            
        Returns:
            codes: (N, D) binary codes (stored as uint8)
        """
        n, dim = data.shape
        assert dim == self.dim
        
        # Step 1: Normalize with centroid
        self.centroid = np.mean(data, axis=0)
        centered = data - self.centroid
        self.norms = np.linalg.norm(centered, axis=1)
        normalized = centered / self.norms[:, None]
        
        # Step 2: Inverse transform with P^-1
        transformed = normalized @ self.P.T  # P^-1 = P^T for orthogonal
        
        # Step 3: Extract sign bits (quantization codes)
        codes = (transformed > 0).astype(np.uint8)
        
        # Step 4: Pre-compute <o_bar, o>
        # Reconstruct o_bar from codes
        x_bar = (2 * codes - 1) / np.sqrt(dim)  # ±1/√D
        o_bar = x_bar @ self.P  # P @ x_bar
        self.inner_products = np.sum(o_bar * normalized, axis=1)
        
        return codes
    
    def search(
        self,
        query: np.ndarray,
        codes: np.ndarray,
        data: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with RaBitQ distance estimation
        
        Args:
            query: (D,) query vector
            codes: (N, D) binary codes
            data: (N, D) raw data vectors
            k: Number of neighbors
            
        Returns:
            distances: (k,) estimated distances
            indices: (k,) indices
        """
        # Step 1: Normalize query
        q_centered = query - self.centroid
        q_norm = np.linalg.norm(q_centered)
        q_normalized = q_centered / q_norm
        
        # Step 2: Inverse transform query
        q_prime = q_normalized @ self.P.T
        
        # Step 3: Quantize query (randomized uniform scalar quantization)
        vl, vr = q_prime.min(), q_prime.max()
        delta = (vr - vl) / (2**self.bq - 1)
        
        # Randomized quantization (Equation 18 in paper)
        u = np.random.uniform(0, 1, self.dim)
        q_bar_u = np.floor((q_prime - vl) / delta + u).astype(np.int32)
        q_bar_u = np.clip(q_bar_u, 0, 2**self.bq - 1)
        
        # Step 4: Compute <x_bar, q_bar> efficiently (Equation 20)
        # Convert codes to ±1/√D
        x_bar = (2 * codes - 1) / np.sqrt(self.dim)
        
        # Bitwise computation (Equation 22)
        inner_xbar_qbar = np.zeros(len(codes))
        for j in range(self.bq):
            # Extract j-th bit of q_bar_u
            q_bar_u_j = (q_bar_u >> j) & 1
            # Compute <x_bar_b, q_bar_u^(j)>
            inner_xbar_qbar += (2**j) * np.sum(codes * q_bar_u_j, axis=1)
        
        # Complete computation (Equation 20)
        sum_xbar_b = np.sum(codes, axis=1)
        sum_qbar_u = np.sum(q_bar_u)
        
        inner_xbar_qbar = (
            2 * delta / np.sqrt(self.dim) * inner_xbar_qbar +
            2 * vl / np.sqrt(self.dim) * sum_xbar_b -
            delta / np.sqrt(self.dim) * sum_qbar_u -
            np.sqrt(self.dim) * vl
        )
        
        # Step 5: Compute unbiased estimator (Equation 13)
        estimated_inner = inner_xbar_qbar / self.inner_products
        
        # Step 6: Convert to distance (Equation 2)
        estimated_dist_sq = (
            self.norms**2 + q_norm**2 -
            2 * self.norms * q_norm * estimated_inner
        )
        
        # Get top-k
        indices = np.argpartition(estimated_dist_sq, k)[:k]
        indices = indices[np.argsort(estimated_dist_sq[indices])]
        
        distances = np.sqrt(estimated_dist_sq[indices])
        
        return distances, indices
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio"""
        original_bits = self.dim * 32  # float32
        compressed_bits = self.dim  # 1 bit per dimension
        return original_bits / compressed_bits
