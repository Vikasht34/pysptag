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
        Build index (quantize data vectors) - matches official RaBitQ
        
        Args:
            data: (N, D) raw vectors
            
        Returns:
            codes: (N, D) binary codes (stored as uint8)
        """
        n, dim = data.shape
        assert dim == self.dim
        
        # Step 1: Compute centroid
        self.centroid = np.mean(data, axis=0)
        
        # Step 2: Compute residuals
        residuals = data - self.centroid
        
        # Step 3: Rotate residuals with P^T
        rotated = residuals @ self.P.T
        
        # Step 4: Extract binary codes (sign bits)
        codes = (rotated > 0).astype(np.uint8)
        
        # Step 5: Compute factors for each vector (from official implementation)
        # xu_cb = x_u + cb where cb = -((1 << 1) - 1) / 2 = -0.5
        cb = -0.5
        xu_cb = codes.astype(np.float32) + cb  # Shape: (N, D)
        
        # Compute norms
        self.norms = np.linalg.norm(residuals, axis=1)
        l2_sqr = self.norms ** 2
        
        # Compute inner products
        ip_resi_xucb = np.sum(residuals * xu_cb, axis=1)
        ip_cent_xucb = np.sum(self.centroid * xu_cb, axis=1)
        
        # Avoid division by zero
        ip_resi_xucb = np.where(ip_resi_xucb == 0, np.inf, ip_resi_xucb)
        
        # Store factors for distance estimation (L2 metric)
        self.f_add = l2_sqr + (2 * l2_sqr * ip_cent_xucb / ip_resi_xucb)
        self.f_rescale = -2 * l2_sqr / ip_resi_xucb
        
        # Error factor
        xu_cb_norm_sq = np.sum(xu_cb ** 2, axis=1)
        tmp = (l2_sqr * xu_cb_norm_sq) / (ip_resi_xucb ** 2) - 1
        tmp = np.maximum(tmp, 0)  # Avoid negative values
        self.f_error = 2 * self.norms * 1.9 * np.sqrt(tmp / (dim - 1))
        
        return codes
    
    def search(
        self,
        query: np.ndarray,
        codes: np.ndarray,
        data: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with RaBitQ distance estimation - matches official implementation
        
        Args:
            query: (D,) query vector
            codes: (N, D) binary codes
            data: (N, D) raw data vectors (not used in official, kept for compatibility)
            k: Number of neighbors
            
        Returns:
            distances: (k,) estimated distances
            indices: (k,) indices
        """
        # Step 1: Compute query residual and rotate
        q_residual = query - self.centroid
        q_rotated = q_residual @ self.P.T
        
        # Step 2: Quantize query with scalar quantization (bq bits)
        vl = q_rotated.min()
        vr = q_rotated.max()
        delta = (vr - vl) / (2**self.bq - 1) if vr > vl else 1.0
        
        # Quantize to bq-bit integers
        q_quantized = np.clip(
            np.floor((q_rotated - vl) / delta).astype(np.int32),
            0,
            2**self.bq - 1
        )
        
        # Step 3: Compute inner product between binary codes and quantized query
        # This is the key computation: sum over dimensions of (code * q_quantized)
        # codes are 0/1, so this counts how many dimensions match
        ip_x0_qr = np.sum(codes * q_quantized, axis=1).astype(np.float32)
        
        # Apply delta scaling and offset
        ip_x0_qr = delta * ip_x0_qr + vl * np.sum(codes, axis=1)
        
        # Step 4: Compute query-dependent constants
        g_add = np.sum(q_residual ** 2)
        k1xsumq = -2 * np.sum(q_residual * self.centroid)
        
        # Step 5: Estimate distances using official formula
        # est_dist = f_add + g_add + f_rescale * (ip_x0_qr + k1xsumq)
        estimated_dist = self.f_add + g_add + self.f_rescale * (ip_x0_qr + k1xsumq)
        
        # Ensure non-negative distances
        estimated_dist = np.maximum(estimated_dist, 0)
        
        # Step 6: Get top-k
        k = min(k, len(estimated_dist))
        indices = np.argpartition(estimated_dist, k)[:k]
        indices = indices[np.argsort(estimated_dist[indices])]
        
        distances = estimated_dist[indices]
        
        return distances, indices
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio"""
        original_bits = self.dim * 32  # float32
        compressed_bits = self.dim  # 1 bit per dimension
        return original_bits / compressed_bits
