"""
RaBitQ - Randomized Binary Quantization
Correct implementation based on official C++ code
"""
import numpy as np
from typing import Tuple
from scipy.stats import ortho_group


class RaBitQ:
    """RaBitQ quantization - correct implementation"""
    
    def __init__(self, dim: int, bq: int = 4):
        """
        Args:
            dim: Vector dimensionality
            bq: Bits per dimension (1, 2, 4, or 8)
        """
        self.dim = dim
        self.bq = bq
        self.n_levels = 2 ** bq  # Number of quantization levels
        
        # Factors computed during build
        self.centroid = None
        self.f_add = None
        self.f_rescale = None
        self.f_error = None
        self.scale = None  # Quantization scale for multi-bit
        
    def build(self, data: np.ndarray):
        """Build index with multi-bit quantization"""
        n, dim = data.shape
        assert dim == self.dim
        
        # Step 1: Compute centroid
        self.centroid = np.mean(data, axis=0).astype(np.float32)
        
        # Step 2: Compute residuals
        residuals = data - self.centroid
        
        # Step 3: Quantize residuals
        if self.bq == 1:
            # 1-bit: just sign
            codes = (residuals > 0).astype(np.uint8)
            cb = -0.5
            self.res_min = None
            self.scale = None
        else:
            # Multi-bit: uniform quantization (global min/max)
            # Use global quantization for compatibility with distance formula
            res_min = residuals.min()
            res_max = residuals.max()
            self.res_min = res_min
            self.scale = (res_max - res_min) / (self.n_levels - 1)
            if self.scale == 0:
                self.scale = 1
            
            # Quantize to [0, n_levels-1]
            codes = np.clip(
                np.round((residuals - res_min) / self.scale),
                0, self.n_levels - 1
            ).astype(np.uint8)
            
            # Center: cb shifts codes to be centered around 0
            cb = -(self.n_levels - 1) / 2.0
        
        # Step 4: Compute xu_cb and factors
        xu_cb = codes.astype(np.float32) + cb
        
        # Norms
        l2_sqr = np.sum(residuals ** 2, axis=1)
        l2_norm = np.sqrt(l2_sqr)
        
        # Inner products
        ip_resi_xucb = np.sum(residuals * xu_cb, axis=1)
        ip_cent_xucb = np.sum(self.centroid * xu_cb, axis=1)
        
        # Avoid division by zero
        ip_resi_xucb = np.where(ip_resi_xucb == 0, np.inf, ip_resi_xucb)
        
        # Factors for L2 distance
        self.f_add = l2_sqr + (2 * l2_sqr * ip_cent_xucb / ip_resi_xucb)
        self.f_rescale = -2 * l2_sqr / ip_resi_xucb
        
        # Error factor
        xu_cb_norm_sq = np.sum(xu_cb ** 2, axis=1)
        tmp = (l2_sqr * xu_cb_norm_sq) / (ip_resi_xucb ** 2) - 1
        tmp = np.maximum(tmp, 0)
        self.f_error = 2 * l2_norm * 1.9 * np.sqrt(tmp / (dim - 1))
        
        return codes
    
    def search(self, query: np.ndarray, codes: np.ndarray, data: np.ndarray, k: int = 10):
        """Search with multi-bit quantization support"""
        # Step 1: Compute query residual
        q_residual = query - self.centroid
        
        # Step 2: Compute cb based on bits
        if self.bq == 1:
            cb = -0.5
        else:
            cb = -(self.n_levels - 1) / 2.0
        
        # Step 3: Compute ip_x0_qr
        ip_codes_qres = np.sum(codes * q_residual, axis=1).astype(np.float32)
        sumq = np.sum(q_residual)
        ip_x0_qr = ip_codes_qres + cb * sumq
        
        # Step 4: Compute G_add
        G_add = np.sum(q_residual ** 2)
        
        # Step 5: Estimate distances
        estimated_dist = self.f_add + G_add + self.f_rescale * ip_x0_qr
        estimated_dist = np.maximum(estimated_dist, 0)
        
        # Step 6: Get top-k
        k = min(k, len(estimated_dist))
        if k == 0:
            return np.array([]), np.array([])
        
        if k >= len(estimated_dist) - 1:
            indices = np.argsort(estimated_dist)[:k]
        else:
            indices = np.argpartition(estimated_dist, k)[:k]
            indices = indices[np.argsort(estimated_dist[indices])]
        
        return estimated_dist[indices], indices
