"""
RaBitQ - Randomized Binary Quantization
Correct implementation based on official C++ code
Supports L2, InnerProduct, and Cosine metrics
"""
import numpy as np
from typing import Tuple, Literal
from scipy.stats import ortho_group

MetricType = Literal['L2', 'IP', 'Cosine']


class RaBitQ:
    """RaBitQ quantization - correct implementation"""
    
    def __init__(self, dim: int, bq: int = 4, metric: MetricType = 'L2'):
        """
        Args:
            dim: Vector dimensionality
            bq: Bits per dimension (1, 2, 4, or 8)
            metric: Distance metric (L2, IP, or Cosine)
        """
        self.dim = dim
        self.bq = bq
        self.metric = metric
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
        
        # Step 3: Quantize RAW residuals (NOT normalized!)
        # This is critical - official RaBitQ quantizes raw residuals
        if self.bq == 1:
            # 1-bit: binary quantization of raw residuals
            codes = (residuals > 0).astype(np.uint8)
        else:
            # Multi-bit: uniform quantization of raw residuals
            res_min = residuals.min()
            res_max = residuals.max()
            self.res_min = res_min
            self.scale = (res_max - res_min) / (self.n_levels - 1)
            if self.scale == 0:
                self.scale = 1
            
            codes = np.clip(
                np.round((residuals - res_min) / self.scale),
                0, self.n_levels - 1
            ).astype(np.uint8)
        
        # Step 4: Compute RaBitQ factors (using RAW residuals!)
        cb = -(self.n_levels - 1) / 2.0
        xu_cb = codes.astype(np.float32) + cb
        
        l2_sqr = np.sum(residuals ** 2, axis=1)
        l2_norm = np.sqrt(l2_sqr)
        
        # Dot products with RAW residuals (not normalized!)
        ip_residual_xucb = np.sum(residuals * xu_cb, axis=1)
        ip_residual_xucb = np.where(ip_residual_xucb == 0, np.inf, ip_residual_xucb)
        
        ip_cent_xucb = np.sum(self.centroid * xu_cb, axis=1)
        
        # Compute factors (same formula for all bit widths!)
        if self.metric == 'L2':
            self.f_add = l2_sqr + 2 * l2_sqr * ip_cent_xucb / ip_residual_xucb
            self.f_rescale = -2 * l2_sqr / ip_residual_xucb
        elif self.metric in ('IP', 'Cosine'):
            ip_residual_c = np.sum(residuals * self.centroid, axis=1)
            self.f_add = -ip_residual_c + l2_sqr * ip_cent_xucb / ip_residual_xucb
            self.f_rescale = -l2_sqr / ip_residual_xucb
        
        return codes
    
    def search(self, query: np.ndarray, codes: np.ndarray, data: np.ndarray, k: int = 10):
        """Search with RaBitQ formula (works for all bit widths!)"""
        q_residual = query - self.centroid
        
        # Compute c_B for current bit width
        cb = -(self.n_levels - 1) / 2.0
        
        # Compute inner product between codes and query residual
        ip_codes_qres = np.sum(codes * q_residual, axis=1).astype(np.float32)
        sumq = np.sum(q_residual)
        ip_x0_qr = ip_codes_qres + cb * sumq
        
        # Metric-specific G_add computation
        if self.metric == 'L2':
            # L2: G_add = ||q - c||^2
            G_add = np.sum(q_residual ** 2)
        elif self.metric in ('IP', 'Cosine'):
            # IP/Cosine: G_add = -<q, c>
            G_add = -np.dot(query, self.centroid)
        
        # RaBitQ distance estimation (same formula for 1-bit, 2-bit, 4-bit!)
        estimated_dist = self.f_add + G_add + self.f_rescale * ip_x0_qr
        estimated_dist = np.maximum(estimated_dist, 0)
        
        # Get top-k
        k = min(k, len(estimated_dist))
        if k == 0:
            return np.array([]), np.array([])
        
        if k >= len(estimated_dist) - 1:
            indices = np.argsort(estimated_dist)[:k]
        else:
            indices = np.argpartition(estimated_dist, k)[:k]
            indices = indices[np.argsort(estimated_dist[indices])]
        
        return estimated_dist[indices], indices
