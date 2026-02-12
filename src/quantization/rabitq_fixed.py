"""
RaBitQ - Simplified correct implementation based on official C++ code
"""
import numpy as np
from typing import Tuple
from scipy.stats import ortho_group


class RaBitQFixed:
    """RaBitQ quantization - simplified but correct"""
    
    def __init__(self, dim: int):
        self.dim = dim
        # Random orthogonal matrix P
        self.P = ortho_group.rvs(dim).astype(np.float32)
        
        # Factors computed during build
        self.centroid = None
        self.f_add = None
        self.f_rescale = None
        self.f_error = None
        
    def build(self, data: np.ndarray):
        """Build index - matches official implementation exactly"""
        n, dim = data.shape
        assert dim == self.dim
        
        # Step 1: Compute centroid
        self.centroid = np.mean(data, axis=0).astype(np.float32)
        
        # Step 2: Compute residuals (data - centroid)
        residuals = data - self.centroid
        
        # Step 3: Rotate residuals with P^T
        rotated = residuals @ self.P.T
        
        # Step 4: Extract binary codes (sign bits)
        codes = (rotated > 0).astype(np.uint8)
        
        # Step 5: Compute factors (from official one_bit_code_with_factor)
        cb = -0.5  # -((1 << 1) - 1) / 2.0
        xu_cb = codes.astype(np.float32) + cb
        
        # Norms
        l2_sqr = np.sum(residuals ** 2, axis=1)
        l2_norm = np.sqrt(l2_sqr)
        
        # Inner products (xu_cb is in ROTATED space, so rotate centroid too)
        centroid_rotated = self.centroid @ self.P.T
        ip_resi_xucb = np.sum(rotated * xu_cb, axis=1)  # residual is already rotated
        ip_cent_xucb = np.sum(centroid_rotated * xu_cb, axis=1)
        
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
    
    def search(self, query: np.ndarray, codes: np.ndarray, k: int = 10):
        """Search - matches official estimator formula"""
        # Step 1: Compute query residual and rotate
        q_residual = query - self.centroid
        q_rotated = q_residual @ self.P.T
        
        # Step 2: Compute query constants
        cb = -0.5
        sumq = np.sum(q_rotated)
        G_k1xSumq = sumq * cb
        G_add = np.sum(q_residual ** 2)
        
        # Step 3: Compute ip_x0_qr (inner product between codes and rotated query)
        # This is the key: sum query values where code bit is 1
        # Equivalent to: sum(q_rotated[i] for i where codes[:, i] == 1)
        ip_x0_qr = np.sum(codes * q_rotated, axis=1).astype(np.float32)
        
        # Step 4: Estimate distances using official formula
        # est_dist = f_add + G_add + f_rescale * (ip_x0_qr + G_k1xSumq)
        estimated_dist = self.f_add + G_add + self.f_rescale * (ip_x0_qr + G_k1xSumq)
        
        # Ensure non-negative
        estimated_dist = np.maximum(estimated_dist, 0)
        
        # Step 5: Get top-k
        k = min(k, len(estimated_dist))
        indices = np.argpartition(estimated_dist, k)[:k]
        indices = indices[np.argsort(estimated_dist[indices])]
        
        return estimated_dist[indices], indices
