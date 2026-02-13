"""RaBitQ with Numba JIT optimization"""
import numpy as np
from numba import njit, prange
from typing import Tuple

@njit(parallel=True, fastmath=True, cache=True)
def compute_l2_distances(query, codes, centroid, scale, res_min):
    """JIT-compiled L2 distance computation"""
    n, dim = codes.shape
    dists = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        dist = 0.0
        for j in range(dim):
            val = codes[i, j] * scale + res_min + centroid[j]
            diff = val - query[j]
            dist += diff * diff
        dists[i] = dist
    return dists

@njit(parallel=True, fastmath=True, cache=True)
def compute_ip_distances(query, codes, centroid, scale, res_min):
    """JIT-compiled IP distance computation"""
    n, dim = codes.shape
    dists = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        dot = 0.0
        for j in range(dim):
            val = codes[i, j] * scale + res_min + centroid[j]
            dot += val * query[j]
        dists[i] = -dot
    return dists

@njit(parallel=True, fastmath=True, cache=True)
def compute_1bit_distances_l2(query, codes, centroid, f_add, f_rescale):
    """JIT-compiled 1-bit L2 distance"""
    n, dim = codes.shape
    q_residual = query - centroid
    G_add = np.sum(q_residual ** 2)
    sumq = np.sum(q_residual)
    cb = -0.5
    
    dists = np.empty(n, dtype=np.float32)
    for i in prange(n):
        ip_codes_qres = 0.0
        for j in range(dim):
            ip_codes_qres += codes[i, j] * q_residual[j]
        ip_x0_qr = ip_codes_qres + cb * sumq
        dists[i] = max(0.0, f_add[i] + G_add + f_rescale[i] * ip_x0_qr)
    return dists

@njit(parallel=True, fastmath=True, cache=True)
def compute_1bit_distances_ip(query, codes, centroid, f_add, f_rescale):
    """JIT-compiled 1-bit IP distance"""
    n, dim = codes.shape
    q_residual = query - centroid
    G_add = -np.dot(query, centroid)
    sumq = np.sum(q_residual)
    cb = -0.5
    
    dists = np.empty(n, dtype=np.float32)
    for i in prange(n):
        ip_codes_qres = 0.0
        for j in range(dim):
            ip_codes_qres += codes[i, j] * q_residual[j]
        ip_x0_qr = ip_codes_qres + cb * sumq
        dists[i] = f_add[i] + G_add + f_rescale[i] * ip_x0_qr
    return dists


class RaBitQNumba:
    """RaBitQ with Numba JIT optimization"""
    
    def __init__(self, dim: int, bq: int = 1, metric: str = 'L2'):
        self.dim = dim
        self.bq = bq
        self.metric = metric
        self.n_levels = 2 ** bq
        
    def build(self, data: np.ndarray) -> np.ndarray:
        """Build quantization (same as original)"""
        self.centroid = np.mean(data, axis=0).astype(np.float32)
        residuals = data - self.centroid
        
        if self.bq == 1:
            codes = (residuals > 0).astype(np.uint8)
        else:
            self.res_min = float(residuals.min())
            self.res_max = float(residuals.max())
            self.scale = (self.res_max - self.res_min) / (self.n_levels - 1)
            codes = np.clip(np.round((residuals - self.res_min) / self.scale), 0, self.n_levels - 1).astype(np.uint8)
        
        if self.bq == 1:
            cb = -(self.n_levels - 1) / 2.0
            xu_cb = codes.astype(np.float32) + cb
            l2_sqr = np.sum(residuals ** 2, axis=1)
            ip_residual_xucb = np.sum(residuals * xu_cb, axis=1)
            ip_residual_xucb = np.where(ip_residual_xucb == 0, np.inf, ip_residual_xucb)
            ip_cent_xucb = np.sum(self.centroid * xu_cb, axis=1)
            
            if self.metric == 'L2':
                self.f_add = l2_sqr + 2 * l2_sqr * ip_cent_xucb / ip_residual_xucb
                self.f_rescale = -2 * l2_sqr / ip_residual_xucb
            elif self.metric in ('IP', 'Cosine'):
                ip_residual_c = np.sum(residuals * self.centroid, axis=1)
                self.f_add = 1 - ip_residual_c + l2_sqr * ip_cent_xucb / ip_residual_xucb
                self.f_rescale = -l2_sqr / ip_residual_xucb
        
        return codes
    
    def search(self, query: np.ndarray, codes: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """JIT-optimized search"""
        if self.bq == 1:
            if self.metric == 'L2':
                dists = compute_1bit_distances_l2(query, codes, self.centroid, self.f_add, self.f_rescale)
            elif self.metric in ('IP', 'Cosine'):
                dists = compute_1bit_distances_ip(query, codes, self.centroid, self.f_add, self.f_rescale)
        else:
            if self.metric == 'L2':
                dists = compute_l2_distances(query, codes, self.centroid, self.scale, self.res_min)
            elif self.metric in ('IP', 'Cosine'):
                dists = compute_ip_distances(query, codes, self.centroid, self.scale, self.res_min)
        
        k = min(k, len(dists))
        if k == 0:
            return np.array([]), np.array([])
        
        top_k_indices = np.argpartition(dists, k-1)[:k]
        top_k_indices = top_k_indices[np.argsort(dists[top_k_indices])]
        return dists[top_k_indices], top_k_indices
