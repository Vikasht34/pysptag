"""Optimized RaBitQ with vectorized operations"""
import numpy as np
from typing import Tuple

class RaBitQOptimized:
    """Optimized RaBitQ quantization with vectorized operations"""
    
    def __init__(self, dim: int, bq: int = 1, metric: str = 'L2'):
        self.dim = dim
        self.bq = bq
        self.metric = metric
        self.n_levels = 2 ** bq
        
        # Precompute constants
        self.cb = -(self.n_levels - 1) / 2.0
        
    def build(self, data: np.ndarray) -> np.ndarray:
        """Build quantization with vectorized operations"""
        n = len(data)
        
        # Step 1: Compute centroid
        self.centroid = np.mean(data, axis=0).astype(np.float32)
        
        # Step 2: Compute residuals (vectorized)
        residuals = data - self.centroid
        
        # Step 3: Quantize
        if self.bq == 1:
            codes = (residuals > 0).astype(np.uint8)
        else:
            self.res_min = residuals.min()
            self.res_max = residuals.max()
            self.scale = (self.res_max - self.res_min) / (self.n_levels - 1)
            codes = np.clip(np.round((residuals - self.res_min) / self.scale), 0, self.n_levels - 1).astype(np.uint8)
        
        # Step 4: Compute factors (vectorized)
        xu_cb = codes.astype(np.float32) + self.cb
        
        if self.bq == 1:
            # Vectorized RaBitQ formula
            l2_sqr = np.sum(residuals ** 2, axis=1)
            ip_residual_xucb = np.einsum('ij,ij->i', residuals, xu_cb)  # Faster than sum(a*b)
            ip_residual_xucb = np.where(ip_residual_xucb == 0, np.inf, ip_residual_xucb)
            ip_cent_xucb = np.einsum('j,ij->i', self.centroid, xu_cb)
            
            if self.metric == 'L2':
                self.f_add = l2_sqr + 2 * l2_sqr * ip_cent_xucb / ip_residual_xucb
                self.f_rescale = -2 * l2_sqr / ip_residual_xucb
            elif self.metric in ('IP', 'Cosine'):
                ip_residual_c = np.einsum('ij,j->i', residuals, self.centroid)
                self.f_add = 1 - ip_residual_c + l2_sqr * ip_cent_xucb / ip_residual_xucb
                self.f_rescale = -l2_sqr / ip_residual_xucb
        else:
            # Multi-bit: store delta
            norm_residual = np.linalg.norm(residuals, axis=1)
            norm_quan = np.linalg.norm(xu_cb, axis=1)
            ip_residual_xucb = np.einsum('ij,ij->i', residuals, xu_cb)
            cos_sim = ip_residual_xucb / (norm_residual * norm_quan + 1e-10)
            self.delta = norm_residual / (norm_quan + 1e-10) * cos_sim
        
        return codes
    
    def search(self, query: np.ndarray, codes: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized search"""
        q_residual = query - self.centroid
        
        if self.bq == 1:
            # Vectorized 1-bit search
            ip_codes_qres = np.einsum('ij,j->i', codes.astype(np.float32), q_residual)
            sumq = np.sum(q_residual)
            ip_x0_qr = ip_codes_qres + self.cb * sumq
            
            if self.metric == 'L2':
                G_add = np.sum(q_residual ** 2)
            elif self.metric in ('IP', 'Cosine'):
                G_add = -np.dot(query, self.centroid)
            
            estimated_dist = self.f_add + G_add + self.f_rescale * ip_x0_qr
            
            if self.metric == 'L2':
                estimated_dist = np.maximum(estimated_dist, 0)
        else:
            # Vectorized multi-bit search
            dequantized_residuals = codes.astype(np.float32) * self.scale + self.res_min
            reconstructed_vectors = dequantized_residuals + self.centroid
            
            if self.metric == 'L2':
                # Vectorized L2: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
                query_norm = np.sum(query ** 2)
                recon_norms = np.sum(reconstructed_vectors ** 2, axis=1)
                dots = np.dot(reconstructed_vectors, query)
                estimated_dist = query_norm + recon_norms - 2 * dots
            elif self.metric in ('IP', 'Cosine'):
                estimated_dist = -np.dot(reconstructed_vectors, query)  # Negate for sorting
        
        # Get top-k (vectorized)
        k = min(k, len(estimated_dist))
        if k == 0:
            return np.array([]), np.array([])
        
        top_k_indices = np.argpartition(estimated_dist, k-1)[:k]
        top_k_indices = top_k_indices[np.argsort(estimated_dist[top_k_indices])]
        top_k_dists = estimated_dist[top_k_indices]
        
        return top_k_dists, top_k_indices
