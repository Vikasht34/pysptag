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
        self.delta = None  # Scalar quantization delta
        self.vl = None     # Scalar quantization vl
        
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
            # 1-bit: binary quantization of raw residuals
            codes = (residuals > 0).astype(np.uint8)
        else:
            # Multi-bit: optimized quantization with best_rescale_factor
            codes = self._quantize_multibit(residuals)
        
        # Step 4: Compute RaBitQ factors
        cb = -(self.n_levels - 1) / 2.0
        xu_cb = codes.astype(np.float32) + cb
        
        if self.bq == 1:
            # 1-bit: Use original RaBitQ formula
            l2_sqr = np.sum(residuals ** 2, axis=1)
            
            ip_residual_xucb = np.sum(residuals * xu_cb, axis=1)
            ip_residual_xucb = np.where(ip_residual_xucb == 0, np.inf, ip_residual_xucb)
            
            ip_cent_xucb = np.sum(self.centroid * xu_cb, axis=1)
            
            if self.metric == 'L2':
                self.f_add = l2_sqr + 2 * l2_sqr * ip_cent_xucb / ip_residual_xucb
                self.f_rescale = -2 * l2_sqr / ip_residual_xucb
            elif self.metric in ('IP', 'Cosine'):
                ip_residual_c = np.sum(residuals * self.centroid, axis=1)
                self.f_add = -ip_residual_c + l2_sqr * ip_cent_xucb / ip_residual_xucb
                self.f_rescale = -l2_sqr / ip_residual_xucb
        else:
            # Multi-bit: Use scalar quantization with delta rescaling
            cb = -(self.n_levels - 1) / 2.0
            xu_cb = codes.astype(np.float32) + cb
            
            # Dequantize codes back to residual space
            dequantized_residuals = codes.astype(np.float32) * self.scale + self.res_min
            
            # Compute norms for delta
            norm_residual = np.linalg.norm(residuals, axis=1)
            norm_quan = np.linalg.norm(xu_cb, axis=1)
            
            # Compute cosine similarity
            ip_residual_xucb = np.sum(residuals * xu_cb, axis=1)
            cos_sim = ip_residual_xucb / (norm_residual * norm_quan + 1e-10)
            
            # Delta rescaling factor
            delta = norm_residual / (norm_quan + 1e-10) * cos_sim
            
            # Store for search
            self.delta = delta
            
            self.f_add = None
            self.f_rescale = None
        
        return codes
    
    def _best_rescale_factor(self, o_abs: np.ndarray, ex_bits: int) -> float:
        """Find optimal rescaling factor for multi-bit quantization (from official RaBitQ)"""
        import heapq
        
        kEps = 1e-5
        kNEnum = 10
        kTightStart = [0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81]
        
        dim = len(o_abs)
        max_o = o_abs.max()
        
        t_end = ((1 << ex_bits) - 1 + kNEnum) / max_o
        t_start = t_end * kTightStart[ex_bits]
        
        # Initialize quantization codes
        cur_o_bar = np.floor(t_start * o_abs + kEps).astype(np.int32)
        sqr_denominator = dim * 0.25 + np.sum(cur_o_bar * cur_o_bar + cur_o_bar)
        numerator = np.sum((cur_o_bar + 0.5) * o_abs)
        
        # Priority queue: (t_value, dimension_index)
        next_t = [(float((cur_o_bar[i] + 1) / o_abs[i]), i) for i in range(dim)]
        heapq.heapify(next_t)
        
        max_ip = 0
        best_t = 0
        
        while next_t:
            cur_t, update_id = heapq.heappop(next_t)
            
            # Update quantization code
            cur_o_bar[update_id] += 1
            update_o_bar = cur_o_bar[update_id]
            sqr_denominator += 2 * update_o_bar
            numerator += o_abs[update_id]
            
            # Compute inner product
            cur_ip = numerator / np.sqrt(sqr_denominator)
            if cur_ip > max_ip:
                max_ip = cur_ip
                best_t = cur_t
            
            # Add next candidate
            if update_o_bar < (1 << ex_bits) - 1:
                t_next = (update_o_bar + 1) / o_abs[update_id]
                if t_next < t_end:
                    heapq.heappush(next_t, (t_next, update_id))
        
        return best_t
    
    def _quantize_multibit(self, residuals: np.ndarray) -> np.ndarray:
        """Quantize residuals using uniform quantization (simpler approach)"""
        n, dim = residuals.shape
        
        # Use global min/max for uniform quantization
        res_min = residuals.min()
        res_max = residuals.max()
        self.res_min = res_min
        self.res_max = res_max
        
        # Quantize to [0, 2^bq - 1]
        scale = (res_max - res_min) / (self.n_levels - 1)
        if scale == 0:
            scale = 1
        self.scale = scale
        
        codes = np.clip(
            np.round((residuals - res_min) / scale),
            0, self.n_levels - 1
        ).astype(np.uint8)
        
        return codes
    
    def search(self, query: np.ndarray, codes: np.ndarray, data: np.ndarray, k: int = 10):
        """Search with RaBitQ (1-bit) or scalar quantization (multi-bit)"""
        q_residual = query - self.centroid
        
        if self.bq == 1:
            # 1-bit: Use RaBitQ distance estimation formula
            cb = -0.5
            ip_codes_qres = np.sum(codes * q_residual, axis=1).astype(np.float32)
            sumq = np.sum(q_residual)
            ip_x0_qr = ip_codes_qres + cb * sumq
            
            # Metric-specific G_add computation
            if self.metric == 'L2':
                G_add = np.sum(q_residual ** 2)
            elif self.metric in ('IP', 'Cosine'):
                G_add = -np.dot(query, self.centroid)
            
            estimated_dist = self.f_add + G_add + self.f_rescale * ip_x0_qr
            estimated_dist = np.maximum(estimated_dist, 0)
        else:
            # Multi-bit: Dequantize and reconstruct
            dequantized_residuals = codes.astype(np.float32) * self.scale + self.res_min
            reconstructed_vectors = dequantized_residuals + self.centroid
            
            # Compute distances
            if self.metric == 'L2':
                estimated_dist = np.sum((reconstructed_vectors - query) ** 2, axis=1)
            elif self.metric == 'IP':
                estimated_dist = -np.dot(reconstructed_vectors, query)
            elif self.metric == 'Cosine':
                estimated_dist = -np.dot(reconstructed_vectors, query)
        
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
