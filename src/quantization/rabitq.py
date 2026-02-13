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
        
        # Step 3: Quantize residuals
        if self.bq == 1:
            # 1-bit: binary quantization of raw residuals
            codes = (residuals > 0).astype(np.uint8)
        else:
            # Multi-bit: optimized quantization with best_rescale_factor
            codes = self._quantize_multibit(residuals)
        
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
        """Quantize residuals using optimized multi-bit quantization
        
        Format: 1 bit for sign + (bq-1) bits for magnitude
        """
        n, dim = residuals.shape
        codes = np.zeros((n, dim), dtype=np.uint8)
        
        kEps = 1e-5
        ex_bits = self.bq - 1  # Reserve 1 bit for sign
        
        for i in range(n):
            # Extract sign
            signs = (residuals[i] > 0).astype(np.uint8)
            
            # Take absolute values for magnitude quantization
            o_abs = np.abs(residuals[i])
            
            # Find optimal rescaling factor
            t = self._best_rescale_factor(o_abs, ex_bits)
            
            # Quantize magnitude
            quantized = np.floor(t * o_abs + kEps).astype(np.int32)
            quantized = np.clip(quantized, 0, (1 << ex_bits) - 1)
            
            # Combine: sign bit in MSB, magnitude in lower bits
            codes[i] = quantized + (signs << ex_bits)
        
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
