"""
Random rotation for RaBitQ
Uses Hadamard transform + random signs (FFHT + Kac's Walk)
"""
import numpy as np
from numba import njit
from typing import Tuple


def generate_rotation_matrix(dim: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate random rotation parameters
    
    Returns 4 sequences of random signs for FFHT + Kac's Walk
    """
    np.random.seed(seed)
    
    # 4 sequences of random signs (+1 or -1)
    signs1 = np.random.choice([-1, 1], size=dim).astype(np.float32)
    signs2 = np.random.choice([-1, 1], size=dim).astype(np.float32)
    signs3 = np.random.choice([-1, 1], size=dim).astype(np.float32)
    signs4 = np.random.choice([-1, 1], size=dim).astype(np.float32)
    
    return signs1, signs2, signs3, signs4


@njit(fastmath=True, cache=True)
def hadamard_transform(x: np.ndarray) -> np.ndarray:
    """Fast Hadamard Transform (in-place, power-of-2 size)"""
    n = len(x)
    result = x.copy()
    
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = result[j]
                b = result[j + h]
                result[j] = a + b
                result[j + h] = a - b
        h *= 2
    
    return result / np.sqrt(n)


@njit(fastmath=True, cache=True)
def rotate_vector(
    x: np.ndarray,
    signs1: np.ndarray,
    signs2: np.ndarray,
    signs3: np.ndarray,
    signs4: np.ndarray
) -> np.ndarray:
    """Apply random rotation using FFHT + Kac's Walk
    
    Repeats 4 times:
    1. Flip signs
    2. Apply Hadamard transform
    3. Apply Givens rotation (45 degrees)
    """
    dim = len(x)
    result = x.copy()
    
    # Find largest power of 2 <= dim
    k = 1
    while k * 2 <= dim:
        k *= 2
    
    signs_list = [signs1, signs2, signs3, signs4]
    
    for round_idx in range(4):
        signs = signs_list[round_idx]
        
        # 1. Flip signs
        result = result * signs
        
        # 2. Apply Hadamard transform on first k elements
        if k > 1:
            result[:k] = hadamard_transform(result[:k])
        
        # 3. Givens rotation (45 degrees) between first and second half
        half = dim // 2
        cos_theta = np.cos(np.pi / 4)
        sin_theta = np.sin(np.pi / 4)
        
        for i in range(half):
            a = result[i]
            b = result[i + half] if i + half < dim else 0
            result[i] = cos_theta * a - sin_theta * b
            if i + half < dim:
                result[i + half] = sin_theta * a + cos_theta * b
    
    return result


class RandomRotation:
    """Random rotation for RaBitQ"""
    
    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        self.signs1, self.signs2, self.signs3, self.signs4 = generate_rotation_matrix(dim, seed)
    
    def rotate(self, x: np.ndarray) -> np.ndarray:
        """Rotate a single vector"""
        return rotate_vector(x, self.signs1, self.signs2, self.signs3, self.signs4)
    
    def rotate_batch(self, X: np.ndarray) -> np.ndarray:
        """Rotate a batch of vectors"""
        n = X.shape[0]
        result = np.zeros_like(X)
        for i in range(n):
            result[i] = self.rotate(X[i])
        return result
