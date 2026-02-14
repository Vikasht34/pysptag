"""
SPTAG-compatible parameters and calculations
"""

# SPTAG defaults from ParameterDefinitionList.h
PAGE_SIZE = 4096  # bytes
PAGE_SIZE_EX = 12  # log2(4096) = 12

# Build parameters
DEFAULT_INTERNAL_RESULT_NUM = 64
DEFAULT_REPLICA_COUNT = 8
DEFAULT_POSTING_PAGE_LIMIT = 3
DEFAULT_POSTING_VECTOR_LIMIT = 118
DEFAULT_RNG_FACTOR = 1.0

# Search parameters
DEFAULT_SEARCH_INTERNAL_RESULT_NUM = 64
DEFAULT_MAX_CHECK = 4096
DEFAULT_MAX_DIST_RATIO = 10000.0
DEFAULT_SEARCH_POSTING_PAGE_LIMIT = 3

# Hierarchical clustering
DEFAULT_RATIO = 0.01
DEFAULT_SELECT_THRESHOLD = 6
DEFAULT_SPLIT_THRESHOLD = 25
DEFAULT_SPLIT_FACTOR = 5
DEFAULT_KMEANS_K = 32
DEFAULT_LEAF_SIZE = 8


def calculate_posting_page_limit(posting_vector_limit: int, dim: int, 
                                 value_size: int = 4) -> int:
    """
    Calculate posting page limit from vector limit (SPTAG formula).
    
    From ExtraStaticSearcher.h:186:
    p_opt.m_postingPageLimit = max(p_opt.m_postingPageLimit, 
        static_cast<int>((p_opt.m_postingVectorLimit * (p_opt.m_dim * sizeof(ValueType) + sizeof(int)) 
        + PageSize - 1) / PageSize));
    
    Args:
        posting_vector_limit: Max vectors per posting
        dim: Vector dimension
        value_size: Bytes per value (4 for float32, 1 for uint8)
    
    Returns:
        Number of pages needed
    """
    vector_info_size = dim * value_size + 4  # vector + ID
    total_bytes = posting_vector_limit * vector_info_size
    pages = (total_bytes + PAGE_SIZE - 1) // PAGE_SIZE
    return pages


def calculate_posting_size_limit(posting_page_limit: int, dim: int,
                                 value_size: int = 4) -> int:
    """
    Calculate max vectors from page limit (SPTAG formula).
    
    From ExtraStaticSearcher.h:787:
    postingSizeLimit = static_cast<int>(p_opt.m_postingPageLimit * PageSize / vectorInfoSize);
    
    Args:
        posting_page_limit: Max pages per posting
        dim: Vector dimension
        value_size: Bytes per value
    
    Returns:
        Max vectors that fit in pages
    """
    vector_info_size = dim * value_size + 4
    return (posting_page_limit * PAGE_SIZE) // vector_info_size


def get_sptag_posting_limit(dim: int, value_size: int = 4,
                            posting_vector_limit: int = DEFAULT_POSTING_VECTOR_LIMIT,
                            posting_page_limit: int = DEFAULT_POSTING_PAGE_LIMIT) -> int:
    """
    Get SPTAG-compatible posting size limit.
    
    SPTAG uses the LARGER of:
    1. posting_page_limit converted to vectors
    2. posting_vector_limit converted to pages, then back to vectors
    
    Args:
        dim: Vector dimension
        value_size: Bytes per value (4 for float32, 1 for uint8 quantized)
        posting_vector_limit: Target vectors per posting
        posting_page_limit: Min pages per posting
    
    Returns:
        Final posting size limit in vectors
    """
    # Calculate pages needed for vector limit
    pages_from_vectors = calculate_posting_page_limit(
        posting_vector_limit, dim, value_size
    )
    
    # Use max of the two
    final_pages = max(posting_page_limit, pages_from_vectors)
    
    # Convert back to vectors
    final_limit = calculate_posting_size_limit(final_pages, dim, value_size)
    
    return final_limit


# Example calculations for SIFT 1M (dim=128, float32)
if __name__ == "__main__":
    dim = 128
    
    print("SPTAG Posting Size Calculations")
    print("=" * 60)
    print(f"Dimension: {dim}")
    print(f"Value size: 4 bytes (float32)")
    print(f"Vector info size: {dim * 4 + 4} bytes (vector + ID)")
    print()
    
    # Default SPTAG settings
    print("Default SPTAG settings:")
    print(f"  posting_vector_limit = {DEFAULT_POSTING_VECTOR_LIMIT}")
    print(f"  posting_page_limit = {DEFAULT_POSTING_PAGE_LIMIT}")
    
    pages = calculate_posting_page_limit(DEFAULT_POSTING_VECTOR_LIMIT, dim, 4)
    print(f"  → Pages needed: {pages}")
    
    limit = get_sptag_posting_limit(dim, 4)
    print(f"  → Final limit: {limit} vectors")
    print(f"  → Storage: {limit * (dim * 4 + 4) / 1024:.1f} KB per posting")
    print()
    
    # With 2-bit quantization (RaBitQ)
    print("With 2-bit quantization:")
    code_dim = (dim + 3) // 4  # 2 bits per dim = 4 dims per byte
    print(f"  Code dimension: {code_dim} bytes")
    print(f"  Vector info size: {code_dim + 4} bytes")
    
    limit_quant = get_sptag_posting_limit(dim, value_size=code_dim/dim)
    print(f"  → Final limit: {limit_quant} vectors")
    print(f"  → Storage: {limit_quant * (code_dim + 4) / 1024:.1f} KB per posting")
    print()
    
    # Comparison
    print("Comparison:")
    print(f"  Unquantized: {limit} vectors, {limit * (dim * 4 + 4) / 1024:.1f} KB")
    print(f"  Quantized:   {limit_quant} vectors, {limit_quant * (code_dim + 4) / 1024:.1f} KB")
    print(f"  Compression: {(limit * (dim * 4 + 4)) / (limit_quant * (code_dim + 4)):.1f}×")
