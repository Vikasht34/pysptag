"""Python wrapper for Rust async I/O module."""

try:
    from .fast_io import AsyncBatchLoader
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    AsyncBatchLoader = None

def load_postings_rust(file_path, requests):
    """Load multiple postings using Rust async I/O.
    
    Args:
        file_path: Path to postings file
        requests: List of (offset, length) tuples
    
    Returns:
        List of bytes objects
    """
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust extension not available. Run: bash build_rust.sh")
    
    loader = AsyncBatchLoader(file_path)
    try:
        return loader.load_batch(requests)
    finally:
        loader.close()
