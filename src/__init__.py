"""
PySPTAG - Python Implementation of Microsoft SPTAG

Complete implementation of SPANN + SPFresh + RaBitQ for billion-scale vector search.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

# Main index classes
from .index.spann import SPANN
from .storage.file_controller import SPANNDisk
from .index.spfresh import SPFreshDynamic

# Core data structures
from .core.version_map import VersionMap
from .core.posting_record import PostingSizeRecord

# Utilities
from .utils.io import load_hdf5_dataset

__all__ = [
    # Main classes
    "SPANN",
    "SPANNDisk",
    "SPFreshDynamic",
    
    # Core structures
    "VersionMap",
    "PostingSizeRecord",
    
    # Utilities
    "load_hdf5_dataset",
]
