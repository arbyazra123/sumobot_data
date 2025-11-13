"""
Utility functions for sumobot analyzer
"""
import polars as pl
import re

# Check if GPU support is available
GPU_AVAILABLE = False
try:
    # Try a simple GPU operation to check availability
    pl.LazyFrame({"test": [1]}).collect(engine="gpu")
    GPU_AVAILABLE = True
    print("✅ GPU support available - will use GPU acceleration")
except Exception:
    print("✅ Using CPU")


def collect_with_gpu(lf):
    """Helper to collect LazyFrame with GPU if available, otherwise uses CPU"""
    if GPU_AVAILABLE:
        try:
            return lf.collect(engine="gpu")
        except Exception:
            # Fallback to CPU if GPU collection fails
            return lf.collect()
    else:
        return lf.collect()


def extract_timer_from_config(config_folder):
    """
    Extract Timer value from config folder name
    e.g., "Timer_15__ActInterval_0.1" -> 15

    Args:
        config_folder: Config folder name

    Returns:
        Timer value as float or None if not found
    """
    match = re.search(r'Timer_(\d+(?:\.\d+)?)', config_folder)
    if match:
        return float(match.group(1))
    return None
