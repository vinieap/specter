import psutil


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb


def get_array_size_mb(array):
    """Calculate array size in MB"""
    return array.nbytes / 1024 / 1024


def format_params_for_display(params):
    """Format parameters for clean display"""
    clean_params = {}
    for k, v in params.items():
        # Convert numpy types to Python native types
        if hasattr(v, "item"):  # numpy scalar
            clean_params[k] = v.item()
        elif str(type(v)).startswith("<class 'numpy."):  # numpy types
            clean_params[k] = type(v).__name__.replace("np.", "") + f"({v})"
        else:
            clean_params[k] = v
    return clean_params
