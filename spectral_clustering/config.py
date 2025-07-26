import warnings
from multiprocessing import cpu_count

warnings.filterwarnings("ignore")


class VerbosityLevel:
    SILENT = 0  # No output except final results
    MINIMAL = 1  # Only best results and major milestones
    MEDIUM = 2  # Best results + progress intervals
    DETAILED = 3  # All evaluation results and progress
    DEBUG = 4  # Everything including debug info


DEFAULT_VERBOSITY = VerbosityLevel.DETAILED

N_CORES = max(1, cpu_count() - 1)

PARAM_NAMES = [
    "affinity",
    "gamma",
    "n_clusters",
    "n_neighbors",
    "eigen_solver",
    "assign_labels",
    "n_components_factor",
    "n_init",
]
