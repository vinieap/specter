"""
Optimization framework for clustering algorithms.

This package provides various optimization strategies for clustering algorithms,
including batch, sequential, and multi-study approaches.
"""

from .base import BaseOptimizer, OptimizationResult
from .batch import BatchOptimizer
from .sequential import SequentialOptimizer
from .multi_study import MultiStudyOptimizer

__all__ = [
    'BaseOptimizer',
    'OptimizationResult',
    'BatchOptimizer',
    'SequentialOptimizer',
    'MultiStudyOptimizer',
]