"""DEAP emotion recognition pipeline (approach 3)."""

from .config import Config
from .eval import run_cross_subject, run_subject_dependent

__all__ = ["Config", "run_cross_subject", "run_subject_dependent"]
