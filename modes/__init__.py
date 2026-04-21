"""Modes package - only QuizGenerator is exposed as the active mode.

This package keeps the modes namespace clean and focused on quiz generation.
"""

from .quiz import QuizGenerator

__all__ = ["QuizGenerator"]
