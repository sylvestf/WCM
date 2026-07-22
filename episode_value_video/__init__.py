"""Cinematic episode videos with frame-aligned predicted value curves.

This package is intentionally independent from ``world_critic`` evaluation.
It consumes the evaluator's ``episode_curves.json`` output and never reads or
renders the supervised return curve.
"""

from .curves import EpisodeCurve, load_episode_curves

__all__ = ["EpisodeCurve", "load_episode_curves"]
__version__ = "0.1.0"
