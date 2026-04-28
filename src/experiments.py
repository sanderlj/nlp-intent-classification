"""Experiment helpers for Assignment 1.

This module is intentionally light at the start: it gives the project one
place to grow the tuning loops and result logging as implementation proceeds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentResult:
    language: str
    model_name: str
    feature_name: str
    dev_accuracy: float
    test_accuracy: float
    hyperparameters: dict
    notes: str = ""
