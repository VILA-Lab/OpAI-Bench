"""
Unified interface for AI-text detection methods used in OpAI-Bench.

This package provides a config-driven interface that wraps multiple
AI-text detectors behind a single `pipeline(...)` entry point.
"""

from opai_bench_detectors.core import pipeline, get_pipeline_from_cfg

__version__ = "0.1.0"
__all__ = ["pipeline", "get_pipeline_from_cfg"]
