"""
Per-detector adapter implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union


class BaseDetector(ABC):
    """
    Abstract base class for all detectors.

    All detector implementations must inherit from this class and implement
    the detect() method to ensure consistent interface.
    """

    def __init__(self, config: Dict):
        """
        Initialize detector with configuration.

        Args:
            config: Configuration dictionary with detector-specific parameters
        """
        self.config = config

    @abstractmethod
    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Detect if text is AI-generated. Supports single text or batch.

        Args:
            text: Input text or list of texts to analyze

        Returns:
            Result dictionary (single) or list of dictionaries (batch):
            {
                'text': str,           # Input text
                'label': int,          # 0=human, 1=AI-generated
                'score': float,        # Detection score (higher = more likely AI)
                'metadata': dict       # Detector-specific debugging info
            }
        """
        pass

    def cleanup(self):
        """
        Release GPU memory and other resources.

        Override in subclasses that load models to GPU.
        Called automatically when pipeline is deleted or used as context manager.
        """
        pass


# Lazy imports — detectors depend on baseline submodules that may not be installed.
# Each detector is only imported when actually accessed, so missing dependencies
# only cause errors for detectors you try to use.
def __getattr__(name):
    _lazy_imports = {
        "BinocularsDetector": "opai_bench_detectors.detectors.binoculars_detector",
        "ClaudeDetector": "opai_bench_detectors.detectors.claude_detector",
        "DAMASHADetector": "opai_bench_detectors.detectors.damasha_detector",
        "DesklibDetector": "opai_bench_detectors.detectors.desklib_detector",
        "DetectLLMDetector": "opai_bench_detectors.detectors.detectllm_detector",
        "DNADetectLLMDetector": "opai_bench_detectors.detectors.dna_detectllm_detector",
        "E5SmallDetector": "opai_bench_detectors.detectors.e5_small_detector",
        "FastDetectGPTDetector": "opai_bench_detectors.detectors.fast_detectgpt_detector",
        "GeminiDetector": "opai_bench_detectors.detectors.gemini_detector",
        "GenAISentenceDetector": "opai_bench_detectors.detectors.genai_sentence_detector",
        "GigacheckDetector": "opai_bench_detectors.detectors.gigacheck_detector",
        "GlimpseDetector": "opai_bench_detectors.detectors.glimpse_detector",
        "MGTDDetector": "opai_bench_detectors.detectors.mgtd_detector",
        "OODLLMDetector": "opai_bench_detectors.detectors.ood_llm_detector",
        "OpenAIDetector": "opai_bench_detectors.detectors.openai_detector",
        "RADARDetector": "opai_bench_detectors.detectors.radar_detector",
        "RoBERTaOpenAIDetector": "opai_bench_detectors.detectors.roberta_openai_detector",
        "RoFTBoundaryDetector": "opai_bench_detectors.detectors.roft_boundary_detector",
        "SeqXGPTDetector": "opai_bench_detectors.detectors.seqxgpt_detector",
        "MIECDetector": "opai_bench_detectors.detectors.miec_detector",
        "GLCLiCDetector": "opai_bench_detectors.detectors.gl_clic_detector",
        "SenDetEXDetector": "opai_bench_detectors.detectors.sendetex_detector",
        "AdaLocDetector": "opai_bench_detectors.detectors.adaloc_detector",
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseDetector",
    "BinocularsDetector",
    "ClaudeDetector",
    "DAMASHADetector",
    "DesklibDetector",
    "DetectLLMDetector",
    "DNADetectLLMDetector",
    "E5SmallDetector",
    "FastDetectGPTDetector",
    "GeminiDetector",
    "GenAISentenceDetector",
    "GigacheckDetector",
    "GlimpseDetector",
    "MGTDDetector",
    "OODLLMDetector",
    "OpenAIDetector",
    "RADARDetector",
    "RoBERTaOpenAIDetector",
    "RoFTBoundaryDetector",
    "SeqXGPTDetector",
    "MIECDetector",
    "GLCLiCDetector",
    "SenDetEXDetector",
    "AdaLocDetector",
]
