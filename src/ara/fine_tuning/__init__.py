"""Fine-tuning pipeline — dataset preparation, QLoRA training, evaluation.

Modules:
    dataset: Diary → ChatML conversion, synthetic data generation, LLM-as-judge scoring
    train: Unsloth QLoRA training, LoRA config, GGUF export, Ollama import
    evaluate: A/B testing framework, pairwise comparison, humor dial consistency
"""

from ara.fine_tuning.dataset import (
    ChatMLMessage,
    ConversationType,
    DiarySegment,
    HumorLevel,
    TrainingSample,
)
from ara.fine_tuning.evaluate import EvalCategory, EvalPrompt, EvalResults
from ara.fine_tuning.train import LoRAConfig, TrainingConfig

__all__ = [
    "ChatMLMessage",
    "ConversationType",
    "DiarySegment",
    "EvalCategory",
    "EvalPrompt",
    "EvalResults",
    "HumorLevel",
    "LoRAConfig",
    "TrainingConfig",
    "TrainingSample",
]
