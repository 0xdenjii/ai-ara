"""Tests for the fine-tuning training module.

Tests configuration, serialization, and non-GPU utility functions.
Actual training functions require CUDA and are not unit-tested.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ara.fine_tuning.train import (
    LoRAConfig,
    QuantizationType,
    TrainingConfig,
    get_training_summary,
    load_training_config,
    save_config,
)

# ── LoRA Config ─────────────────────────────────────────────────────────────


class TestLoRAConfig:
    def test_default_values(self) -> None:
        config = LoRAConfig()
        assert config.rank == 16
        assert config.alpha == 32
        assert config.dropout == 0.05
        assert "q_proj" in config.target_modules
        assert "k_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert "o_proj" in config.target_modules
        assert "gate_proj" in config.target_modules
        assert "up_proj" in config.target_modules
        assert "down_proj" in config.target_modules
        assert len(config.target_modules) == 7

    def test_custom_values(self) -> None:
        config = LoRAConfig(rank=32, alpha=64, use_dora=True)
        assert config.rank == 32
        assert config.alpha == 64
        assert config.use_dora is True

    def test_use_rslora_default(self) -> None:
        config = LoRAConfig()
        assert config.use_rslora is True


# ── Training Config ─────────────────────────────────────────────────────────


class TestTrainingConfig:
    def test_default_values(self) -> None:
        config = TrainingConfig()
        assert config.base_model == "unsloth/Qwen3-30B-A3B-bnb-4bit"
        assert config.max_seq_length == 4096
        assert config.load_in_4bit is True
        assert config.num_epochs == 3
        assert config.batch_size == 2
        assert config.gradient_accumulation_steps == 4
        assert config.learning_rate == 2e-4
        assert config.gradient_checkpointing is True
        assert config.bf16 is True
        assert config.seed == 42

    def test_effective_batch_size(self) -> None:
        config = TrainingConfig()
        effective = config.batch_size * config.gradient_accumulation_steps
        assert effective == 8

    def test_to_dict(self) -> None:
        config = TrainingConfig()
        data = config.to_dict()
        assert data["base_model"] == "unsloth/Qwen3-30B-A3B-bnb-4bit"
        assert data["lora_rank"] == 16
        assert data["lora_alpha"] == 32
        assert data["effective_batch_size"] == 8
        assert data["learning_rate"] == 2e-4
        assert data["seed"] == 42

    def test_to_dict_includes_all_keys(self) -> None:
        config = TrainingConfig()
        data = config.to_dict()
        expected_keys = {
            "base_model",
            "max_seq_length",
            "load_in_4bit",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "lora_use_dora",
            "lora_use_rslora",
            "num_epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "effective_batch_size",
            "learning_rate",
            "weight_decay",
            "warmup_ratio",
            "lr_scheduler_type",
            "gradient_checkpointing",
            "optim",
            "bf16",
            "fp16",
            "seed",
            "train_data_path",
            "val_data_path",
            "output_dir",
        }
        assert set(data.keys()) == expected_keys

    def test_custom_lora(self) -> None:
        lora = LoRAConfig(rank=64, alpha=128)
        config = TrainingConfig(lora=lora)
        assert config.lora.rank == 64
        data = config.to_dict()
        assert data["lora_rank"] == 64


# ── Config Save/Load ────────────────────────────────────────────────────────


class TestSaveLoadConfig:
    def test_save_config(self, tmp_path: Path) -> None:
        config = TrainingConfig()
        output = tmp_path / "config.json"
        save_config(config, output)
        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["base_model"] == "unsloth/Qwen3-30B-A3B-bnb-4bit"

    def test_save_creates_directories(self, tmp_path: Path) -> None:
        config = TrainingConfig()
        output = tmp_path / "nested" / "dir" / "config.json"
        save_config(config, output)
        assert output.exists()

    def test_load_config(self, tmp_path: Path) -> None:
        config = TrainingConfig(num_epochs=5, learning_rate=1e-4)
        output = tmp_path / "config.json"
        save_config(config, output)
        loaded = load_training_config(output)
        assert loaded.num_epochs == 5
        assert loaded.learning_rate == 1e-4
        assert loaded.lora.rank == 16

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_training_config(Path("/nonexistent/config.json"))

    def test_roundtrip_preserves_values(self, tmp_path: Path) -> None:
        original = TrainingConfig(
            num_epochs=10,
            batch_size=4,
            learning_rate=5e-5,
            lora=LoRAConfig(rank=32, alpha=64, use_dora=True),
        )
        path = tmp_path / "rt.json"
        save_config(original, path)
        loaded = load_training_config(path)
        assert loaded.num_epochs == 10
        assert loaded.batch_size == 4
        assert loaded.learning_rate == 5e-5
        assert loaded.lora.rank == 32
        assert loaded.lora.alpha == 64


# ── Quantization Types ──────────────────────────────────────────────────────


class TestQuantizationType:
    def test_all_types(self) -> None:
        assert len(QuantizationType) == 5
        assert QuantizationType.Q4_K_M.value == "q4_k_m"
        assert QuantizationType.Q8_0.value == "q8_0"
        assert QuantizationType.F16.value == "f16"


# ── Training Summary ────────────────────────────────────────────────────────


class TestGetTrainingSummary:
    def test_empty_directory(self, tmp_path: Path) -> None:
        summary = get_training_summary(tmp_path)
        assert summary["has_config"] is False
        assert summary["has_adapter"] is False
        assert summary["has_gguf"] is False

    def test_with_config(self, tmp_path: Path) -> None:
        config = TrainingConfig()
        save_config(config, tmp_path / "training_config.json")
        summary = get_training_summary(tmp_path)
        assert summary["has_config"] is True
        assert "config" in summary

    def test_with_adapter(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_text("fake", encoding="utf-8")
        summary = get_training_summary(tmp_path)
        assert summary["has_adapter"] is True
        assert "adapter_model.safetensors" in summary["adapter_files"]

    def test_with_gguf(self, tmp_path: Path) -> None:
        gguf_dir = tmp_path / "gguf"
        gguf_dir.mkdir()
        gguf_file = gguf_dir / "model.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)
        summary = get_training_summary(tmp_path)
        assert summary["has_gguf"] is True
        assert "gguf_path" in summary
