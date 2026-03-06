"""Tests for Phase 7: Vision Pipeline.

Covers: VisionResult, VisionTask, MoondreamModel, QwenVLModel, VisionPipeline,
PipelineResult, personality injection, memory integration, task routing.

All tests are sandboxed — no real model loading, no real GPU inference,
no real Ollama calls.  Everything is mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ara.config.settings import AraSettings
from ara.utils.errors import VisionError
from ara.vision.moondream import MoondreamModel, VisionResult, VisionTask
from ara.vision.pipeline import (
    _TASK_COMPLEXITY,
    PipelineResult,
    VisionComplexity,
    VisionPipeline,
    VisionTaskType,
)
from ara.vision.qwen_vl import QwenVLModel

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> AraSettings:
    return AraSettings(env="testing")


@pytest.fixture
def settings_vision_off() -> AraSettings:
    s = AraSettings(env="testing")
    s.vision.vision_always_on = False
    return s


@pytest.fixture
def mock_image() -> MagicMock:
    """Create a fake PIL Image."""
    img = MagicMock()
    img.width = 640
    img.height = 480
    img.convert.return_value = img
    return img


@pytest.fixture
def sample_result() -> VisionResult:
    return VisionResult(
        task=VisionTask.CAPTION,
        text="A dog sitting on a couch.",
        model_name="moondream2 (test)",
        inference_time_ms=150.0,
        image_size=(640, 480),
    )


def _make_moondream(settings: AraSettings) -> MoondreamModel:
    """Create a MoondreamModel with mocked internals."""
    model = MoondreamModel(settings)
    model._loaded = True
    model._device = "cpu"
    model._model = MagicMock()
    model._tokenizer = MagicMock()
    return model


def _make_qwen_vl(settings: AraSettings) -> QwenVLModel:
    """Create a QwenVLModel marked as loaded."""
    model = QwenVLModel(settings, ollama_model="qwen2.5-vl:7b")
    model._loaded = True
    return model


# ══════════════════════════════════════════════════════════════════════════════
# VisionResult & VisionTask
# ══════════════════════════════════════════════════════════════════════════════


class TestVisionTask:
    """VisionTask enum tests."""

    def test_all_values(self) -> None:
        assert VisionTask.CAPTION == "caption"
        assert VisionTask.VQA == "vqa"
        assert VisionTask.DETECT == "detect"

    def test_is_string_enum(self) -> None:
        assert isinstance(VisionTask.CAPTION, str)


class TestVisionResult:
    """VisionResult dataclass tests."""

    def test_create(self, sample_result: VisionResult) -> None:
        assert sample_result.task == VisionTask.CAPTION
        assert sample_result.text == "A dog sitting on a couch."
        assert sample_result.model_name == "moondream2 (test)"
        assert sample_result.inference_time_ms == 150.0
        assert sample_result.image_size == (640, 480)

    def test_defaults(self) -> None:
        r = VisionResult(
            task=VisionTask.VQA,
            text="yes",
            model_name="test",
            inference_time_ms=50.0,
        )
        assert r.image_size == (0, 0)
        assert r.metadata == {}

    def test_frozen(self, sample_result: VisionResult) -> None:
        with pytest.raises(AttributeError):
            sample_result.text = "changed"  # type: ignore[misc]

    def test_with_metadata(self) -> None:
        r = VisionResult(
            task=VisionTask.DETECT,
            text="found 2 people",
            model_name="test",
            inference_time_ms=100.0,
            metadata={"target": "person"},
        )
        assert r.metadata["target"] == "person"


# ══════════════════════════════════════════════════════════════════════════════
# MoondreamModel
# ══════════════════════════════════════════════════════════════════════════════


class TestMoondreamProperties:
    """MoondreamModel property tests."""

    def test_initial_state(self, settings: AraSettings) -> None:
        model = MoondreamModel(settings)
        assert not model.loaded
        assert model.device == "cpu"
        assert "moondream2" in model.model_name

    def test_loaded_state(self, settings: AraSettings) -> None:
        model = _make_moondream(settings)
        assert model.loaded
        assert model.device == "cpu"

    def test_get_stats(self, settings: AraSettings) -> None:
        model = MoondreamModel(settings)
        stats = model.get_stats()
        assert stats["loaded"] is False
        assert "model_id" in stats
        assert "model_name" in stats
        assert "device" in stats


class TestMoondreamLoad:
    """MoondreamModel load/unload tests."""

    @pytest.mark.asyncio
    async def test_load_already_loaded_noop(self, settings: AraSettings) -> None:
        model = _make_moondream(settings)
        # Should not raise
        await model.load()

    @pytest.mark.asyncio
    async def test_load_import_error(self, settings: AraSettings) -> None:
        model = MoondreamModel(settings)
        with (
            patch.dict("sys.modules", {"torch": None}),
            patch("builtins.__import__", side_effect=ImportError("no torch")),
            pytest.raises(VisionError, match="Required package not installed"),
        ):
            await model.load()

    @pytest.mark.asyncio
    async def test_unload_when_not_loaded(self, settings: AraSettings) -> None:
        model = MoondreamModel(settings)
        # Should not raise
        await model.unload()

    @pytest.mark.asyncio
    async def test_unload_clears_state(self, settings: AraSettings) -> None:
        model = _make_moondream(settings)
        with patch("ara.vision.moondream.torch", create=True) as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            await model.unload()
        assert not model.loaded
        assert model._model is None
        assert model._tokenizer is None


class TestMoondreamEnsureLoaded:
    """_ensure_loaded guard tests."""

    def test_raises_when_not_loaded(self, settings: AraSettings) -> None:
        model = MoondreamModel(settings)
        with pytest.raises(VisionError, match="not loaded"):
            model._ensure_loaded()

    def test_passes_when_loaded(self, settings: AraSettings) -> None:
        model = _make_moondream(settings)
        model._ensure_loaded()  # Should not raise


class TestMoondreamCaption:
    """MoondreamModel caption tests."""

    @pytest.mark.asyncio
    async def test_caption_returns_result(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        model = _make_moondream(settings)
        model._model.encode_image.return_value = "encoded"
        model._model.answer_question.return_value = "A cat on a mat."

        result = await model.caption(mock_image)
        assert result.task == VisionTask.CAPTION
        assert result.text == "A cat on a mat."
        assert result.image_size == (640, 480)
        assert result.inference_time_ms >= 0.0

    @pytest.mark.asyncio
    async def test_caption_strips_whitespace(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        model = _make_moondream(settings)
        model._model.encode_image.return_value = "enc"
        model._model.answer_question.return_value = "  padded text  \n"

        result = await model.caption(mock_image)
        assert result.text == "padded text"

    @pytest.mark.asyncio
    async def test_caption_error_raises_vision_error(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        model = _make_moondream(settings)
        model._model.encode_image.side_effect = RuntimeError("GPU OOM")

        with pytest.raises(VisionError, match="Caption failed"):
            await model.caption(mock_image)

    @pytest.mark.asyncio
    async def test_caption_not_loaded_raises(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        model = MoondreamModel(settings)
        with pytest.raises(VisionError, match="not loaded"):
            await model.caption(mock_image)


class TestMoondreamVQA:
    """MoondreamModel VQA tests."""

    @pytest.mark.asyncio
    async def test_ask_returns_result(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        model = _make_moondream(settings)
        model._model.encode_image.return_value = "enc"
        model._model.answer_question.return_value = "It is red."

        result = await model.ask(mock_image, "What color is the car?")
        assert result.task == VisionTask.VQA
        assert result.text == "It is red."
        assert result.metadata["question"] == "What color is the car?"

    @pytest.mark.asyncio
    async def test_ask_error_raises(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        model = _make_moondream(settings)
        model._model.encode_image.side_effect = RuntimeError("fail")

        with pytest.raises(VisionError, match="VQA failed"):
            await model.ask(mock_image, "What is this?")


class TestMoondreamDetect:
    """MoondreamModel detection tests."""

    @pytest.mark.asyncio
    async def test_detect_returns_result(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        model = _make_moondream(settings)
        model._model.encode_image.return_value = "enc"
        model._model.answer_question.return_value = "Two people in center of image."

        result = await model.detect(mock_image, "person")
        assert result.task == VisionTask.DETECT
        assert "people" in result.text.lower()
        assert result.metadata["target"] == "person"

    @pytest.mark.asyncio
    async def test_detect_error_raises(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        model = _make_moondream(settings)
        model._model.encode_image.side_effect = RuntimeError("fail")

        with pytest.raises(VisionError, match="Detection failed"):
            await model.detect(mock_image, "car")


class TestMoondreamCaptionFromPath:
    """MoondreamModel caption_from_path tests."""

    @pytest.mark.asyncio
    async def test_nonexistent_path_raises(self, settings: AraSettings, tmp_path: Any) -> None:
        model = _make_moondream(settings)
        fake_path = tmp_path / "nonexistent.jpg"

        with pytest.raises(VisionError, match="not found"):
            await model.caption_from_path(fake_path)

    @pytest.mark.asyncio
    async def test_invalid_image_raises(self, settings: AraSettings, tmp_path: Any) -> None:
        model = _make_moondream(settings)
        bad_file = tmp_path / "bad.jpg"
        bad_file.write_text("not an image")

        with pytest.raises(VisionError, match="Failed to open image"):
            await model.caption_from_path(bad_file)


# ══════════════════════════════════════════════════════════════════════════════
# QwenVLModel
# ══════════════════════════════════════════════════════════════════════════════


class TestQwenVLProperties:
    """QwenVLModel property tests."""

    def test_initial_state(self, settings: AraSettings) -> None:
        model = QwenVLModel(settings)
        assert not model.loaded
        assert "qwen-vl" in model.model_name

    def test_custom_model_name(self, settings: AraSettings) -> None:
        model = QwenVLModel(settings, ollama_model="custom:latest")
        assert "custom:latest" in model.model_name

    def test_get_stats(self, settings: AraSettings) -> None:
        model = QwenVLModel(settings)
        stats = model.get_stats()
        assert stats["loaded"] is False
        assert "model_name" in stats
        assert "ollama_model" in stats
        assert "ollama_base" in stats


class TestQwenVLLoad:
    """QwenVLModel load/unload tests."""

    @pytest.mark.asyncio
    async def test_load_already_loaded_noop(self, settings: AraSettings) -> None:
        model = _make_qwen_vl(settings)
        await model.load()  # Should not raise or make requests

    @pytest.mark.asyncio
    async def test_load_model_exists(self, settings: AraSettings) -> None:
        model = QwenVLModel(settings, ollama_model="qwen2.5-vl:7b")

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("ara.vision.qwen_vl.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            await model.load()
            assert model.loaded

    @pytest.mark.asyncio
    async def test_load_connection_error(self, settings: AraSettings) -> None:
        import httpx

        model = QwenVLModel(settings, ollama_model="qwen2.5-vl:7b")

        with patch("ara.vision.qwen_vl.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            with pytest.raises(VisionError, match="Cannot connect to Ollama"):
                await model.load()

    @pytest.mark.asyncio
    async def test_unload_when_not_loaded(self, settings: AraSettings) -> None:
        model = QwenVLModel(settings)
        await model.unload()  # Should not raise

    @pytest.mark.asyncio
    async def test_unload_sends_keep_alive_zero(self, settings: AraSettings) -> None:
        model = _make_qwen_vl(settings)

        with patch("ara.vision.qwen_vl.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            await model.unload()
            assert not model.loaded

            # Verify keep_alive=0 was sent
            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["keep_alive"] == 0


class TestQwenVLEnsureLoaded:
    """_ensure_loaded guard tests."""

    def test_raises_when_not_loaded(self, settings: AraSettings) -> None:
        model = QwenVLModel(settings)
        with pytest.raises(VisionError, match="not loaded"):
            model._ensure_loaded()


class TestQwenVLImageConversion:
    """_image_to_base64 tests."""

    def test_converts_image(self, mock_image: MagicMock) -> None:
        import base64

        # Mock the save method to write some bytes
        def fake_save(buf: Any, format: str) -> None:
            buf.write(b"fake-png-data")

        mock_image.save = fake_save

        result = QwenVLModel._image_to_base64(mock_image)
        decoded = base64.b64decode(result)
        assert decoded == b"fake-png-data"


class TestQwenVLAnalyze:
    """QwenVLModel analyze tests."""

    @pytest.mark.asyncio
    async def test_analyze_returns_result(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        model = _make_qwen_vl(settings)

        # Mock image conversion
        mock_image.save = lambda buf, format: buf.write(b"data")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "response": "A landscape with mountains.",
            "total_duration": 5000000000,
            "eval_count": 15,
        }

        with patch("ara.vision.qwen_vl.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await model.analyze(mock_image, "Describe this image.")
            assert result.task == VisionTask.VQA
            assert result.text == "A landscape with mountains."
            assert result.metadata["eval_count"] == 15

    @pytest.mark.asyncio
    async def test_analyze_not_loaded_raises(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        model = QwenVLModel(settings)
        with pytest.raises(VisionError, match="not loaded"):
            await model.analyze(mock_image)

    @pytest.mark.asyncio
    async def test_analyze_connection_error(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        import httpx

        model = _make_qwen_vl(settings)
        mock_image.save = lambda buf, format: buf.write(b"data")

        with patch("ara.vision.qwen_vl.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            with pytest.raises(VisionError, match="Cannot connect to Ollama"):
                await model.analyze(mock_image)


class TestQwenVLOCR:
    """QwenVLModel OCR tests."""

    @pytest.mark.asyncio
    async def test_ocr_calls_analyze_with_ocr_prompt(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        model = _make_qwen_vl(settings)

        fake_result = VisionResult(
            task=VisionTask.VQA,
            text="Invoice #12345\nTotal: $99.99",
            model_name="qwen-vl",
            inference_time_ms=200.0,
        )

        with patch.object(model, "analyze", new_callable=AsyncMock, return_value=fake_result):
            result = await model.ocr(mock_image)
            assert result.text == "Invoice #12345\nTotal: $99.99"

            # Verify prompt contains OCR-related words
            call_args = model.analyze.call_args
            prompt_used = call_args[1].get(
                "prompt", call_args[0][1] if len(call_args[0]) > 1 else ""
            )
            assert "text" in prompt_used.lower()


class TestQwenVLDocumentAnalysis:
    """QwenVLModel analyze_document tests."""

    @pytest.mark.asyncio
    async def test_document_calls_analyze(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        model = _make_qwen_vl(settings)

        fake_result = VisionResult(
            task=VisionTask.VQA,
            text="This is a tax form.",
            model_name="qwen-vl",
            inference_time_ms=300.0,
        )

        with patch.object(model, "analyze", new_callable=AsyncMock, return_value=fake_result):
            result = await model.analyze_document(mock_image)
            assert result.text == "This is a tax form."


# ══════════════════════════════════════════════════════════════════════════════
# VisionTaskType & VisionComplexity
# ══════════════════════════════════════════════════════════════════════════════


class TestVisionTaskType:
    """VisionTaskType enum tests."""

    def test_all_values(self) -> None:
        assert VisionTaskType.CAPTION == "caption"
        assert VisionTaskType.VQA == "vqa"
        assert VisionTaskType.DETECT == "detect"
        assert VisionTaskType.OCR == "ocr"
        assert VisionTaskType.DOCUMENT == "document"
        assert VisionTaskType.DETAILED == "detailed"

    def test_is_string_enum(self) -> None:
        assert isinstance(VisionTaskType.CAPTION, str)


class TestVisionComplexity:
    """VisionComplexity enum tests."""

    def test_values(self) -> None:
        assert VisionComplexity.SIMPLE == "simple"
        assert VisionComplexity.COMPLEX == "complex"


class TestTaskComplexityMapping:
    """Task-to-complexity routing map tests."""

    def test_simple_tasks(self) -> None:
        assert _TASK_COMPLEXITY[VisionTaskType.CAPTION] == VisionComplexity.SIMPLE
        assert _TASK_COMPLEXITY[VisionTaskType.VQA] == VisionComplexity.SIMPLE
        assert _TASK_COMPLEXITY[VisionTaskType.DETECT] == VisionComplexity.SIMPLE

    def test_complex_tasks(self) -> None:
        assert _TASK_COMPLEXITY[VisionTaskType.OCR] == VisionComplexity.COMPLEX
        assert _TASK_COMPLEXITY[VisionTaskType.DOCUMENT] == VisionComplexity.COMPLEX
        assert _TASK_COMPLEXITY[VisionTaskType.DETAILED] == VisionComplexity.COMPLEX

    def test_all_task_types_mapped(self) -> None:
        for task_type in VisionTaskType:
            assert task_type in _TASK_COMPLEXITY


# ══════════════════════════════════════════════════════════════════════════════
# PipelineResult
# ══════════════════════════════════════════════════════════════════════════════


class TestPipelineResult:
    """PipelineResult dataclass tests."""

    def test_create(self, sample_result: VisionResult) -> None:
        pr = PipelineResult(
            raw_result=sample_result,
            personality_text="Oh nice, a dog on the couch!",
            task_type=VisionTaskType.CAPTION,
            model_used="moondream",
            total_time_ms=200.0,
        )
        assert pr.text == "Oh nice, a dog on the couch!"
        assert pr.model_used == "moondream"
        assert pr.total_time_ms == 200.0

    def test_text_falls_back_to_raw(self, sample_result: VisionResult) -> None:
        pr = PipelineResult(
            raw_result=sample_result,
            personality_text="",
            task_type=VisionTaskType.CAPTION,
            model_used="moondream",
            total_time_ms=100.0,
        )
        assert pr.text == sample_result.text

    def test_defaults(self, sample_result: VisionResult) -> None:
        pr = PipelineResult(
            raw_result=sample_result,
            personality_text="",
            task_type=VisionTaskType.CAPTION,
            model_used="moondream",
            total_time_ms=100.0,
        )
        assert pr.stored_in_memory is False
        assert pr.metadata == {}

    def test_frozen(self, sample_result: VisionResult) -> None:
        pr = PipelineResult(
            raw_result=sample_result,
            personality_text="text",
            task_type=VisionTaskType.CAPTION,
            model_used="moondream",
            total_time_ms=100.0,
        )
        with pytest.raises(AttributeError):
            pr.model_used = "other"  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# VisionPipeline
# ══════════════════════════════════════════════════════════════════════════════


class TestPipelineInit:
    """VisionPipeline initialization tests."""

    def test_default_init(self, settings: AraSettings) -> None:
        pipeline = VisionPipeline(settings)
        assert not pipeline.started
        assert not pipeline.personality_enabled

    def test_with_llm_client(self, settings: AraSettings) -> None:
        mock_llm = MagicMock()
        pipeline = VisionPipeline(settings, llm_client=mock_llm)
        assert pipeline.personality_enabled

    def test_with_memory_store(self, settings: AraSettings) -> None:
        mock_fn = AsyncMock()
        pipeline = VisionPipeline(settings, memory_store_fn=mock_fn)
        stats = pipeline.get_stats()
        assert stats["memory_enabled"] is True


class TestPipelineLifecycle:
    """VisionPipeline start/stop tests."""

    @pytest.mark.asyncio
    async def test_start_loads_moondream(self, settings: AraSettings) -> None:
        pipeline = VisionPipeline(settings)

        with patch.object(pipeline._moondream, "load", new_callable=AsyncMock) as mock_load:
            await pipeline.start()
            assert pipeline.started
            mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_already_started_noop(self, settings: AraSettings) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True
        # Should not raise
        await pipeline.start()

    @pytest.mark.asyncio
    async def test_start_vision_off_skips_moondream(
        self,
        settings_vision_off: AraSettings,
    ) -> None:
        pipeline = VisionPipeline(settings_vision_off)

        with patch.object(pipeline._moondream, "load", new_callable=AsyncMock) as mock_load:
            await pipeline.start()
            assert pipeline.started
            mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_unloads_both_models(self, settings: AraSettings) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True

        with (
            patch.object(pipeline._moondream, "unload", new_callable=AsyncMock) as mock_moon,
            patch.object(pipeline._qwen_vl, "unload", new_callable=AsyncMock) as mock_qwen,
        ):
            await pipeline.stop()
            assert not pipeline.started
            mock_moon.assert_called_once()
            mock_qwen.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_when_not_started_noop(self, settings: AraSettings) -> None:
        pipeline = VisionPipeline(settings)
        await pipeline.stop()  # Should not raise


class TestPipelineRouting:
    """VisionPipeline task routing tests."""

    def test_get_complexity_simple(self, settings: AraSettings) -> None:
        pipeline = VisionPipeline(settings)
        assert pipeline._get_complexity(VisionTaskType.CAPTION) == VisionComplexity.SIMPLE
        assert pipeline._get_complexity(VisionTaskType.VQA) == VisionComplexity.SIMPLE
        assert pipeline._get_complexity(VisionTaskType.DETECT) == VisionComplexity.SIMPLE

    def test_get_complexity_complex(self, settings: AraSettings) -> None:
        pipeline = VisionPipeline(settings)
        assert pipeline._get_complexity(VisionTaskType.OCR) == VisionComplexity.COMPLEX
        assert pipeline._get_complexity(VisionTaskType.DOCUMENT) == VisionComplexity.COMPLEX
        assert pipeline._get_complexity(VisionTaskType.DETAILED) == VisionComplexity.COMPLEX


class TestPipelineEnsureStarted:
    """_ensure_started guard tests."""

    def test_raises_when_not_started(self, settings: AraSettings) -> None:
        pipeline = VisionPipeline(settings)
        with pytest.raises(VisionError, match="not started"):
            pipeline._ensure_started()

    def test_passes_when_started(self, settings: AraSettings) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True
        pipeline._ensure_started()  # Should not raise


class TestPipelineAnalyze:
    """VisionPipeline analyze method tests."""

    @pytest.mark.asyncio
    async def test_caption_routes_to_moondream(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
        sample_result: VisionResult,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True

        with patch.object(
            pipeline,
            "_run_simple_task",
            new_callable=AsyncMock,
            return_value=sample_result,
        ):
            result = await pipeline.analyze(mock_image, VisionTaskType.CAPTION)
            assert result.model_used == "moondream"
            assert result.task_type == VisionTaskType.CAPTION

    @pytest.mark.asyncio
    async def test_ocr_routes_to_qwen_vl(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True

        ocr_result = VisionResult(
            task=VisionTask.VQA,
            text="Hello World",
            model_name="qwen-vl",
            inference_time_ms=200.0,
        )

        with patch.object(
            pipeline,
            "_run_complex_task",
            new_callable=AsyncMock,
            return_value=ocr_result,
        ):
            result = await pipeline.analyze(mock_image, VisionTaskType.OCR)
            assert result.model_used == "qwen_vl"
            assert result.task_type == VisionTaskType.OCR

    @pytest.mark.asyncio
    async def test_analyze_not_started_raises(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        pipeline = VisionPipeline(settings)
        with pytest.raises(VisionError, match="not started"):
            await pipeline.analyze(mock_image)

    @pytest.mark.asyncio
    async def test_analyze_with_personality(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
        sample_result: VisionResult,
    ) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="Oh look, a cute doggo on the couch!")

        pipeline = VisionPipeline(settings, llm_client=mock_llm)
        pipeline._started = True

        with patch.object(
            pipeline,
            "_run_simple_task",
            new_callable=AsyncMock,
            return_value=sample_result,
        ):
            result = await pipeline.analyze(
                mock_image,
                VisionTaskType.CAPTION,
                personality=True,
            )
            assert result.personality_text == "Oh look, a cute doggo on the couch!"
            assert result.text == "Oh look, a cute doggo on the couch!"

    @pytest.mark.asyncio
    async def test_analyze_personality_disabled(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
        sample_result: VisionResult,
    ) -> None:
        pipeline = VisionPipeline(settings)  # No LLM client = no personality
        pipeline._started = True

        with patch.object(
            pipeline,
            "_run_simple_task",
            new_callable=AsyncMock,
            return_value=sample_result,
        ):
            result = await pipeline.analyze(
                mock_image,
                VisionTaskType.CAPTION,
                personality=True,
            )
            assert result.personality_text == ""
            assert result.text == sample_result.text

    @pytest.mark.asyncio
    async def test_analyze_with_memory_store(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
        sample_result: VisionResult,
    ) -> None:
        mock_store = AsyncMock()
        pipeline = VisionPipeline(settings, memory_store_fn=mock_store)
        pipeline._started = True

        with patch.object(
            pipeline,
            "_run_simple_task",
            new_callable=AsyncMock,
            return_value=sample_result,
        ):
            result = await pipeline.analyze(
                mock_image,
                VisionTaskType.CAPTION,
                store_memory=True,
            )
            assert result.stored_in_memory is True
            mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_memory_store_failure_does_not_crash(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
        sample_result: VisionResult,
    ) -> None:
        mock_store = AsyncMock(side_effect=RuntimeError("DB error"))
        pipeline = VisionPipeline(settings, memory_store_fn=mock_store)
        pipeline._started = True

        with patch.object(
            pipeline,
            "_run_simple_task",
            new_callable=AsyncMock,
            return_value=sample_result,
        ):
            result = await pipeline.analyze(
                mock_image,
                VisionTaskType.CAPTION,
                store_memory=True,
            )
            assert result.stored_in_memory is False

    @pytest.mark.asyncio
    async def test_analyze_total_time_tracked(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
        sample_result: VisionResult,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True

        with patch.object(
            pipeline,
            "_run_simple_task",
            new_callable=AsyncMock,
            return_value=sample_result,
        ):
            result = await pipeline.analyze(mock_image, VisionTaskType.CAPTION)
            assert result.total_time_ms >= 0.0


class TestPipelinePersonality:
    """Personality injection tests."""

    @pytest.mark.asyncio
    async def test_personality_for_caption(self, settings: AraSettings) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="Nice sunset, huh?")

        pipeline = VisionPipeline(settings, llm_client=mock_llm)
        result = await pipeline._apply_personality(
            "A sunset over the ocean.", VisionTaskType.CAPTION
        )
        assert result == "Nice sunset, huh?"

    @pytest.mark.asyncio
    async def test_personality_for_vqa(self, settings: AraSettings) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="Yeah, it's red.")

        pipeline = VisionPipeline(settings, llm_client=mock_llm)
        result = await pipeline._apply_personality(
            "The car is red.",
            VisionTaskType.VQA,
            question="What color is it?",
        )
        assert result == "Yeah, it's red."

    @pytest.mark.asyncio
    async def test_personality_for_ocr(self, settings: AraSettings) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="Here's what I found: Invoice #123")

        pipeline = VisionPipeline(settings, llm_client=mock_llm)
        result = await pipeline._apply_personality("Invoice #123", VisionTaskType.OCR)
        assert "Invoice" in result

    @pytest.mark.asyncio
    async def test_personality_fallback_on_llm_error(self, settings: AraSettings) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

        pipeline = VisionPipeline(settings, llm_client=mock_llm)
        result = await pipeline._apply_personality("raw text", VisionTaskType.CAPTION)
        assert result == "raw text"

    @pytest.mark.asyncio
    async def test_personality_empty_text_passthrough(self, settings: AraSettings) -> None:
        mock_llm = AsyncMock()
        pipeline = VisionPipeline(settings, llm_client=mock_llm)
        result = await pipeline._apply_personality("", VisionTaskType.CAPTION)
        assert result == ""
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_personality_disabled_returns_raw(self, settings: AraSettings) -> None:
        pipeline = VisionPipeline(settings)  # No LLM
        result = await pipeline._apply_personality("raw text", VisionTaskType.CAPTION)
        assert result == "raw text"


class TestPipelineMemoryIntegration:
    """Memory store integration tests."""

    @pytest.mark.asyncio
    async def test_store_calls_callback(self, settings: AraSettings) -> None:
        mock_fn = AsyncMock()
        pipeline = VisionPipeline(settings, memory_store_fn=mock_fn)

        stored = await pipeline._store_in_memory("A cat photo", VisionTaskType.CAPTION, "cat.jpg")
        assert stored is True
        mock_fn.assert_called_once()

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["text"] == "A cat photo"
        assert call_kwargs["source_file"] == "cat.jpg"
        assert call_kwargs["metadata"]["task_type"] == "caption"

    @pytest.mark.asyncio
    async def test_store_without_callback_returns_false(self, settings: AraSettings) -> None:
        pipeline = VisionPipeline(settings)
        stored = await pipeline._store_in_memory("text", VisionTaskType.CAPTION)
        assert stored is False

    @pytest.mark.asyncio
    async def test_store_with_extra_metadata(self, settings: AraSettings) -> None:
        mock_fn = AsyncMock()
        pipeline = VisionPipeline(settings, memory_store_fn=mock_fn)

        await pipeline._store_in_memory(
            "text",
            VisionTaskType.CAPTION,
            extra_metadata={"model": "moondream"},
        )
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["metadata"]["model"] == "moondream"

    @pytest.mark.asyncio
    async def test_store_error_returns_false(self, settings: AraSettings) -> None:
        mock_fn = AsyncMock(side_effect=RuntimeError("DB fail"))
        pipeline = VisionPipeline(settings, memory_store_fn=mock_fn)

        stored = await pipeline._store_in_memory("text", VisionTaskType.CAPTION)
        assert stored is False

    @pytest.mark.asyncio
    async def test_store_default_source_file(self, settings: AraSettings) -> None:
        mock_fn = AsyncMock()
        pipeline = VisionPipeline(settings, memory_store_fn=mock_fn)

        await pipeline._store_in_memory("text", VisionTaskType.OCR)
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["source_file"] == "vision_ocr"


class TestPipelineConvenienceMethods:
    """Test convenience methods (describe, ask, detect, ocr, etc.)."""

    @pytest.mark.asyncio
    async def test_describe(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
        sample_result: VisionResult,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True

        with patch.object(
            pipeline,
            "analyze",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = PipelineResult(
                raw_result=sample_result,
                personality_text="",
                task_type=VisionTaskType.CAPTION,
                model_used="moondream",
                total_time_ms=100.0,
            )
            result = await pipeline.describe(mock_image)
            assert result.task_type == VisionTaskType.CAPTION

    @pytest.mark.asyncio
    async def test_ask(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
        sample_result: VisionResult,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True

        with patch.object(pipeline, "analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = PipelineResult(
                raw_result=sample_result,
                personality_text="",
                task_type=VisionTaskType.VQA,
                model_used="moondream",
                total_time_ms=100.0,
            )
            await pipeline.ask(mock_image, "What is this?")
            mock_analyze.assert_called_once()
            call_kwargs = mock_analyze.call_args
            assert call_kwargs[1]["question"] == "What is this?"

    @pytest.mark.asyncio
    async def test_detect(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
        sample_result: VisionResult,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True

        with patch.object(pipeline, "analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = PipelineResult(
                raw_result=sample_result,
                personality_text="",
                task_type=VisionTaskType.DETECT,
                model_used="moondream",
                total_time_ms=100.0,
            )
            await pipeline.detect(mock_image, "cat")
            call_kwargs = mock_analyze.call_args
            assert call_kwargs[1]["target"] == "cat"

    @pytest.mark.asyncio
    async def test_ocr(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
        sample_result: VisionResult,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True

        with patch.object(pipeline, "analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = PipelineResult(
                raw_result=sample_result,
                personality_text="",
                task_type=VisionTaskType.OCR,
                model_used="qwen_vl",
                total_time_ms=100.0,
            )
            result = await pipeline.ocr(mock_image)
            assert result.task_type == VisionTaskType.OCR

    @pytest.mark.asyncio
    async def test_analyze_document(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
        sample_result: VisionResult,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True

        with patch.object(pipeline, "analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = PipelineResult(
                raw_result=sample_result,
                personality_text="",
                task_type=VisionTaskType.DOCUMENT,
                model_used="qwen_vl",
                total_time_ms=100.0,
            )
            result = await pipeline.analyze_document(mock_image)
            assert result.task_type == VisionTaskType.DOCUMENT

    @pytest.mark.asyncio
    async def test_detailed_analysis(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
        sample_result: VisionResult,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True

        with patch.object(pipeline, "analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = PipelineResult(
                raw_result=sample_result,
                personality_text="",
                task_type=VisionTaskType.DETAILED,
                model_used="qwen_vl",
                total_time_ms=100.0,
            )
            await pipeline.detailed_analysis(mock_image, "Count all objects.")
            call_kwargs = mock_analyze.call_args
            assert call_kwargs[1]["prompt"] == "Count all objects."


class TestPipelineDescribeFromPath:
    """describe_from_path tests."""

    @pytest.mark.asyncio
    async def test_nonexistent_path_raises(self, settings: AraSettings, tmp_path: Any) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True

        with pytest.raises(VisionError, match="not found"):
            await pipeline.describe_from_path(tmp_path / "missing.jpg")

    @pytest.mark.asyncio
    async def test_invalid_image_raises(self, settings: AraSettings, tmp_path: Any) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._started = True

        bad_file = tmp_path / "bad.png"
        bad_file.write_text("not an image")

        with pytest.raises(VisionError, match="Failed to open image"):
            await pipeline.describe_from_path(bad_file)


class TestPipelineGetStats:
    """get_stats tests."""

    def test_stats_structure(self, settings: AraSettings) -> None:
        pipeline = VisionPipeline(settings)
        stats = pipeline.get_stats()

        assert "started" in stats
        assert "personality_enabled" in stats
        assert "memory_enabled" in stats
        assert "moondream" in stats
        assert "qwen_vl" in stats

    def test_stats_reflect_state(self, settings: AraSettings) -> None:
        mock_llm = MagicMock()
        mock_fn = AsyncMock()
        pipeline = VisionPipeline(settings, llm_client=mock_llm, memory_store_fn=mock_fn)

        stats = pipeline.get_stats()
        assert stats["personality_enabled"] is True
        assert stats["memory_enabled"] is True
        assert stats["started"] is False


class TestPipelineSimpleTaskExecution:
    """_run_simple_task internal tests."""

    @pytest.mark.asyncio
    async def test_caption_calls_moondream_caption(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
        sample_result: VisionResult,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._moondream._loaded = True

        with patch.object(
            pipeline._moondream,
            "caption",
            new_callable=AsyncMock,
            return_value=sample_result,
        ):
            result = await pipeline._run_simple_task(mock_image, VisionTaskType.CAPTION)
            assert result.task == VisionTask.CAPTION

    @pytest.mark.asyncio
    async def test_vqa_requires_question(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._moondream._loaded = True

        with pytest.raises(VisionError, match="VQA requires a question"):
            await pipeline._run_simple_task(mock_image, VisionTaskType.VQA)

    @pytest.mark.asyncio
    async def test_detect_requires_target(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._moondream._loaded = True

        with pytest.raises(VisionError, match="Detection requires a target"):
            await pipeline._run_simple_task(mock_image, VisionTaskType.DETECT)


class TestPipelineComplexTaskExecution:
    """_run_complex_task internal tests."""

    @pytest.mark.asyncio
    async def test_ocr_calls_qwen_ocr(self, settings: AraSettings, mock_image: MagicMock) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._qwen_vl._loaded = True

        fake_result = VisionResult(
            task=VisionTask.VQA,
            text="text",
            model_name="qwen",
            inference_time_ms=100.0,
        )

        with patch.object(
            pipeline._qwen_vl,
            "ocr",
            new_callable=AsyncMock,
            return_value=fake_result,
        ):
            result = await pipeline._run_complex_task(mock_image, VisionTaskType.OCR)
            assert result.text == "text"

    @pytest.mark.asyncio
    async def test_document_calls_qwen_analyze_document(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._qwen_vl._loaded = True

        fake_result = VisionResult(
            task=VisionTask.VQA,
            text="doc",
            model_name="qwen",
            inference_time_ms=100.0,
        )

        with patch.object(
            pipeline._qwen_vl,
            "analyze_document",
            new_callable=AsyncMock,
            return_value=fake_result,
        ):
            result = await pipeline._run_complex_task(mock_image, VisionTaskType.DOCUMENT)
            assert result.text == "doc"

    @pytest.mark.asyncio
    async def test_detailed_uses_custom_prompt(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        pipeline = VisionPipeline(settings)
        pipeline._qwen_vl._loaded = True

        fake_result = VisionResult(
            task=VisionTask.VQA,
            text="detailed",
            model_name="qwen",
            inference_time_ms=100.0,
        )

        with patch.object(
            pipeline._qwen_vl,
            "analyze",
            new_callable=AsyncMock,
            return_value=fake_result,
        ):
            await pipeline._run_complex_task(
                mock_image,
                VisionTaskType.DETAILED,
                prompt="Count everything.",
            )
            call_args = pipeline._qwen_vl.analyze.call_args
            assert call_args[0][1] == "Count everything."


# ══════════════════════════════════════════════════════════════════════════════
# Integration-Style Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestPipelineIntegration:
    """Higher-level integration tests with mocked models."""

    @pytest.mark.asyncio
    async def test_full_caption_flow(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        """Full flow: start → caption → personality → memory → stop."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="Cute dog vibes!")
        mock_store = AsyncMock()

        pipeline = VisionPipeline(settings, llm_client=mock_llm, memory_store_fn=mock_store)

        caption_result = VisionResult(
            task=VisionTask.CAPTION,
            text="A dog on a couch.",
            model_name="moondream2",
            inference_time_ms=100.0,
            image_size=(640, 480),
        )

        with (
            patch.object(pipeline._moondream, "load", new_callable=AsyncMock),
            patch.object(pipeline._moondream, "unload", new_callable=AsyncMock),
            patch.object(pipeline._qwen_vl, "unload", new_callable=AsyncMock),
            patch.object(
                pipeline._moondream,
                "caption",
                new_callable=AsyncMock,
                return_value=caption_result,
            ),
        ):
            # Mark moondream as loaded after start
            pipeline._moondream._loaded = True

            await pipeline.start()
            result = await pipeline.describe(
                mock_image,
                personality=True,
                store_memory=True,
            )
            await pipeline.stop()

        assert result.text == "Cute dog vibes!"
        assert result.stored_in_memory is True
        assert result.model_used == "moondream"
        mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_ocr_flow(
        self,
        settings: AraSettings,
        mock_image: MagicMock,
    ) -> None:
        """Full flow: start → OCR (complex) → personality."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="Found this text: Invoice #42")

        pipeline = VisionPipeline(settings, llm_client=mock_llm)

        ocr_result = VisionResult(
            task=VisionTask.VQA,
            text="Invoice #42",
            model_name="qwen-vl",
            inference_time_ms=300.0,
            image_size=(800, 600),
        )

        with (
            patch.object(pipeline._moondream, "load", new_callable=AsyncMock),
            patch.object(pipeline._moondream, "unload", new_callable=AsyncMock),
            patch.object(pipeline._qwen_vl, "load", new_callable=AsyncMock),
            patch.object(pipeline._qwen_vl, "unload", new_callable=AsyncMock),
            patch.object(
                pipeline._qwen_vl,
                "ocr",
                new_callable=AsyncMock,
                return_value=ocr_result,
            ),
        ):
            pipeline._moondream._loaded = True

            await pipeline.start()

            # Mark qwen as loaded after ensure_qwen_vl call
            pipeline._qwen_vl._loaded = True

            result = await pipeline.ocr(mock_image, personality=True)
            await pipeline.stop()

        assert result.text == "Found this text: Invoice #42"
        assert result.model_used == "qwen_vl"

    @pytest.mark.asyncio
    async def test_exports_from_init(self) -> None:
        """Verify __init__.py exports all expected symbols."""
        from ara.vision import (
            MoondreamModel,
            PipelineResult,
            QwenVLModel,
            VisionComplexity,
            VisionPipeline,
            VisionResult,
            VisionTask,
            VisionTaskType,
        )

        assert MoondreamModel is not None
        assert QwenVLModel is not None
        assert VisionPipeline is not None
        assert PipelineResult is not None
        assert VisionResult is not None
        assert VisionTask is not None
        assert VisionTaskType is not None
        assert VisionComplexity is not None
