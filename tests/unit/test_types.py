"""Tests for ARA shared type definitions."""

from __future__ import annotations

from datetime import datetime

from ara.utils.types import (
    AudioSegment,
    ChatMessage,
    ConversationEndReason,
    ConversationTurn,
    EntityType,
    GraphEntity,
    GraphRelation,
    IntentType,
    MemoryChunk,
    MemorySource,
    MessageRole,
    RelationType,
    RetrievalContext,
    RetrievalResult,
    ToolCall,
    ToolResult,
    TranscriptionResult,
    VoicePipelineState,
    VRAMSnapshot,
    WebSearchResult,
)


class TestEnums:
    """Tests for enum types."""

    def test_message_role_values(self) -> None:
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.TOOL == "tool"

    def test_intent_type_values(self) -> None:
        assert IntentType.FACTUAL == "factual"
        assert IntentType.WEB_SEARCH == "web_search"
        assert IntentType.RESEARCH == "research"

    def test_conversation_end_reason_values(self) -> None:
        assert ConversationEndReason.SILENCE_TIMEOUT == "silence_timeout"
        assert ConversationEndReason.USER_FAREWELL == "user_farewell"

    def test_memory_source_values(self) -> None:
        assert MemorySource.DIARY == "diary"
        assert MemorySource.CONVERSATION == "conversation"
        assert MemorySource.WEB_RESEARCH == "web_research"

    def test_voice_pipeline_state_values(self) -> None:
        assert VoicePipelineState.IDLE == "idle"
        assert VoicePipelineState.SPEAKING == "speaking"


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_create_message(self) -> None:
        msg = ChatMessage(role=MessageRole.USER, content="Hello Ara")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello Ara"
        assert isinstance(msg.timestamp, datetime)

    def test_message_is_frozen(self) -> None:
        msg = ChatMessage(role=MessageRole.USER, content="test")
        try:
            msg.content = "changed"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass


class TestConversationTurn:
    """Tests for ConversationTurn dataclass."""

    def test_create_turn(self) -> None:
        turn = ConversationTurn(
            conversation_id="conv-001",
            user_message="What did we discuss yesterday?",
            assistant_message="We talked about your project deadline.",
            intent=IntentType.REMINISCING,
        )
        assert turn.conversation_id == "conv-001"
        assert turn.intent == IntentType.REMINISCING
        assert len(turn.turn_id) == 12

    def test_turn_generates_unique_ids(self) -> None:
        t1 = ConversationTurn()
        t2 = ConversationTurn()
        assert t1.turn_id != t2.turn_id


class TestMemoryChunk:
    """Tests for MemoryChunk dataclass."""

    def test_create_chunk(self) -> None:
        chunk = MemoryChunk(
            chunk_id="ch-001",
            text="Today was a good day.",
            source=MemorySource.DIARY,
            source_file="2024-01-15.md",
            section_title="Morning",
        )
        assert chunk.source == MemorySource.DIARY
        assert chunk.source_file == "2024-01-15.md"


class TestRetrievalTypes:
    """Tests for retrieval result types."""

    def test_retrieval_result(self) -> None:
        chunk = MemoryChunk(chunk_id="c1", text="text", source=MemorySource.DIARY)
        result = RetrievalResult(chunk=chunk, score=0.85, retrieval_method="hybrid")
        assert result.score == 0.85
        assert result.rerank_score is None

    def test_retrieval_context(self) -> None:
        chunk = MemoryChunk(chunk_id="c1", text="text", source=MemorySource.DIARY)
        result = RetrievalResult(chunk=chunk, score=0.9, retrieval_method="vector")
        ctx = RetrievalContext(
            results=[result],
            query="test query",
            query_type=IntentType.FACTUAL,
            total_candidates=100,
            retrieval_time_ms=45.0,
        )
        assert len(ctx.results) == 1
        assert ctx.retrieval_time_ms == 45.0


class TestGraphTypes:
    """Tests for knowledge graph types."""

    def test_graph_entity(self) -> None:
        entity = GraphEntity(
            entity_id="e1",
            name="Mom",
            entity_type=EntityType.PERSON,
            aliases=["my mother", "mama"],
        )
        assert entity.entity_type == EntityType.PERSON
        assert entity.entity_type == "person"
        assert len(entity.aliases) == 2
        assert entity.mention_count == 1
        assert entity.confidence == 1.0

    def test_graph_entity_all_types(self) -> None:
        """All 16 entity types are accessible."""
        assert len(EntityType) == 16
        for et in EntityType:
            entity = GraphEntity(entity_id=f"e_{et}", name=f"test_{et}", entity_type=et)
            assert entity.entity_type == et

    def test_graph_relation(self) -> None:
        rel = GraphRelation(
            source_id="e1",
            target_id="e2",
            relation_type=RelationType.DISCUSSED,
            weight=2.0,
        )
        assert rel.weight == 2.0
        assert rel.relation_type == RelationType.DISCUSSED
        assert rel.confidence == 1.0

    def test_graph_relation_all_types(self) -> None:
        """All 18 relation types are accessible."""
        assert len(RelationType) == 18
        for rt in RelationType:
            rel = GraphRelation(source_id="a", target_id="b", relation_type=rt)
            assert rel.relation_type == rt


class TestToolTypes:
    """Tests for tool call/result types."""

    def test_tool_call(self) -> None:
        call = ToolCall(tool_name="desktop.click", parameters={"x": 100, "y": 200})
        assert call.tool_name == "desktop.click"
        assert len(call.call_id) == 12

    def test_tool_result(self) -> None:
        result = ToolResult(
            call_id="abc",
            tool_name="desktop.click",
            success=True,
            output="clicked",
            execution_time_ms=15.0,
        )
        assert result.success is True


class TestWebSearchResult:
    """Tests for web search result type."""

    def test_create_result(self) -> None:
        result = WebSearchResult(
            title="Example",
            url="https://example.com",
            snippet="An example page",
            source="duckduckgo",
        )
        assert result.source == "duckduckgo"


class TestAudioTypes:
    """Tests for audio-related types."""

    def test_audio_segment(self) -> None:
        seg = AudioSegment(samples=b"\x00" * 100, sample_rate=16000, channels=1, duration_s=0.5)
        assert seg.duration_s == 0.5

    def test_transcription_result(self) -> None:
        result = TranscriptionResult(
            text="Hello Ara",
            confidence=0.95,
            duration_s=1.2,
            latency_ms=48.0,
        )
        assert result.text == "Hello Ara"
        assert result.latency_ms == 48.0


class TestVRAMSnapshot:
    """Tests for VRAM monitoring type."""

    def test_vram_snapshot(self) -> None:
        snap = VRAMSnapshot(
            total_gb=24.0,
            used_gb=19.0,
            free_gb=5.0,
            models_loaded=["Qwen3-30B-A3B", "Moondream-2B"],
        )
        assert snap.free_gb == 5.0
        assert len(snap.models_loaded) == 2
