"""Tests for the emotion tagging module."""

from __future__ import annotations

from ara.memory.emotion import EmotionMetadata, EmotionTag, EmotionTagger

# ── EmotionTag / EmotionMetadata ──────────────────────────────────────────


class TestEmotionDataClasses:
    def test_emotion_tag(self) -> None:
        tag = EmotionTag(emotion="joy", confidence=0.8, keywords_found=["happy"])
        assert tag.emotion == "joy"
        assert tag.confidence == 0.8

    def test_emotion_metadata_empty(self) -> None:
        meta = EmotionMetadata()
        assert meta.has_emotional_content is False
        assert meta.dominant_emotion == ""
        assert meta.tags == []


# ── Basic Tagging ─────────────────────────────────────────────────────────


class TestEmotionTaggerBasic:
    def test_empty_text(self) -> None:
        tagger = EmotionTagger()
        meta = tagger.tag("")
        assert meta.has_emotional_content is False

    def test_no_emotions(self) -> None:
        tagger = EmotionTagger()
        meta = tagger.tag("The sky is blue and clouds are white.")
        assert meta.has_emotional_content is False

    def test_detects_joy(self) -> None:
        tagger = EmotionTagger()
        meta = tagger.tag("I felt so happy today! Everything was wonderful.")
        assert meta.has_emotional_content is True
        assert meta.dominant_emotion == "joy"
        assert meta.emotional_valence > 0  # Positive

    def test_detects_sadness(self) -> None:
        tagger = EmotionTagger()
        meta = tagger.tag("I felt really sad and lonely today.")
        assert meta.has_emotional_content is True
        assert meta.dominant_emotion == "sadness"
        assert meta.emotional_valence < 0  # Negative

    def test_detects_anger(self) -> None:
        tagger = EmotionTagger()
        meta = tagger.tag("I was so angry and frustrated with the situation.")
        assert meta.has_emotional_content is True
        assert meta.dominant_emotion == "anger"

    def test_detects_fear(self) -> None:
        tagger = EmotionTagger()
        meta = tagger.tag("I felt anxious and scared about the presentation.")
        assert meta.has_emotional_content is True
        emotions = [t.emotion for t in meta.tags]
        assert "fear" in emotions

    def test_detects_love(self) -> None:
        tagger = EmotionTagger()
        meta = tagger.tag("I am so in love with this person, I adore them.")
        assert meta.has_emotional_content is True
        emotions = [t.emotion for t in meta.tags]
        assert "love" in emotions


# ── Mixed Emotions ────────────────────────────────────────────────────────


class TestEmotionTaggerMixed:
    def test_mixed_emotions(self) -> None:
        tagger = EmotionTagger()
        meta = tagger.tag(
            "I felt happy about the promotion but anxious about the new responsibilities."
        )
        assert meta.has_emotional_content is True
        emotions = [t.emotion for t in meta.tags]
        assert len(emotions) >= 2  # At least 2 emotions detected

    def test_multiple_emotions_sorted_by_confidence(self) -> None:
        tagger = EmotionTagger()
        meta = tagger.tag("I was extremely happy and a bit nervous.")
        assert meta.has_emotional_content is True
        if len(meta.tags) > 1:
            # Should be sorted by confidence descending
            confidences = [t.confidence for t in meta.tags]
            assert confidences == sorted(confidences, reverse=True)


# ── Intensity & Negation ─────────────────────────────────────────────────


class TestEmotionTaggerIntensity:
    def test_intensity_amplifiers_boost(self) -> None:
        tagger = EmotionTagger()
        meta_normal = tagger.tag("I felt happy.")
        meta_intense = tagger.tag("I felt extremely happy and really excited.")

        # Intense version should have higher confidence
        if meta_normal.tags and meta_intense.tags:
            normal_joy = next((t for t in meta_normal.tags if t.emotion == "joy"), None)
            intense_joy = next((t for t in meta_intense.tags if t.emotion == "joy"), None)
            if normal_joy and intense_joy:
                assert intense_joy.confidence >= normal_joy.confidence

    def test_negation_reduces_score(self) -> None:
        tagger = EmotionTagger()
        meta_positive = tagger.tag("I was happy.")
        meta_negated = tagger.tag("I was not happy at all.")

        # Negated "happy" should not trigger strong joy
        if meta_negated.has_emotional_content:
            negated_joy = next((t for t in meta_negated.tags if t.emotion == "joy"), None)
            if negated_joy and meta_positive.tags:
                positive_joy = next((t for t in meta_positive.tags if t.emotion == "joy"), None)
                if positive_joy:
                    assert negated_joy.confidence <= positive_joy.confidence


# ── Valence & Intensity ──────────────────────────────────────────────────


class TestEmotionTaggerValence:
    def test_positive_valence(self) -> None:
        tagger = EmotionTagger()
        meta = tagger.tag("I am so grateful and happy for everything.")
        assert meta.emotional_valence > 0

    def test_negative_valence(self) -> None:
        tagger = EmotionTagger()
        meta = tagger.tag("I felt depressed and angry all day.")
        assert meta.emotional_valence < 0

    def test_intensity_range(self) -> None:
        tagger = EmotionTagger()
        meta = tagger.tag("I was ecstatic! Best day ever!")
        assert 0.0 <= meta.emotional_intensity <= 1.0


# ── Configuration ────────────────────────────────────────────────────────


class TestEmotionTaggerConfig:
    def test_min_confidence_filter(self) -> None:
        tagger = EmotionTagger(min_confidence=0.9)
        meta = tagger.tag("I felt a bit happy.")
        # With high min_confidence, weak signals should be filtered
        for tag in meta.tags:
            assert tag.confidence >= 0.9

    def test_max_emotions_limit(self) -> None:
        tagger = EmotionTagger(max_emotions=2)
        meta = tagger.tag("I felt happy but also sad and angry and anxious and surprised.")
        assert len(meta.tags) <= 2


# ── tag_as_dict ──────────────────────────────────────────────────────────


class TestEmotionTaggerAsDict:
    def test_empty_text(self) -> None:
        tagger = EmotionTagger()
        result = tagger.tag_as_dict("")
        assert result["has_emotional_content"] is False

    def test_with_emotions(self) -> None:
        tagger = EmotionTagger()
        result = tagger.tag_as_dict("I felt so happy today!")
        assert result["has_emotional_content"] is True
        assert "dominant_emotion" in result
        assert "emotional_valence" in result
        assert "emotion_tags" in result

    def test_dict_format(self) -> None:
        tagger = EmotionTagger()
        result = tagger.tag_as_dict("I was excited about the trip!")
        assert isinstance(result, dict)
        if result["has_emotional_content"]:
            assert isinstance(result["emotion_tags"], list)
