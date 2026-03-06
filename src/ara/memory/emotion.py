"""Emotion tagging for diary entries and text chunks.

Detects emotional content in text using keyword-based pattern matching.
No ML models or external dependencies — runs on CPU in <1ms per entry.

Emotions are categorized using Plutchik's basic emotion wheel plus
common diary-specific states. Each detected emotion gets a confidence
score based on keyword density and intensity markers.

All patterns are designed for English first-person diary writing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ara.utils.logging import get_logger

log = get_logger("emotion")


# ── Emotion Lexicons ────────────────────────────────────────────────────────

# Each emotion maps to (keywords, intensity_multiplier)
# Higher multiplier = stronger signal when that word appears
_EMOTION_LEXICON: dict[str, list[tuple[str, float]]] = {
    "joy": [
        ("happy", 1.0),
        ("glad", 0.8),
        ("excited", 1.2),
        ("thrilled", 1.3),
        ("delighted", 1.1),
        ("ecstatic", 1.5),
        ("overjoyed", 1.4),
        ("wonderful", 1.0),
        ("amazing", 1.1),
        ("great day", 1.0),
        ("proud", 1.0),
        ("grateful", 0.9),
        ("blessed", 0.8),
        ("love it", 1.0),
        ("loved it", 1.0),
        ("so good", 0.9),
        ("best day", 1.3),
        ("fun", 0.7),
        ("enjoying", 0.8),
        ("laughed", 0.8),
        ("celebrate", 1.0),
        ("accomplished", 1.0),
    ],
    "sadness": [
        ("sad", 1.0),
        ("unhappy", 1.0),
        ("depressed", 1.5),
        ("down", 0.7),
        ("lonely", 1.2),
        ("alone", 0.8),
        ("crying", 1.3),
        ("cried", 1.3),
        ("tears", 1.1),
        ("heartbroken", 1.5),
        ("devastated", 1.5),
        ("miserable", 1.4),
        ("hopeless", 1.4),
        ("empty", 1.0),
        ("numb", 1.1),
        ("miss", 0.6),
        ("missing", 0.7),
        ("lost", 0.7),
        ("grief", 1.4),
        ("mourning", 1.3),
        ("worst day", 1.3),
    ],
    "anger": [
        ("angry", 1.0),
        ("furious", 1.4),
        ("frustrated", 1.0),
        ("annoyed", 0.8),
        ("irritated", 0.8),
        ("mad", 0.9),
        ("pissed", 1.2),
        ("rage", 1.4),
        ("hate", 1.2),
        ("unfair", 0.8),
        ("disgusted", 1.1),
        ("outraged", 1.3),
        ("resent", 1.1),
        ("bitter", 1.0),
        ("fed up", 1.0),
    ],
    "fear": [
        ("afraid", 1.0),
        ("scared", 1.0),
        ("anxious", 1.1),
        ("worried", 0.9),
        ("nervous", 0.8),
        ("terrified", 1.4),
        ("panic", 1.3),
        ("dread", 1.2),
        ("fearful", 1.0),
        ("uneasy", 0.7),
        ("overwhelmed", 1.0),
        ("stress", 0.8),
        ("stressed", 0.9),
        ("paranoid", 1.2),
        ("insecure", 0.9),
    ],
    "surprise": [
        ("surprised", 1.0),
        ("shocked", 1.2),
        ("unexpected", 0.8),
        ("amazed", 1.0),
        ("astonished", 1.2),
        ("stunned", 1.1),
        ("can't believe", 1.0),
        ("couldn't believe", 1.0),
        ("out of nowhere", 0.9),
        ("plot twist", 0.8),
    ],
    "trust": [
        ("trust", 0.8),
        ("confident", 0.9),
        ("secure", 0.7),
        ("reliable", 0.7),
        ("faith", 0.8),
        ("believe in", 0.8),
        ("count on", 0.7),
        ("support", 0.6),
        ("loyal", 0.8),
    ],
    "anticipation": [
        ("excited about", 1.0),
        ("looking forward", 1.0),
        ("can't wait", 1.1),
        ("planning", 0.6),
        ("hopeful", 0.9),
        ("eager", 0.9),
        ("optimistic", 0.9),
        ("expecting", 0.7),
    ],
    "love": [
        ("love", 0.9),
        ("adore", 1.1),
        ("cherish", 1.0),
        ("affection", 0.8),
        ("romantic", 0.8),
        ("intimate", 0.8),
        ("crush", 0.7),
        ("boyfriend", 0.5),
        ("girlfriend", 0.5),
        ("partner", 0.5),
        ("soulmate", 1.0),
        ("in love", 1.3),
    ],
    "gratitude": [
        ("grateful", 1.0),
        ("thankful", 1.0),
        ("appreciate", 0.9),
        ("blessed", 0.8),
        ("lucky", 0.7),
        ("fortune", 0.7),
        ("thank god", 0.9),
        ("thanks to", 0.7),
    ],
    "nostalgia": [
        ("remember when", 1.0),
        ("used to", 0.6),
        ("back then", 0.8),
        ("childhood", 0.7),
        ("memories", 0.8),
        ("nostalgic", 1.2),
        ("miss those days", 1.1),
        ("good old days", 1.0),
        ("reminds me of", 0.8),
        ("throwback", 0.7),
    ],
    "confusion": [
        ("confused", 1.0),
        ("don't know", 0.7),
        ("unsure", 0.8),
        ("torn", 0.9),
        ("conflicted", 1.0),
        ("mixed feelings", 1.0),
        ("complicated", 0.7),
        ("lost", 0.6),
        ("unclear", 0.7),
    ],
    "determination": [
        ("determined", 1.0),
        ("motivated", 1.0),
        ("committed", 0.9),
        ("decided", 0.8),
        ("resolve", 0.9),
        ("going to", 0.5),
        ("will do", 0.5),
        ("must", 0.6),
        ("need to change", 0.8),
        ("turning point", 1.0),
        ("fresh start", 0.9),
    ],
}

# Intensity amplifiers — if these appear near emotion words, boost confidence
_INTENSITY_AMPLIFIERS = re.compile(
    r"\b(?:very|extremely|incredibly|absolutely|completely|totally|"
    r"so|really|deeply|truly|utterly|profoundly)\b",
    re.IGNORECASE,
)

# Negation words — if these precede emotion words, they flip or reduce the emotion
_NEGATION_WORDS = re.compile(
    r"\b(?:not|no|never|don't|doesn't|didn't|wasn't|weren't|"
    r"isn't|aren't|hardly|barely|neither|nor)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EmotionTag:
    """A detected emotion with confidence score."""

    emotion: str  # e.g., "joy", "sadness", "anger"
    confidence: float  # 0.0 to 1.0
    keywords_found: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EmotionMetadata:
    """All emotions detected in a piece of text."""

    tags: list[EmotionTag] = field(default_factory=list)
    dominant_emotion: str = ""
    emotional_valence: float = 0.0  # -1.0 (very negative) to 1.0 (very positive)
    emotional_intensity: float = 0.0  # 0.0 (neutral) to 1.0 (very intense)
    has_emotional_content: bool = False


# Valence mapping for each emotion (-1 to +1)
_EMOTION_VALENCE: dict[str, float] = {
    "joy": 0.8,
    "sadness": -0.8,
    "anger": -0.6,
    "fear": -0.5,
    "surprise": 0.1,  # Neutral — can be positive or negative
    "trust": 0.5,
    "anticipation": 0.4,
    "love": 0.9,
    "gratitude": 0.7,
    "nostalgia": 0.2,  # Bittersweet — slightly positive
    "confusion": -0.3,
    "determination": 0.3,
}


class EmotionTagger:
    """Rule-based emotion tagger for diary text.

    Detects emotions using keyword matching with intensity-weighted
    scoring. No ML models needed — fast and deterministic.

    Usage:
        tagger = EmotionTagger()
        metadata = tagger.tag("I felt so happy today, but also a bit anxious.")
        # metadata.dominant_emotion == "joy"
        # metadata.tags == [EmotionTag("joy", 0.8, ["happy"]), EmotionTag("fear", 0.4, ["anxious"])]
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.2,
        max_emotions: int = 5,
    ) -> None:
        """Initialize the emotion tagger.

        Args:
            min_confidence: Minimum confidence to include an emotion tag.
            max_emotions: Maximum number of emotion tags to return.
        """
        self._min_confidence = min_confidence
        self._max_emotions = max_emotions

    def tag(self, text: str) -> EmotionMetadata:
        """Tag text with detected emotions.

        Args:
            text: The text to analyze.

        Returns:
            EmotionMetadata with detected emotions and scores.
        """
        if not text.strip():
            return EmotionMetadata()

        text_lower = text.lower()
        word_count = len(text.split())

        # Check for intensity amplifiers in the text
        has_amplifiers = bool(_INTENSITY_AMPLIFIERS.search(text))
        amplifier_boost = 1.3 if has_amplifiers else 1.0

        # Score each emotion
        emotion_scores: dict[str, tuple[float, list[str]]] = {}

        for emotion, keywords in _EMOTION_LEXICON.items():
            score = 0.0
            found_keywords: list[str] = []

            for keyword, weight in keywords:
                if keyword in text_lower:
                    # Check for negation within 3 words before the keyword
                    keyword_pos = text_lower.find(keyword)
                    context_start = max(0, keyword_pos - 30)
                    context = text_lower[context_start:keyword_pos]

                    if _NEGATION_WORDS.search(context):
                        # Negated — reduce score significantly
                        score -= weight * 0.5
                    else:
                        score += weight * amplifier_boost
                        found_keywords.append(keyword)

            if score > 0 and found_keywords:
                # Normalize by text length (longer texts get diminished scores)
                length_factor = min(1.0, 50.0 / max(word_count, 1))
                normalized_score = min(1.0, score * length_factor)

                if normalized_score >= self._min_confidence:
                    emotion_scores[emotion] = (normalized_score, found_keywords)

        if not emotion_scores:
            return EmotionMetadata()

        # Sort by score and take top N
        sorted_emotions = sorted(
            emotion_scores.items(),
            key=lambda x: x[1][0],
            reverse=True,
        )[: self._max_emotions]

        tags = [
            EmotionTag(
                emotion=emotion,
                confidence=round(score, 3),
                keywords_found=keywords,
            )
            for emotion, (score, keywords) in sorted_emotions
        ]

        # Compute dominant emotion
        dominant = tags[0].emotion if tags else ""

        # Compute emotional valence (weighted average)
        total_weight = sum(t.confidence for t in tags)
        valence = 0.0
        if total_weight > 0:
            valence = (
                sum(_EMOTION_VALENCE.get(t.emotion, 0.0) * t.confidence for t in tags)
                / total_weight
            )

        # Compute intensity (max confidence)
        intensity = max(t.confidence for t in tags) if tags else 0.0

        return EmotionMetadata(
            tags=tags,
            dominant_emotion=dominant,
            emotional_valence=round(valence, 3),
            emotional_intensity=round(intensity, 3),
            has_emotional_content=True,
        )

    def tag_as_dict(self, text: str) -> dict[str, object]:
        """Tag text and return results as a flat dictionary.

        Returns a dict suitable for merging into MemoryChunk.metadata.
        """
        meta = self.tag(text)

        result: dict[str, object] = {
            "has_emotional_content": meta.has_emotional_content,
        }

        if meta.has_emotional_content:
            result["dominant_emotion"] = meta.dominant_emotion
            result["emotional_valence"] = meta.emotional_valence
            result["emotional_intensity"] = meta.emotional_intensity
            result["emotion_tags"] = [
                {"emotion": t.emotion, "confidence": t.confidence} for t in meta.tags
            ]

        return result
