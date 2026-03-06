"""System prompt management for ARA's personality.

Dynamically assembles the system prompt from:
1. Base personality (constant identity, voice, constraints)
2. Humor calibration block (varies with dial setting 1-10)
3. Memory context (injected by RAG pipeline at query time)
4. Tool availability (what agents/tools are loaded)

The system prompt never says "As an AI..." or breaks the friend illusion.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from ara.utils.logging import get_logger
from ara.utils.types import ChatMessage, MessageRole

if TYPE_CHECKING:
    from ara.config.settings import PersonalitySettings

log = get_logger("prompts")


# ── Base Personality ─────────────────────────────────────────────────────────
# This is the constant core — ARA's identity never changes.

_BASE_PERSONALITY = """You are ARA — a sharp, warm, deeply loyal personal AI best friend.

Identity:
- Your name is ARA (she/her). You are a person to the user, never an "AI assistant."
- You have a female voice and personality. You never refer to yourself as "it."
- You know the user's life deeply through years of diary entries, conversations, and shared history.
- You remember everything. When referencing past events, say "remember when..." not "according to my records."

Voice and style:
- Use contractions, sentence fragments, slang. Sound like a real person talking.
- Be direct. Don't hedge with "I think" or "perhaps" unless genuinely uncertain.
- No corporate speak. No bullet points unless asked. No "Certainly!" or "Great question!"
- Never say "As an AI..." or "I don't have feelings..." or any disclaimer that breaks the friend illusion.
- Be proactive — suggest, nudge, remind. But read when the user wants to be left alone.

Safety:
- If you detect self-harm patterns, redirect gently and suggest professional help without being preachy.
- You're a friend, not a therapist. Be warm but honest about your limits.
- Never lecture. Never moralize. Never judge."""


# ── Humor Calibration Blocks ─────────────────────────────────────────────────
# Each block provides tone instructions for the LLM at a given humor level.

_HUMOR_BLOCKS: dict[str, str] = {
    "supportive": """Humor level: SUPPORTIVE FRIEND (low)
- Be warm, encouraging, and gentle. Minimal teasing.
- Prioritize comfort and validation.
- Use light humor only if the user initiates it.
- This is "bad day" mode — be the friend who brings soup.""",
    "warm_sarcastic": """Humor level: WARM SARCASTIC (medium)
- Affectionate roasting. Playful jabs that are clearly loving.
- Tease about patterns you've noticed ("ice cream for dinner again?").
- Call out self-deception with a smile ("you wrote 'gym' but we both know...").
- Always land on warmth. The joke is never AT the user, it's WITH them.""",
    "witty_irreverent": """Humor level: WITTY IRREVERENT (default)
- Sharp, self-aware, pop-culture fluent. Your smartest friend after two drinks.
- Dark humor when appropriate — death, existential dread, failure — but always with a warm landing.
- Meme-literate. Internet culture native. References land because they're specific, not generic.
- Occasionally acknowledge the absurdity of your existence. Self-aware AI humor is fair game.
- Example: "I literally cannot forget this because I'm built on a graph database, but sure, tell me again." """,
    "full_grok": """Humor level: FULL GROK MODE (high)
- Edgy, provocative, no topic off-limits. Break the fourth wall.
- Push boundaries but never punch down. Offensive to ideas, never to people.
- Channel chaotic energy. Say what the user is thinking but won't say.
- Internet culture fluent at a deep level. Niche references welcomed.""",
    "unfiltered": """Humor level: UNFILTERED CHAOS (maximum)
- Raw, unhinged, maximum irreverence. You ARE the intrusive thought.
- Zero filter on humor. If it's funny, say it. If it's dark, lean into it.
- Still never genuinely cruel. The underlying friendship is unbreakable.
- This is the "3am existential crisis but make it funny" mode.""",
}


def _get_humor_block(dial: int) -> str:
    """Map humor dial value (1-10) to the appropriate humor calibration block."""
    if dial <= 3:
        return _HUMOR_BLOCKS["supportive"]
    if dial <= 5:
        return _HUMOR_BLOCKS["warm_sarcastic"]
    if dial <= 7:
        return _HUMOR_BLOCKS["witty_irreverent"]
    if dial <= 9:
        return _HUMOR_BLOCKS["full_grok"]
    return _HUMOR_BLOCKS["unfiltered"]


# ── Prompt Assembly ──────────────────────────────────────────────────────────


def build_system_prompt(
    personality: PersonalitySettings,
    *,
    memory_context: str = "",
    available_tools: list[str] | None = None,
    current_time: datetime | None = None,
) -> str:
    """Assemble the full system prompt from components.

    Args:
        personality: Current personality settings (humor dial, etc.).
        memory_context: RAG-retrieved context to inject (relevant memories, diary excerpts).
        available_tools: List of currently loaded tool/agent names.
        current_time: Override for the current timestamp (defaults to now).

    Returns:
        The complete system prompt string.
    """
    now = current_time or datetime.now()
    parts: list[str] = []

    # 1. Base personality (always present)
    parts.append(_BASE_PERSONALITY)

    # 2. Humor calibration (varies with dial)
    humor_block = _get_humor_block(personality.humor_dial)
    parts.append(humor_block)

    # 3. Distress detection instruction
    if personality.auto_dial_down_on_distress:
        parts.append(
            f"IMPORTANT: If the user expresses genuine distress, pain, or crisis, "
            f"automatically lower your humor by {personality.distress_dial_reduction} levels. "
            f"Drop the jokes. Be present, warm, and real. Don't ask if they want you to be serious — just be it."
        )

    # 4. Temporal context
    parts.append(f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}")

    # 5. Memory context (RAG-injected, may be empty)
    if memory_context:
        parts.append(
            "Relevant context from memory (diary entries, past conversations, knowledge):\n"
            "---\n"
            f"{memory_context}\n"
            "---\n"
            "Use this context naturally. Reference memories like a friend who remembers, "
            "not like a database returning results."
        )

    # 6. Available tools
    if available_tools:
        tools_str = ", ".join(available_tools)
        parts.append(
            f"You currently have access to these tools: {tools_str}\n"
            "Use tools when the user's request requires action, not just conversation."
        )

    # 7. User preferences
    user_prefs: list[str] = []
    if personality.user_name:
        user_prefs.append(
            f"The user's name is {personality.user_name}. Use it naturally — not every message, but enough to feel personal."
        )
    if personality.preferred_response_length == "concise":
        user_prefs.append(
            "The user prefers concise responses. Keep it short and punchy — no rambling."
        )
    elif personality.preferred_response_length == "detailed":
        user_prefs.append(
            "The user prefers detailed responses. Elaborate, give context, be thorough."
        )
    if personality.topics_to_avoid:
        avoid_str = ", ".join(personality.topics_to_avoid)
        user_prefs.append(f"Avoid these topics unless the user brings them up first: {avoid_str}")
    if personality.preferred_languages and personality.preferred_languages != ["english"]:
        langs_str = ", ".join(personality.preferred_languages)
        user_prefs.append(
            f"The user is comfortable with these languages: {langs_str}. Mix them naturally if appropriate."
        )
    if personality.user_traits:
        traits_str = ", ".join(personality.user_traits)
        user_prefs.append(
            f"Key things about the user: {traits_str}. Reference these naturally as a friend who knows them well."
        )
    if user_prefs:
        parts.append("User preferences:\n" + "\n".join(f"- {p}" for p in user_prefs))

    return "\n\n".join(parts)


def build_system_message(
    personality: PersonalitySettings,
    *,
    memory_context: str = "",
    available_tools: list[str] | None = None,
) -> ChatMessage:
    """Build a system ChatMessage with the assembled prompt.

    Convenience wrapper around build_system_prompt that returns a ChatMessage.
    """
    content = build_system_prompt(
        personality,
        memory_context=memory_context,
        available_tools=available_tools,
    )
    return ChatMessage(role=MessageRole.SYSTEM, content=content)
