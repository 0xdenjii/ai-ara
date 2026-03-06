"""ARA Text Chat — interactive CLI to talk with ARA.

Run: uv run python scripts/chat.py
Options:
    --no-memory    Skip memory retrieval (faster, no diary context)
    --humor N      Set humor dial 1-10 (default: 7 = witty irreverent)

Type your message and press Enter. ARA responds with streaming tokens.
Type 'quit', 'exit', or Ctrl+C to leave.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import re
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows to handle model-generated emojis/unicode
if sys.platform == "win32" and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
elif sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Suppress structlog output during interactive chat.
# Must happen BEFORE importing ara modules (they call get_logger() at import).
# Without setup_logging(), structlog defaults to PrintLogger (stdout) which
# clutters the chat. We configure it to use stdlib logging at CRITICAL level.
logging.disable(logging.CRITICAL)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ara.config.settings import AraSettings
from ara.core.llm import LLMClient
from ara.core.prompts import build_system_prompt
from ara.utils.logging import setup_logging
from ara.utils.types import ChatMessage, MessageRole

# Configure structlog to use stdlib logging (routed to stderr) at CRITICAL level
# This prevents structlog's default PrintLogger from writing to stdout
setup_logging(log_level="CRITICAL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with ARA")
    parser.add_argument("--no-memory", action="store_true", help="Skip memory retrieval")
    parser.add_argument("--humor", type=int, default=7, help="Humor dial 1-10 (default: 7)")
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Max response tokens (includes think block)"
    )
    return parser.parse_args()


async def setup_memory(settings: AraSettings):
    """Load memory retrieval pipeline if available."""
    try:
        from ara.memory.embeddings import EmbeddingModel
        from ara.memory.retrieval import RetrievalPipeline
        from ara.memory.vector_store import VectorStore

        print("  Loading embedding model (nomic-embed-text)...")
        embedder = EmbeddingModel(settings.memory)
        embedder.load()

        print("  Connecting to vector store...")
        vector_store = VectorStore(settings.memory)
        vector_store.connect()

        pipeline = RetrievalPipeline(
            settings.memory,
            vector_store,
            embedder,
        )
        print(f"  Memory ready ({vector_store.count()} chunks indexed)")
        return pipeline
    except Exception as exc:
        print(f"  Memory unavailable: {exc}")
        return None


async def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  ARA — Your AI Best Friend")
    print("=" * 60)
    print()

    # Setup
    settings = AraSettings()
    settings.personality.humor_dial = args.humor

    print(f"  Humor level: {args.humor}/10")
    print(f"  Model: {settings.llm.primary_model}")
    print()

    # Connect to LLM
    print("  Connecting to Ollama...")
    client = LLMClient(settings)
    try:
        await client.connect()
    except Exception as exc:
        print(f"\n  ERROR: Can't connect to Ollama: {exc}")
        print("  Make sure Ollama is running: ollama serve")
        return

    print(f"  Connected! Model: {client.active_model}")

    # Setup memory retrieval (optional)
    retrieval = None
    if not args.no_memory:
        print("\n  Loading memory system...")
        retrieval = await setup_memory(settings)
    else:
        print("  Memory: disabled (--no-memory)")

    # Build initial system prompt
    print()
    print("-" * 60)
    print("  Ready! Type your message. 'quit' to exit.")
    print("-" * 60)
    print()

    # Conversation state
    messages: list[ChatMessage] = []
    turn_count = 0

    while True:
        try:
            user_input = input("\033[96mYou:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  See ya!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            print("\n  \033[95mARA:\033[0m Later! Don't do anything I wouldn't do.\n")
            break

        turn_count += 1

        # Retrieve relevant memory context
        memory_context = ""
        if retrieval is not None:
            try:
                context = retrieval.retrieve(user_input)
                if context.results:
                    memory_context = retrieval.assemble_context(context.results)
            except Exception:
                pass  # Silently skip retrieval errors

        # Build system prompt with personality + memory
        system_prompt = build_system_prompt(
            settings.personality,
            memory_context=memory_context,
        )

        # Build message list
        conv_messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ]

        # Add conversation history (last 10 turns = 20 messages)
        if messages:
            conv_messages.extend(messages[-20:])

        # Add current user message
        conv_messages.append(ChatMessage(role=MessageRole.USER, content=user_input))

        # Stream response with <think> block filtering
        # Qwen3 emits <think>...</think> reasoning blocks — hide them from user
        print("\033[95mARA:\033[0m ", end="", flush=True)

        response_parts: list[str] = []
        in_think_block = False
        try:
            async for token in client.generate_stream(
                conv_messages,
                max_tokens=args.max_tokens,
            ):
                response_parts.append(token)
                accumulated = "".join(response_parts)

                # Track think block state
                if "<think>" in accumulated and "</think>" not in accumulated:
                    in_think_block = True
                    continue
                if in_think_block and "</think>" in accumulated:
                    # Think block just ended — strip it and print remaining
                    in_think_block = False
                    cleaned = re.sub(r"<think>.*?</think>", "", accumulated, flags=re.DOTALL)
                    print(cleaned, end="", flush=True)
                    response_parts = [cleaned]
                    continue
                if in_think_block:
                    continue

                # Normal token — print directly
                print(token, end="", flush=True)
        except Exception as exc:
            print(f"\n  [Error: {exc}]")
            continue

        full_response = "".join(response_parts)
        # Final cleanup of any remaining think tags
        full_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()

        print("\n")

        # Update conversation history
        messages.append(ChatMessage(role=MessageRole.USER, content=user_input))
        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=full_response))

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
