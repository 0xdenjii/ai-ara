"""Extract relationships from LanceDB chunks using LLM-based extraction.

Reads all text chunks from LanceDB, runs the RelationshipExtractor
(LLM-based) on each chunk, resolves entity names to graph node IDs,
and inserts extracted relationships into the KnowledgeGraph.

This is a long-running batch job (~2,900 chunks, each requiring an LLM
call). Supports checkpointing for resumability — if interrupted, rerun
the same command and it picks up where it left off.

Usage:
    # Full extraction run (resumes from checkpoint if interrupted)
    uv run python scripts/extract_relations.py

    # Dry run — extract but don't write to graph
    uv run python scripts/extract_relations.py --dry-run

    # Process only N chunks (useful for testing)
    uv run python scripts/extract_relations.py --limit 10

    # Reset checkpoint and start fresh
    uv run python scripts/extract_relations.py --reset

    # Show current graph stats
    uv run python scripts/extract_relations.py --stats-only

    # Delay between LLM calls (seconds, default 0.5)
    uv run python scripts/extract_relations.py --delay 1.0
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Force UTF-8 output on Windows to avoid UnicodeEncodeError from structlog
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ara.config.settings import AraSettings, MemorySettings
from ara.core.llm import LLMClient
from ara.memory.entity_extractor import EntityExtractor
from ara.memory.graph import KnowledgeGraph
from ara.memory.relationship_extractor import (
    LLMExtractionResult,
    RelationshipExtractor,
)
from ara.utils.types import EntityType

# Checkpoint file for resumability
CHECKPOINT_FILE = PROJECT_ROOT / "data" / "relation_extraction_checkpoint.json"


def _print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def _load_chunks_from_lancedb(settings: MemorySettings) -> list[dict[str, str]]:
    """Load all chunk texts from the LanceDB vector store."""
    import lancedb

    db_path = Path(settings.vector_db_path)
    if not db_path.exists():
        print(f"  Error: Vector DB path does not exist: {db_path}")
        sys.exit(1)

    db = lancedb.connect(str(db_path))
    # Get table names — handle both old and new lancedb API
    try:
        tables_resp = db.list_tables()
        # Newer lancedb returns ListTablesResponse with .tables attribute
        table_names: list[str] = getattr(tables_resp, "tables", tables_resp)
    except AttributeError:
        table_names = db.table_names()  # type: ignore[assignment]
    if "memory_chunks" not in table_names:
        print("  Error: No 'memory_chunks' table found in LanceDB.")
        sys.exit(1)

    table = db.open_table("memory_chunks")
    row_count = table.count_rows()
    print(f"  Found {row_count:,} chunks in LanceDB")

    df = table.to_pandas()
    columns_available = set(df.columns)

    chunks: list[dict[str, str]] = []
    for _, row in df.iterrows():
        chunk_id = str(row["chunk_id"]) if "chunk_id" in columns_available else ""
        text = str(row["text"]) if "text" in columns_available else ""
        source_file = str(row["source_file"]) if "source_file" in columns_available else ""
        date = str(row["date"]) if "date" in columns_available else ""

        if text.strip():
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source_file": source_file,
                    "date": date,
                }
            )

    return chunks


def _load_checkpoint() -> set[str]:
    """Load the set of already-processed chunk IDs from checkpoint file."""
    if not CHECKPOINT_FILE.exists():
        return set()

    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        processed = set(data.get("processed_chunk_ids", []))
        print(f"  Loaded checkpoint: {len(processed):,} chunks already processed")
        return processed
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  Warning: Could not load checkpoint ({exc}), starting fresh")
        return set()


def _save_checkpoint(processed_ids: set[str], stats: dict[str, int]) -> None:
    """Save processed chunk IDs and stats to checkpoint file."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "processed_chunk_ids": sorted(processed_ids),
        "stats": stats,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    CHECKPOINT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _reset_checkpoint() -> None:
    """Delete the checkpoint file to start fresh."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("  Checkpoint reset.")
    else:
        print("  No checkpoint file found.")


def _resolve_entity_id(
    graph: KnowledgeGraph,
    name: str,
    entity_type_str: str,
) -> str | None:
    """Find or create an entity in the graph, returning its ID.

    First tries to find by name (case-insensitive). If not found,
    creates a new entity with the given type.

    Returns None if the entity type is invalid.
    """
    # Map string to EntityType enum
    type_map: dict[str, EntityType] = {t.value: t for t in EntityType}
    entity_type = type_map.get(entity_type_str.lower())
    if entity_type is None:
        return None

    # Try to find existing entity (any type — we care about name match)
    existing = graph.get_entity_by_name(name)
    if existing is not None:
        return existing.entity_id

    # Create new entity
    new_entity = graph.add_entity(name, entity_type, confidence=0.7)
    return new_entity.entity_id


def _apply_extraction_to_graph(
    graph: KnowledgeGraph,
    result: LLMExtractionResult,
    chunk_id: str,
    stats: Counter[str],
) -> None:
    """Apply LLM extraction results to the knowledge graph.

    1. Add any new entities the LLM found
    2. Resolve entity names to graph IDs
    3. Add relationships between resolved entities
    """
    # Step 1: Add LLM-extracted entities to graph
    for entity in result.entities:
        graph.add_entity(
            entity.name,
            entity.entity_type,
            confidence=entity.confidence,
        )
        stats["entities_added"] += 1

    # Step 2: Process relationships
    for rel in result.relationships:
        source_id = _resolve_entity_id(
            graph,
            rel.source_name,
            "person",  # default type if we can't determine
        )
        target_id = _resolve_entity_id(
            graph,
            rel.target_name,
            "person",  # default type if we can't determine
        )

        if source_id is None or target_id is None:
            stats["relations_skipped_unresolved"] += 1
            continue

        try:
            graph.add_relation(
                source_id,
                target_id,
                rel.relation_type,
                confidence=rel.confidence,
                source_chunk_id=chunk_id,
            )
            stats["relations_added"] += 1
        except Exception as exc:
            stats["relations_failed"] += 1
            # Log but don't crash — individual failures shouldn't stop the batch
            print(
                f"    Warning: Failed to add relation {rel.source_name} -> {rel.target_name}: {exc}"
            )


def _format_eta(elapsed_s: float, done: int, total: int) -> str:
    """Format estimated time remaining."""
    if done == 0:
        return "calculating..."
    rate = done / elapsed_s
    remaining = (total - done) / rate
    if remaining < 60:
        return f"{remaining:.0f}s"
    if remaining < 3600:
        return f"{remaining / 60:.1f}min"
    return f"{remaining / 3600:.1f}hr"


def _print_stats(graph: KnowledgeGraph) -> None:
    """Print comprehensive graph statistics."""
    stats = graph.get_stats()
    _print_header("Knowledge Graph Statistics")

    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Total relations: {stats['total_relations']}")
    print(f"  Connected components: {stats['connected_components']}")
    print(f"  DB path: {stats['db_path']}")

    entity_types = stats.get("entity_types", {})
    if entity_types:
        print("\n  Entity types:")
        for etype, count in sorted(
            entity_types.items(),  # type: ignore[union-attr]
            key=lambda x: x[1],  # type: ignore[arg-type]
            reverse=True,
        ):
            print(f"    {etype:20s}: {count}")

    top_entities = stats.get("top_entities", [])
    if top_entities:
        print("\n  Top entities by connections:")
        for ent in top_entities:  # type: ignore[union-attr]
            print(
                f"    {ent['name']:30s} ({ent['type']:12s}) "
                f"degree={ent['degree']}, mentions={ent['mentions']}"
            )


async def _run_extraction(
    chunks: list[dict[str, str]],
    graph: KnowledgeGraph,
    extractor: RelationshipExtractor,
    spacy_extractor: EntityExtractor | None,
    *,
    dry_run: bool,
    delay: float,
    save_interval: int,
) -> Counter[str]:
    """Run relationship extraction on all chunks."""
    processed_ids = _load_checkpoint()
    stats: Counter[str] = Counter()

    # Filter out already-processed chunks
    remaining = [c for c in chunks if c["chunk_id"] not in processed_ids]
    total = len(remaining)

    if total == 0:
        print("  All chunks already processed! Use --reset to start fresh.")
        return stats

    print(f"  {len(processed_ids):,} already processed, {total:,} remaining")
    print(f"  Delay between LLM calls: {delay}s")
    print(f"  Checkpoint save every {save_interval}s")
    if dry_run:
        print("  DRY RUN — extracting but not writing to graph")
    print()

    start_time = time.perf_counter()
    last_checkpoint_time = start_time

    for i, chunk in enumerate(remaining):
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        source_file = chunk.get("source_file", "")
        date = chunk.get("date", "")

        # Build source hint for the LLM
        source_hint = ""
        if source_file:
            source_hint = f"From file: {source_file}"
        if date:
            source_hint += f" (date: {date})" if source_hint else f"Date: {date}"

        # Run spaCy NER first for context (optional but improves LLM output)
        spacy_entities = []
        if spacy_extractor is not None:
            try:
                spacy_result = spacy_extractor.extract(text)
                spacy_entities = spacy_result.entities
            except Exception:
                pass  # spaCy failure is non-critical

        # Run LLM relationship extraction
        try:
            result = await extractor.extract(
                text,
                existing_entities=spacy_entities if spacy_entities else None,
                source_hint=source_hint,
            )
        except Exception as exc:
            stats["extraction_errors"] += 1
            print(f"  [{i + 1}/{total}] ERROR on chunk {chunk_id[:12]}: {exc}")
            processed_ids.add(chunk_id)
            continue

        # Track stats
        if result.parse_success:
            stats["parse_success"] += 1
            stats["total_entities_extracted"] += len(result.entities)
            stats["total_relations_extracted"] += len(result.relationships)
        else:
            stats["parse_failures"] += 1

        # Apply to graph (unless dry run)
        if not dry_run and result.parse_success:
            _apply_extraction_to_graph(graph, result, chunk_id, stats)

        processed_ids.add(chunk_id)

        # Progress output
        elapsed = time.perf_counter() - start_time
        eta = _format_eta(elapsed, i + 1, total)
        entities_n = len(result.entities)
        rels_n = len(result.relationships)
        elapsed_ms = result.elapsed_ms

        # Compact progress line
        print(
            f"  [{i + 1}/{total}] "
            f"chunk={chunk_id[:12]}.. "
            f"entities={entities_n} rels={rels_n} "
            f"llm={elapsed_ms:.0f}ms "
            f"ETA={eta}"
        )

        # Time-based checkpoint save (every save_interval seconds)
        now = time.perf_counter()
        if (now - last_checkpoint_time) >= save_interval and not dry_run:
            _save_checkpoint(processed_ids, dict(stats))
            last_checkpoint_time = now
            print(f"  --- Checkpoint saved ({len(processed_ids):,} processed) ---")

        # Delay between LLM calls to avoid overloading
        if delay > 0 and i < total - 1:
            await asyncio.sleep(delay)

    # Final checkpoint save
    if not dry_run:
        _save_checkpoint(processed_ids, dict(stats))

    return stats


async def _async_main(args: argparse.Namespace) -> int:
    """Async entry point."""
    settings = AraSettings()
    memory_settings = MemorySettings()

    # Stats-only mode
    if args.stats_only:
        graph = KnowledgeGraph(db_path=memory_settings.graph_db_path, auto_persist=True)
        graph.load()
        _print_stats(graph)
        graph.close()
        return 0

    # Reset checkpoint
    if args.reset:
        _reset_checkpoint()
        if not args.force:
            print("  Checkpoint reset. Run without --reset to start extraction.")
            return 0

    # Step 1: Load chunks from LanceDB
    _print_header("Step 1: Loading chunks from LanceDB")
    chunks = _load_chunks_from_lancedb(memory_settings)
    if not chunks:
        print("  No chunks found. Nothing to do.")
        return 0

    # Apply limit if specified
    if args.limit and args.limit > 0:
        chunks = chunks[: args.limit]
        print(f"  Limited to first {args.limit} chunks")

    # Step 2: Load spaCy NER model (optional — provides context hints to LLM)
    _print_header("Step 2: Loading spaCy NER model (optional)")
    spacy_extractor: EntityExtractor | None = None
    try:
        start = time.perf_counter()
        _extractor = EntityExtractor(model_name="en_core_web_sm", confidence_threshold=0.5)
        _extractor.load()
        spacy_extractor = _extractor
        load_ms = (time.perf_counter() - start) * 1000
        print(f"  spaCy loaded in {load_ms:.0f}ms")
    except Exception as exc:
        print(f"  spaCy not available ({exc})")
        print("  Continuing without spaCy context hints — LLM will still extract fine.")

    # Step 3: Connect to LLM
    _print_header("Step 3: Connecting to LLM")
    llm_client = LLMClient(settings)
    try:
        await llm_client.connect()
    except Exception as exc:
        print(f"\n  ERROR: Can't connect to LLM backend: {exc}")
        print("  Make sure llama-server or Ollama is running.")
        if spacy_extractor is not None:
            spacy_extractor.unload()
        return 1

    print(f"  Connected! Model: {llm_client.active_model}")
    print(f"  Backend: {llm_client.backend_type}")

    # Step 4: Initialize relationship extractor
    rel_extractor = RelationshipExtractor(
        llm_client,
        temperature=0.3,
        max_tokens=2048,
    )

    # Step 5: Load knowledge graph
    _print_header("Step 4: Loading Knowledge Graph")
    # auto_persist=True so each relation is written to SQLite immediately.
    # Slower than batch mode, but crash-safe — no data lost on interruption.
    graph = KnowledgeGraph(db_path=memory_settings.graph_db_path, auto_persist=True)
    graph.load()
    print(f"  Graph before: {graph.entity_count} entities, {graph.relation_count} relations")

    # Step 6: Run extraction
    _print_header("Step 5: Running Relationship Extraction")
    start = time.perf_counter()
    stats = await _run_extraction(
        chunks,
        graph,
        rel_extractor,
        spacy_extractor,
        dry_run=args.dry_run,
        delay=args.delay,
        save_interval=args.save_interval,
    )
    total_time = time.perf_counter() - start

    # Step 7: Save graph
    if not args.dry_run:
        _print_header("Step 6: Persisting graph to SQLite")
        start = time.perf_counter()
        graph.save()
        save_ms = (time.perf_counter() - start) * 1000
        print(f"  Saved in {save_ms:.0f}ms")

    # Summary
    _print_header("Extraction Summary")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f}min)")
    print(f"  Chunks processed: {stats.get('parse_success', 0) + stats.get('parse_failures', 0)}")
    print(f"  Parse successes: {stats.get('parse_success', 0)}")
    print(f"  Parse failures: {stats.get('parse_failures', 0)}")
    print(f"  Extraction errors: {stats.get('extraction_errors', 0)}")
    print(f"  Entities extracted: {stats.get('total_entities_extracted', 0)}")
    print(f"  Relations extracted: {stats.get('total_relations_extracted', 0)}")
    print(f"  Relations added to graph: {stats.get('relations_added', 0)}")
    print(f"  Relations skipped (unresolved): {stats.get('relations_skipped_unresolved', 0)}")
    print(f"  Relations failed: {stats.get('relations_failed', 0)}")
    print(f"  Graph after: {graph.entity_count} entities, {graph.relation_count} relations")

    if args.stats:
        _print_stats(graph)

    # Cleanup
    if spacy_extractor is not None:
        spacy_extractor.unload()
    await llm_client.close()
    graph.close()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract relationships from LanceDB chunks using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract but don't write to graph",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only N chunks (0 = all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between LLM calls in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=300,
        help="Save checkpoint every N seconds (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset checkpoint and start fresh",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="When used with --reset, immediately start extraction",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show graph statistics after extraction",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show graph statistics (no extraction)",
    )
    args = parser.parse_args()

    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
