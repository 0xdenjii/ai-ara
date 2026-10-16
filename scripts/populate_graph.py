"""Populate the Knowledge Graph from existing LanceDB vector store chunks.

Reads all text chunks already stored in LanceDB, runs spaCy NER on them
in batches, and inserts extracted entities into the KnowledgeGraph SQLite
database. Designed as a one-time backfill for the ~2,600 chunks that were
ingested before entity extraction was wired into the pipeline.

Usage:
    # Full population run
    uv run python scripts/populate_graph.py

    # Dry run — show what would be extracted without writing
    uv run python scripts/populate_graph.py --dry-run

    # Control batch size (default 64)
    uv run python scripts/populate_graph.py --batch-size 32

    # Show graph stats after population
    uv run python scripts/populate_graph.py --stats
"""

from __future__ import annotations

import argparse
import io
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

from ara.config.settings import MemorySettings
from ara.memory.entity_extractor import EntityExtractor, ExtractionResult
from ara.memory.graph import KnowledgeGraph


def _print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def _load_chunks_from_lancedb(settings: MemorySettings) -> list[dict[str, object]]:
    """Load all chunk texts from the LanceDB vector store.

    Returns a list of dicts with 'chunk_id', 'text', 'source_file', 'date'.
    """
    import lancedb

    db_path = Path(settings.vector_db_path)
    if not db_path.exists():
        print(f"  Error: Vector DB path does not exist: {db_path}")
        sys.exit(1)

    db = lancedb.connect(str(db_path))
    table_names = db.table_names()
    if "memory_chunks" not in table_names:
        print("  Error: No 'memory_chunks' table found in LanceDB.")
        sys.exit(1)

    table = db.open_table("memory_chunks")
    row_count = table.count_rows()
    print(f"  Found {row_count:,} chunks in LanceDB")

    # Read all rows — only fetch columns we need (skip heavy vector column)
    df = table.to_pandas()
    columns_available = set(df.columns)

    chunks: list[dict[str, object]] = []
    for _, row in df.iterrows():
        chunk: dict[str, object] = {
            "chunk_id": row["chunk_id"] if "chunk_id" in columns_available else "",
            "text": row["text"] if "text" in columns_available else "",
            "source_file": row["source_file"] if "source_file" in columns_available else "",
            "date": row["date"] if "date" in columns_available else "",
        }
        chunks.append(chunk)

    return chunks


def _extract_entities_batch(
    extractor: EntityExtractor,
    texts: list[str],
    batch_size: int,
) -> list[ExtractionResult]:
    """Run entity extraction on texts in manageable batches."""
    all_results: list[ExtractionResult] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch_texts = texts[i : i + batch_size]
        results = extractor.extract_batch(batch_texts)
        all_results.extend(results)

        entity_count = sum(len(r.entities) for r in results)
        print(
            f"  Batch {batch_num}/{total_batches}: "
            f"{len(batch_texts)} chunks -> {entity_count} entities"
        )

    return all_results


def _populate_graph(
    graph: KnowledgeGraph,
    results: list[ExtractionResult],
) -> Counter[str]:
    """Add all extracted entities to the knowledge graph. Returns type counts."""
    type_counts: Counter[str] = Counter()

    for result in results:
        for entity in result.entities:
            graph.add_entity(
                entity.name,
                entity.entity_type,
                confidence=entity.confidence,
            )
            type_counts[entity.entity_type.value] += 1

    return type_counts


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Populate ARA Knowledge Graph from existing LanceDB chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract entities but don't write to graph",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks per NER batch (default: 64)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show graph statistics after population",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show graph statistics (no extraction)",
    )
    args = parser.parse_args()

    settings = MemorySettings()

    # Stats-only mode
    if args.stats_only:
        graph = KnowledgeGraph(db_path=settings.graph_db_path, auto_persist=True)
        graph.load()
        _print_stats(graph)
        graph.close()
        return 0

    # Step 1: Load chunks from LanceDB
    _print_header("Step 1: Loading chunks from LanceDB")
    chunks = _load_chunks_from_lancedb(settings)
    if not chunks:
        print("  No chunks found. Nothing to do.")
        return 0

    texts = [str(c["text"]) for c in chunks if c.get("text")]
    print(f"  {len(texts):,} non-empty texts to process")

    # Step 2: Load NER model
    _print_header("Step 2: Loading spaCy NER model")
    start = time.perf_counter()
    extractor = EntityExtractor(model_name="en_core_web_sm", confidence_threshold=0.5)
    extractor.load()
    load_ms = (time.perf_counter() - start) * 1000
    print(f"  NER model loaded in {load_ms:.0f}ms")

    # Step 3: Extract entities
    _print_header("Step 3: Extracting entities from chunks")
    start = time.perf_counter()
    results = _extract_entities_batch(extractor, texts, batch_size=args.batch_size)
    extract_ms = (time.perf_counter() - start) * 1000

    total_entities = sum(len(r.entities) for r in results)
    unique_names: set[tuple[str, str]] = set()
    for r in results:
        for e in r.entities:
            unique_names.add((e.name.lower(), e.entity_type.value))

    print(f"\n  Total entity mentions: {total_entities:,}")
    print(f"  Unique entities: {len(unique_names):,}")
    print(f"  Extraction time: {extract_ms:.0f}ms")

    if args.dry_run:
        _print_header("DRY RUN — Entity Summary")
        type_counts: Counter[str] = Counter()
        for r in results:
            for e in r.entities:
                type_counts[e.entity_type.value] += 1

        print("  Entity type distribution:")
        for etype, count in type_counts.most_common():
            print(f"    {etype:20s}: {count}")

        # Show top entities by frequency
        name_freq: Counter[str] = Counter()
        for r in results:
            for e in r.entities:
                name_freq[f"{e.name} ({e.entity_type.value})"] += 1

        print("\n  Top 30 entities by frequency:")
        for name, count in name_freq.most_common(30):
            print(f"    {name:40s}: {count}")

        extractor.unload()
        return 0

    # Step 4: Populate the graph
    _print_header("Step 4: Populating Knowledge Graph")
    start = time.perf_counter()

    # Use auto_persist=False for batch insert, then save() once at the end
    graph = KnowledgeGraph(db_path=settings.graph_db_path, auto_persist=False)
    graph.load()

    print(f"  Graph before: {graph.entity_count} entities, {graph.relation_count} relations")

    type_counts = _populate_graph(graph, results)
    populate_ms = (time.perf_counter() - start) * 1000
    print(f"  Graph after: {graph.entity_count} entities, {graph.relation_count} relations")
    print(f"  Population time: {populate_ms:.0f}ms")

    print("\n  Entity type distribution:")
    for etype, count in type_counts.most_common():
        print(f"    {etype:20s}: {count} mentions")

    # Step 5: Save to SQLite
    _print_header("Step 5: Persisting graph to SQLite")
    start = time.perf_counter()
    graph.save()
    save_ms = (time.perf_counter() - start) * 1000
    print(f"  Saved in {save_ms:.0f}ms")
    print(f"  DB path: {settings.graph_db_path}")

    # Cleanup
    extractor.unload()

    if args.stats:
        _print_stats(graph)

    graph.close()

    _print_header("Done")
    print(f"  Chunks processed: {len(texts):,}")
    print(f"  Entity mentions: {total_entities:,}")
    print(f"  Unique graph entities: {graph.entity_count}")
    print(f"  Total time: {load_ms + extract_ms + populate_ms + save_ms:.0f}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
