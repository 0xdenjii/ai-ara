"""Embedding model wrapper for memory retrieval.

Provides text embedding using nomic-embed-text-v1.5 via sentence-transformers.
Runs entirely on CPU — no VRAM consumed. Produces 768-dimensional vectors.

The model is loaded lazily on first embed() call to avoid blocking startup.
All configuration comes from AraSettings.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ara.utils.errors import AraMemoryError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from ara.config.settings import MemorySettings

log = get_logger("embeddings")

# Nomic embed prefix — the model expects task-specific prefixes
# for improved retrieval quality.
_SEARCH_DOCUMENT_PREFIX = "search_document: "
_SEARCH_QUERY_PREFIX = "search_query: "


class EmbeddingModel:
    """Text embedding model for vector search.

    Wraps sentence-transformers to produce embeddings for memory chunks
    and search queries. Uses nomic-embed-text-v1.5 by default (768-dim,
    ~274MB, CPU-only).

    The model uses task-specific prefixes:
    - Documents get "search_document: " prefix
    - Queries get "search_query: " prefix
    This improves retrieval accuracy via asymmetric embedding.

    Usage:
        model = EmbeddingModel(settings)
        model.load()
        doc_vecs = model.embed_documents(["text1", "text2"])
        query_vec = model.embed_query("what happened yesterday?")
    """

    def __init__(self, settings: MemorySettings) -> None:
        self._settings = settings
        self._model: object | None = None
        self._loaded = False
        self._dimension: int = 0

    @property
    def is_loaded(self) -> bool:
        """Whether the embedding model is loaded."""
        return self._loaded

    @property
    def dimension(self) -> int:
        """Embedding vector dimension (768 for nomic-embed-text-v1.5)."""
        return self._dimension

    def load(self) -> None:
        """Load the embedding model into memory (CPU).

        Raises:
            AraMemoryError: If model loading fails.
        """
        if self._loaded:
            return

        model_name = self._settings.embedding_model
        log.info("embedding_model_loading", model=model_name)
        start = time.monotonic()

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device="cpu",
            )
            # Get dimension from a test encoding
            test_vec = self._model.encode(["test"], show_progress_bar=False)
            self._dimension = test_vec.shape[1]
            self._loaded = True

            elapsed_ms = (time.monotonic() - start) * 1000
            log.info(
                "embedding_model_loaded",
                model=model_name,
                dimension=self._dimension,
                load_time_ms=round(elapsed_ms, 1),
            )
        except ImportError as exc:
            raise AraMemoryError(
                "sentence-transformers not installed. Run: pip install sentence-transformers",
                operation="embedding_load",
            ) from exc
        except Exception as exc:
            raise AraMemoryError(
                f"Failed to load embedding model '{model_name}': {exc}",
                operation="embedding_load",
                details={"model": model_name},
            ) from exc

    def embed_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Embed a list of document texts.

        Prepends "search_document: " to each text for asymmetric retrieval.

        Args:
            texts: List of document strings to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.

        Returns:
            List of embedding vectors (each is a list of floats).

        Raises:
            AraMemoryError: If embedding fails or model not loaded.
        """
        if not self._loaded or self._model is None:
            raise AraMemoryError(
                "Embedding model not loaded — call load() first",
                operation="embed_documents",
            )

        if not texts:
            return []

        # Prepend document prefix for nomic's asymmetric retrieval
        prefixed = [_SEARCH_DOCUMENT_PREFIX + t for t in texts]

        start = time.monotonic()
        try:
            vectors = self._model.encode(
                prefixed,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            log.debug(
                "documents_embedded",
                count=len(texts),
                elapsed_ms=round(elapsed_ms, 1),
            )
            return vectors.tolist()
        except Exception as exc:
            raise AraMemoryError(
                f"Document embedding failed: {exc}",
                operation="embed_documents",
                details={"text_count": len(texts)},
            ) from exc

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query.

        Prepends "search_query: " for asymmetric retrieval.

        Args:
            query: The search query string.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            AraMemoryError: If embedding fails or model not loaded.
        """
        if not self._loaded or self._model is None:
            raise AraMemoryError(
                "Embedding model not loaded — call load() first",
                operation="embed_query",
            )

        prefixed = _SEARCH_QUERY_PREFIX + query

        start = time.monotonic()
        try:
            vector = self._model.encode(
                [prefixed],
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            log.debug(
                "query_embedded",
                query_len=len(query),
                elapsed_ms=round(elapsed_ms, 1),
            )
            return vector[0].tolist()
        except Exception as exc:
            raise AraMemoryError(
                f"Query embedding failed: {exc}",
                operation="embed_query",
                details={"query_len": len(query)},
            ) from exc

    def embed_batch(
        self,
        texts: list[str],
        *,
        prefix: str = "",
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed texts with a custom prefix (or no prefix).

        Lower-level method for cases where the caller controls the prefix.

        Args:
            texts: Texts to embed.
            prefix: Optional prefix to prepend.
            batch_size: Batch size.

        Returns:
            List of embedding vectors.
        """
        if not self._loaded or self._model is None:
            raise AraMemoryError(
                "Embedding model not loaded — call load() first",
                operation="embed_batch",
            )

        if not texts:
            return []

        prefixed = [prefix + t for t in texts] if prefix else texts

        try:
            vectors = self._model.encode(
                prefixed,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return vectors.tolist()
        except Exception as exc:
            raise AraMemoryError(
                f"Batch embedding failed: {exc}",
                operation="embed_batch",
                details={"text_count": len(texts)},
            ) from exc

    def unload(self) -> None:
        """Release the model from memory."""
        self._model = None
        self._loaded = False
        self._dimension = 0
        log.info("embedding_model_unloaded")
