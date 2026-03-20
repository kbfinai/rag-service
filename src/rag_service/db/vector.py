"""Vector store for similarity search with multi-dimension support."""

import uuid
from typing import Literal

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.core.config import Settings, get_settings
from rag_service.db.models import DocumentChunk


EmbeddingDimension = Literal[384, 768, 1024, 1536, 3072]

# Mapping of dimensions to column names
DIMENSION_COLUMNS = {
    384: "embedding_384",
    768: "embedding_768",
    1024: "embedding_1024",
    1536: "embedding",  # Default column
    3072: "embedding_3072",
}


class VectorStore:
    def __init__(self, session: AsyncSession, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()

    async def ensure_extension(self) -> None:
        """Ensure pgvector extension is enabled."""
        await self.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await self.session.commit()

    def _get_embedding_column(self, dimension: int):
        """Get the appropriate embedding column for the given dimension."""
        column_name = DIMENSION_COLUMNS.get(dimension, "embedding")
        return getattr(DocumentChunk, column_name)

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        filter_document_ids: list[uuid.UUID] | None = None,
        min_similarity: float = 0.0,
        embedding_dimension: int | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Perform similarity search using cosine distance.

        Args:
            query_embedding: The query vector
            top_k: Maximum number of results to return
            filter_document_ids: Only search within these documents (for RBAC)
            min_similarity: Minimum similarity score to include
            embedding_dimension: Dimension of the embedding (determines which column to search)

        Returns:
            List of (chunk, similarity_score) tuples
        """
        top_k = top_k or self.settings.top_k

        # Determine dimension from embedding if not specified
        if embedding_dimension is None:
            embedding_dimension = len(query_embedding)

        # Get the appropriate column for this dimension
        embedding_column = self._get_embedding_column(embedding_dimension)

        # Build the query
        # Cosine distance in pgvector: <=> operator
        # Similarity = 1 - distance
        stmt = (
            select(
                DocumentChunk,
                (1 - embedding_column.cosine_distance(query_embedding)).label("similarity"),
            )
            .where(embedding_column.isnot(None))
            .order_by(embedding_column.cosine_distance(query_embedding))
            .limit(top_k)
        )

        # Apply document filter if provided (for RBAC)
        if filter_document_ids is not None:
            if len(filter_document_ids) == 0:
                # No accessible documents - return empty
                return []
            stmt = stmt.where(DocumentChunk.document_id.in_(filter_document_ids))

        result = await self.session.execute(stmt)
        rows = result.all()

        # Filter by minimum similarity
        return [(chunk, sim) for chunk, sim in rows if sim >= min_similarity]

    async def add_embedding(
        self,
        chunk_id: uuid.UUID,
        embedding: list[float],
        dimension: int | None = None,
        provider_id: uuid.UUID | None = None,
    ) -> None:
        """
        Add or update embedding for a chunk.

        Args:
            chunk_id: ID of the chunk to update
            embedding: The embedding vector
            dimension: Dimension of the embedding (auto-detected if not provided)
            provider_id: ID of the embedding provider used
        """
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.id == chunk_id)
        )
        result = await self.session.execute(stmt)
        chunk = result.scalar_one_or_none()

        if not chunk:
            return

        # Determine dimension
        if dimension is None:
            dimension = len(embedding)

        # Get the appropriate column
        column_name = DIMENSION_COLUMNS.get(dimension, "embedding")

        # Set the embedding on the appropriate column
        setattr(chunk, column_name, embedding)

        # Update metadata
        chunk.embedding_dimension = dimension
        if provider_id:
            chunk.embedding_provider_id = provider_id

        await self.session.commit()

    async def bulk_add_embeddings(
        self,
        embeddings: list[tuple[uuid.UUID, list[float]]],
        dimension: int | None = None,
        provider_id: uuid.UUID | None = None,
    ) -> None:
        """
        Bulk add embeddings for multiple chunks.

        Args:
            embeddings: List of (chunk_id, embedding) tuples
            dimension: Dimension of the embeddings
            provider_id: ID of the embedding provider used
        """
        for chunk_id, embedding in embeddings:
            await self.add_embedding(chunk_id, embedding, dimension, provider_id)

    async def delete_embeddings(
        self,
        document_id: uuid.UUID,
    ) -> int:
        """Delete all embeddings for a document's chunks."""
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
        )
        result = await self.session.execute(stmt)
        chunks = result.scalars().all()

        count = 0
        for chunk in chunks:
            # Clear all embedding columns
            if chunk.embedding is not None:
                chunk.embedding = None
                count += 1
            if chunk.embedding_384 is not None:
                chunk.embedding_384 = None
            if chunk.embedding_768 is not None:
                chunk.embedding_768 = None
            if chunk.embedding_1024 is not None:
                chunk.embedding_1024 = None
            if chunk.embedding_3072 is not None:
                chunk.embedding_3072 = None

            chunk.embedding_dimension = None
            chunk.embedding_provider_id = None

        await self.session.commit()
        return count

    async def get_embedding_stats(
        self,
        document_id: uuid.UUID | None = None,
    ) -> dict:
        """Get statistics about embeddings in the store."""
        base_stmt = select(DocumentChunk)
        if document_id:
            base_stmt = base_stmt.where(DocumentChunk.document_id == document_id)

        result = await self.session.execute(base_stmt)
        chunks = result.scalars().all()

        stats = {
            "total_chunks": len(chunks),
            "chunks_with_embeddings": 0,
            "dimensions": {},
        }

        for chunk in chunks:
            has_embedding = False
            if chunk.embedding is not None:
                has_embedding = True
                stats["dimensions"]["1536"] = stats["dimensions"].get("1536", 0) + 1
            if chunk.embedding_384 is not None:
                has_embedding = True
                stats["dimensions"]["384"] = stats["dimensions"].get("384", 0) + 1
            if chunk.embedding_768 is not None:
                has_embedding = True
                stats["dimensions"]["768"] = stats["dimensions"].get("768", 0) + 1
            if chunk.embedding_1024 is not None:
                has_embedding = True
                stats["dimensions"]["1024"] = stats["dimensions"].get("1024", 0) + 1
            if chunk.embedding_3072 is not None:
                has_embedding = True
                stats["dimensions"]["3072"] = stats["dimensions"].get("3072", 0) + 1

            if has_embedding:
                stats["chunks_with_embeddings"] += 1

        return stats
