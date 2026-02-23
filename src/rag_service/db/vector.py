import uuid

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.core.config import Settings, get_settings
from rag_service.db.models import DocumentChunk


class VectorStore:
    def __init__(self, session: AsyncSession, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()

    async def ensure_extension(self) -> None:
        """Ensure pgvector extension is enabled."""
        await self.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await self.session.commit()

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        filter_document_ids: list[uuid.UUID] | None = None,
        min_similarity: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Perform similarity search using cosine distance.

        Returns chunks with their similarity scores (1 - cosine_distance).
        """
        top_k = top_k or self.settings.top_k

        # Build the query
        # Cosine distance in pgvector: <=> operator
        # Similarity = 1 - distance
        stmt = (
            select(
                DocumentChunk,
                (1 - DocumentChunk.embedding.cosine_distance(query_embedding)).label("similarity"),
            )
            .where(DocumentChunk.embedding.isnot(None))
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )

        # Apply document filter if provided
        if filter_document_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(filter_document_ids))

        result = await self.session.execute(stmt)
        rows = result.all()

        # Filter by minimum similarity
        return [(chunk, sim) for chunk, sim in rows if sim >= min_similarity]

    async def add_embedding(
        self,
        chunk_id: uuid.UUID,
        embedding: list[float],
    ) -> None:
        """Add or update embedding for a chunk."""
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.id == chunk_id)
        )
        result = await self.session.execute(stmt)
        chunk = result.scalar_one_or_none()

        if chunk:
            chunk.embedding = embedding
            await self.session.commit()

    async def bulk_add_embeddings(
        self,
        embeddings: list[tuple[uuid.UUID, list[float]]],
    ) -> None:
        """Bulk add embeddings for multiple chunks."""
        for chunk_id, embedding in embeddings:
            await self.add_embedding(chunk_id, embedding)

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
            if chunk.embedding is not None:
                chunk.embedding = None
                count += 1

        await self.session.commit()
        return count
