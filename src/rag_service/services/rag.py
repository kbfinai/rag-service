import uuid

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.core.config import Settings, get_settings
from rag_service.core.logging import get_logger
from rag_service.db.models import Document, DocumentChunk
from rag_service.db.vector import VectorStore
from rag_service.models.schemas import (
    DocumentCreate,
    QueryRequest,
    QueryResponse,
    SourceReference,
)
from rag_service.services.embeddings import EmbeddingService

logger = get_logger(__name__)


class RAGService:
    def __init__(
        self,
        session: AsyncSession,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        settings: Settings | None = None,
    ):
        self.session = session
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.settings = settings or get_settings()
        self.llm_client = AsyncOpenAI(api_key=self.settings.openai_api_key)

    async def ingest_document(
        self,
        document: DocumentCreate,
        owner_id: uuid.UUID | None = None,
    ) -> tuple[uuid.UUID, int]:
        """Ingest a document: store, chunk, and embed."""
        # Create document
        db_document = Document(
            title=document.title,
            content=document.content,
            source=document.source,
            metadata_=document.metadata,
            owner_id=owner_id,
        )
        self.session.add(db_document)
        await self.session.flush()

        # Chunk the document
        chunks = self.embedding_service.chunk_text(document.content)

        # Create chunk records
        db_chunks = []
        for idx, chunk_content in enumerate(chunks):
            db_chunk = DocumentChunk(
                document_id=db_document.id,
                content=chunk_content,
                chunk_index=idx,
                metadata_={"chunk_index": idx, "total_chunks": len(chunks)},
            )
            self.session.add(db_chunk)
            db_chunks.append(db_chunk)

        await self.session.flush()

        # Generate embeddings
        embeddings = await self.embedding_service.get_embeddings(chunks)

        # Store embeddings
        for db_chunk, embedding in zip(db_chunks, embeddings, strict=True):
            db_chunk.embedding = embedding

        await self.session.commit()

        logger.info(
            "Document ingested",
            document_id=str(db_document.id),
            chunks_created=len(chunks),
        )

        return db_document.id, len(chunks)

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Perform RAG query: retrieve relevant chunks and generate answer."""
        # Generate query embedding
        query_embedding = await self.embedding_service.get_embedding(request.query)

        # Retrieve relevant chunks
        results = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            min_similarity=0.5,
        )

        if not results:
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                metadata={"chunks_retrieved": 0},
            )

        # Build context from chunks
        context_parts = []
        sources = []

        for chunk, similarity in results:
            # Get document info
            stmt = select(Document).where(Document.id == chunk.document_id)
            result = await self.session.execute(stmt)
            document = result.scalar_one_or_none()

            if document:
                context_parts.append(f"[Source: {document.title}]\n{chunk.content}")
                sources.append(
                    SourceReference(
                        document_id=document.id,
                        document_title=document.title,
                        chunk_content=chunk.content[:500],
                        relevance_score=round(similarity, 4),
                    )
                )

        context = "\n\n---\n\n".join(context_parts)

        # Generate answer using LLM
        answer = await self._generate_answer(request.query, context)

        return QueryResponse(
            answer=answer,
            sources=sources if request.include_sources else [],
            metadata={
                "chunks_retrieved": len(results),
                "model": self.settings.llm_model,
            },
        )

    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use the context to answer the question accurately.
If the context doesn't contain enough information to answer the question, say so clearly.
Always be concise and factual."""

        user_prompt = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""

        response = await self.llm_client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        return response.choices[0].message.content or "Unable to generate response."
