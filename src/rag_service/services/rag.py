"""RAG Service for document ingestion, querying, and file processing."""

import json
import uuid
from pathlib import Path

from fastapi import UploadFile
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.core.config import Settings, get_settings, get_embedding_dimension
from rag_service.core.logging import get_logger
from rag_service.db.models import (
    ChunkingStrategy,
    Document,
    DocumentChunk,
    EmbeddingProvider,
)
from rag_service.db.vector import VectorStore
from rag_service.models.schemas import (
    DocumentCreate,
    FileUploadResponse,
    QueryRequest,
    QueryResponse,
    SourceReference,
)
from rag_service.services.chunking import ChunkingStrategyFactory
from rag_service.services.embeddings import EmbeddingService
from rag_service.services.embeddings.factory import EmbeddingProviderFactory
from rag_service.services.parsers import FileParserFactory
from rag_service.services.permissions import PermissionService

logger = get_logger(__name__)


class RAGService:
    def __init__(
        self,
        session: AsyncSession,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        permission_service: PermissionService | None = None,
        settings: Settings | None = None,
    ):
        self.session = session
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.permission_service = permission_service
        self.settings = settings or get_settings()
        self.llm_client = AsyncOpenAI(api_key=self.settings.openai_api_key)

    # =========================================================================
    # Document Ingestion
    # =========================================================================

    async def ingest_document(
        self,
        document: DocumentCreate,
        owner_id: uuid.UUID | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        chunking_strategy: str | None = None,
        chunking_config: dict | None = None,
    ) -> tuple[uuid.UUID, int]:
        """
        Ingest a document: store, chunk, and embed.

        Args:
            document: Document content to ingest
            owner_id: Owner user ID
            embedding_provider: Provider to use (openai, huggingface, etc.)
            embedding_model: Model name for embeddings
            chunking_strategy: Chunking strategy name
            chunking_config: Strategy-specific configuration

        Returns:
            Tuple of (document_id, chunk_count)
        """
        # Use defaults if not specified
        embedding_provider = embedding_provider or self.settings.default_embedding_provider
        embedding_model = embedding_model or self.settings.default_embedding_model
        chunking_strategy = chunking_strategy or self.settings.default_chunking_strategy
        chunking_config = chunking_config or {}

        # Get embedding dimension
        dimension = get_embedding_dimension(embedding_provider, embedding_model)

        # Get or create provider record
        provider_record = await self._get_or_create_provider(
            embedding_provider, embedding_model, dimension
        )

        # Get or create strategy record
        strategy_record = await self._get_or_create_strategy(chunking_strategy)

        # Create document record
        db_document = Document(
            title=document.title,
            content=document.content,
            source=document.source,
            metadata_=document.metadata,
            owner_id=owner_id,
            embedding_provider_id=provider_record.id if provider_record else None,
            chunking_strategy_id=strategy_record.id if strategy_record else None,
        )
        self.session.add(db_document)
        await self.session.flush()

        # Chunk the document using selected strategy
        chunker = ChunkingStrategyFactory.create(chunking_strategy)
        chunk_results = chunker.chunk(document.content, chunking_config)

        # Create chunk records
        db_chunks = []
        for idx, chunk_result in enumerate(chunk_results):
            db_chunk = DocumentChunk(
                document_id=db_document.id,
                content=chunk_result.content,
                chunk_index=idx,
                embedding_dimension=dimension,
                embedding_provider_id=provider_record.id if provider_record else None,
                metadata_={
                    "chunk_index": idx,
                    "total_chunks": len(chunk_results),
                    **chunk_result.metadata,
                },
            )
            self.session.add(db_chunk)
            db_chunks.append(db_chunk)

        await self.session.flush()

        # Generate embeddings using selected provider
        try:
            provider = EmbeddingProviderFactory.create(
                embedding_provider, embedding_model, settings=self.settings
            )
            chunk_texts = [c.content for c in chunk_results]
            embedding_results = await provider.embed_batch(chunk_texts)

            # Store embeddings
            for db_chunk, emb_result in zip(db_chunks, embedding_results, strict=True):
                await self.vector_store.add_embedding(
                    chunk_id=db_chunk.id,
                    embedding=emb_result.embedding,
                    dimension=emb_result.dimension,
                    provider_id=provider_record.id if provider_record else None,
                )
        except Exception as e:
            logger.error("Failed to generate embeddings", error=str(e))
            # Continue without embeddings - can be generated later
            pass

        await self.session.commit()

        logger.info(
            "Document ingested",
            document_id=str(db_document.id),
            chunks_created=len(chunk_results),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chunking_strategy=chunking_strategy,
        )

        return db_document.id, len(chunk_results)

    async def ingest_file(
        self,
        file: UploadFile,
        owner_id: uuid.UUID,
        title: str | None = None,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        chunking_strategy: str = "fixed",
        chunking_config: dict | None = None,
        is_public: bool = False,
    ) -> FileUploadResponse:
        """
        Ingest a file: parse, chunk, and embed.

        Args:
            file: Uploaded file
            owner_id: Owner user ID
            title: Document title (defaults to filename)
            embedding_provider: Embedding provider name
            embedding_model: Embedding model name
            chunking_strategy: Chunking strategy name
            chunking_config: Strategy-specific configuration
            is_public: Make document publicly accessible

        Returns:
            FileUploadResponse with document details
        """
        if not file.filename:
            raise ValueError("File must have a filename")

        # Validate file type
        if not FileParserFactory.is_supported(file.filename):
            raise ValueError(
                f"Unsupported file type. Supported: {FileParserFactory.list_supported_extensions()}"
            )

        # Parse the file
        parser = FileParserFactory.get_parser(file.filename)
        parse_result = await parser.parse(file)

        # Get embedding dimension
        dimension = get_embedding_dimension(embedding_provider, embedding_model)

        # Get or create provider record
        provider_record = await self._get_or_create_provider(
            embedding_provider, embedding_model, dimension
        )

        # Get or create strategy record
        strategy_record = await self._get_or_create_strategy(chunking_strategy)

        # Create document record
        doc_title = title or Path(file.filename).stem
        db_document = Document(
            title=doc_title,
            content=parse_result.content,
            source=file.filename,
            metadata_=parse_result.metadata,
            owner_id=owner_id,
            file_type=parse_result.file_type,
            file_size=parse_result.metadata.get("file_size"),
            original_filename=file.filename,
            is_public=is_public,
            embedding_provider_id=provider_record.id if provider_record else None,
            chunking_strategy_id=strategy_record.id if strategy_record else None,
        )
        self.session.add(db_document)
        await self.session.flush()

        # Chunk the content
        chunker = ChunkingStrategyFactory.create(chunking_strategy)
        chunking_config = chunking_config or {}
        chunk_results = chunker.chunk(parse_result.content, chunking_config)

        # Create chunk records
        db_chunks = []
        for idx, chunk_result in enumerate(chunk_results):
            db_chunk = DocumentChunk(
                document_id=db_document.id,
                content=chunk_result.content,
                chunk_index=idx,
                embedding_dimension=dimension,
                embedding_provider_id=provider_record.id if provider_record else None,
                metadata_={
                    "chunk_index": idx,
                    "total_chunks": len(chunk_results),
                    "file_type": parse_result.file_type,
                    **chunk_result.metadata,
                },
            )
            self.session.add(db_chunk)
            db_chunks.append(db_chunk)

        await self.session.flush()

        # Generate embeddings
        try:
            provider = EmbeddingProviderFactory.create(
                embedding_provider, embedding_model, settings=self.settings
            )
            chunk_texts = [c.content for c in chunk_results]
            embedding_results = await provider.embed_batch(chunk_texts)

            for db_chunk, emb_result in zip(db_chunks, embedding_results, strict=True):
                await self.vector_store.add_embedding(
                    chunk_id=db_chunk.id,
                    embedding=emb_result.embedding,
                    dimension=emb_result.dimension,
                    provider_id=provider_record.id if provider_record else None,
                )
        except Exception as e:
            logger.error("Failed to generate embeddings for file", error=str(e))

        await self.session.commit()

        logger.info(
            "File ingested",
            document_id=str(db_document.id),
            filename=file.filename,
            file_type=parse_result.file_type,
            chunks_created=len(chunk_results),
        )

        return FileUploadResponse(
            document_id=db_document.id,
            filename=file.filename,
            file_type=parse_result.file_type,
            file_size=parse_result.metadata.get("file_size", 0),
            chunks_created=len(chunk_results),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chunking_strategy=chunking_strategy,
            message=f"Successfully processed {file.filename}",
        )

    # =========================================================================
    # Query
    # =========================================================================

    async def query(
        self,
        request: QueryRequest,
        user_id: uuid.UUID | None = None,
    ) -> QueryResponse:
        """
        Perform RAG query: retrieve relevant chunks and generate answer.

        Args:
            request: Query request
            user_id: User ID for RBAC filtering (if None, no filtering)

        Returns:
            QueryResponse with answer and sources
        """
        # Get accessible document IDs for RBAC
        filter_document_ids = None
        if user_id and self.permission_service:
            filter_document_ids = await self.permission_service.get_accessible_document_ids(
                user_id, "read"
            )
            if not filter_document_ids:
                return QueryResponse(
                    answer="You don't have access to any documents.",
                    sources=[],
                    metadata={"chunks_retrieved": 0, "access_denied": True},
                )

        # Determine embedding provider/model
        embedding_provider = request.embedding_provider or self.settings.default_embedding_provider
        embedding_model = request.embedding_model or self.settings.default_embedding_model
        dimension = get_embedding_dimension(embedding_provider, embedding_model)

        # Generate query embedding
        try:
            provider = EmbeddingProviderFactory.create(
                embedding_provider, embedding_model, settings=self.settings
            )
            emb_result = await provider.embed(request.query)
            query_embedding = emb_result.embedding
        except Exception as e:
            logger.error("Failed to generate query embedding", error=str(e))
            return QueryResponse(
                answer="Failed to process your query. Please try again.",
                sources=[],
                metadata={"error": str(e)},
            )

        # Retrieve relevant chunks
        results = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            filter_document_ids=filter_document_ids,
            min_similarity=self.settings.min_similarity,
            embedding_dimension=dimension,
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
                        chunk_metadata=chunk.metadata_,
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
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
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

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_or_create_provider(
        self,
        provider_name: str,
        model_name: str,
        dimension: int,
    ) -> EmbeddingProvider | None:
        """Get or create embedding provider record."""
        stmt = select(EmbeddingProvider).where(
            EmbeddingProvider.name == provider_name,
            EmbeddingProvider.model_name == model_name,
        )
        result = await self.session.execute(stmt)
        provider = result.scalar_one_or_none()

        if not provider:
            provider = EmbeddingProvider(
                name=provider_name,
                model_name=model_name,
                dimension=dimension,
                is_active=True,
                config={},
            )
            self.session.add(provider)
            await self.session.flush()

        return provider

    async def _get_or_create_strategy(
        self,
        strategy_name: str,
    ) -> ChunkingStrategy | None:
        """Get or create chunking strategy record."""
        stmt = select(ChunkingStrategy).where(ChunkingStrategy.name == strategy_name)
        result = await self.session.execute(stmt)
        strategy = result.scalar_one_or_none()

        if not strategy:
            default_config = ChunkingStrategyFactory.get_default_config(strategy_name)
            strategy = ChunkingStrategy(
                name=strategy_name,
                description=f"{strategy_name} chunking strategy",
                default_config=default_config,
            )
            self.session.add(strategy)
            await self.session.flush()

        return strategy
