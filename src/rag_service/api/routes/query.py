"""Query routes for RAG and semantic search."""

import uuid

from fastapi import APIRouter, status

from rag_service.api.dependencies import (
    CurrentUserDep,
    EmbeddingServiceDep,
    PermissionServiceDep,
    RAGServiceDep,
    VectorStoreDep,
)
from rag_service.core.config import get_embedding_dimension
from rag_service.models.schemas import (
    QueryRequest,
    QueryResponse,
    SearchResponse,
    SearchResult,
)
from rag_service.services.embeddings.factory import EmbeddingProviderFactory

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
)
async def query(
    request: QueryRequest,
    rag_service: RAGServiceDep,
    current_user: CurrentUserDep,
) -> QueryResponse:
    """
    Perform a RAG query with RBAC filtering.

    Retrieves relevant document chunks (only from accessible documents)
    and generates an answer using LLM.

    Optionally specify embedding_provider and embedding_model to override defaults.
    """
    user_id = uuid.UUID(current_user.sub) if current_user.sub else None
    return await rag_service.query(request, user_id=user_id)


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
)
async def semantic_search(
    request: QueryRequest,
    vector_store: VectorStoreDep,
    permission_service: PermissionServiceDep,
    current_user: CurrentUserDep,
) -> SearchResponse:
    """
    Perform semantic search without LLM generation.

    Returns relevant document chunks based on vector similarity.
    Only searches within documents the user has access to.

    Optionally specify embedding_provider and embedding_model to override defaults.
    """
    from rag_service.core.config import get_settings
    from rag_service.db.models import Document
    from sqlalchemy import select

    settings = get_settings()

    # Get user ID
    user_id = uuid.UUID(current_user.sub) if current_user.sub else None

    # Get accessible documents
    accessible_ids = await permission_service.get_accessible_document_ids(user_id, "read")

    if not accessible_ids:
        return SearchResponse(
            results=[],
            query=request.query,
            embedding_dimension=0,
            total_results=0,
        )

    # Determine embedding provider/model
    embedding_provider = request.embedding_provider or settings.default_embedding_provider
    embedding_model = request.embedding_model or settings.default_embedding_model
    dimension = get_embedding_dimension(embedding_provider, embedding_model)

    # Generate query embedding
    provider = EmbeddingProviderFactory.create(
        embedding_provider, embedding_model, settings=settings
    )
    emb_result = await provider.embed(request.query)

    # Perform search with RBAC filtering
    results = await vector_store.similarity_search(
        query_embedding=emb_result.embedding,
        top_k=request.top_k,
        filter_document_ids=accessible_ids,
        embedding_dimension=dimension,
    )

    # Build response with document info
    search_results = []
    for chunk, similarity in results:
        # Get document title
        stmt = select(Document).where(Document.id == chunk.document_id)
        result = await vector_store.session.execute(stmt)
        document = result.scalar_one_or_none()

        if document:
            search_results.append(
                SearchResult(
                    document_id=chunk.document_id,
                    document_title=document.title,
                    chunk_id=chunk.id,
                    chunk_content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    similarity_score=round(similarity, 4),
                    metadata=chunk.metadata_,
                )
            )

    return SearchResponse(
        results=search_results,
        query=request.query,
        embedding_dimension=dimension,
        total_results=len(search_results),
    )
