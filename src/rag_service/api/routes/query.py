from fastapi import APIRouter, status

from rag_service.api.dependencies import CurrentUserDep, RAGServiceDep
from rag_service.models.schemas import QueryRequest, QueryResponse

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
    Perform a RAG query.

    Retrieves relevant document chunks and generates an answer using LLM.
    """
    return await rag_service.query(request)


@router.post(
    "/search",
    status_code=status.HTTP_200_OK,
)
async def semantic_search(
    request: QueryRequest,
    rag_service: RAGServiceDep,
    current_user: CurrentUserDep,
) -> dict:
    """
    Perform semantic search without LLM generation.

    Returns relevant document chunks based on vector similarity.
    """
    query_embedding = await rag_service.embedding_service.get_embedding(request.query)
    results = await rag_service.vector_store.similarity_search(
        query_embedding=query_embedding,
        top_k=request.top_k,
    )

    return {
        "results": [
            {
                "chunk_id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "content": chunk.content,
                "similarity": round(similarity, 4),
            }
            for chunk, similarity in results
        ],
        "total": len(results),
    }
