import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import delete, select

from rag_service.api.dependencies import CurrentUserDep, RAGServiceDep, SessionDep
from rag_service.db.models import Document, DocumentChunk
from rag_service.models.schemas import (
    DocumentCreate,
    DocumentResponse,
    IngestRequest,
    IngestResponse,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_documents(
    request: IngestRequest,
    rag_service: RAGServiceDep,
    current_user: CurrentUserDep,
) -> IngestResponse:
    """
    Ingest one or more documents.

    Documents are chunked, embedded, and stored in the vector database.
    """
    document_ids = []
    total_chunks = 0

    owner_id = uuid.UUID(current_user.sub) if current_user.sub else None

    for doc in request.documents:
        doc_id, chunks = await rag_service.ingest_document(doc, owner_id)
        document_ids.append(doc_id)
        total_chunks += chunks

    return IngestResponse(
        document_ids=document_ids,
        chunks_created=total_chunks,
        message=f"Successfully ingested {len(document_ids)} document(s)",
    )


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
    document: DocumentCreate,
    rag_service: RAGServiceDep,
    current_user: CurrentUserDep,
) -> DocumentResponse:
    """Create and ingest a single document."""
    owner_id = uuid.UUID(current_user.sub) if current_user.sub else None
    doc_id, _ = await rag_service.ingest_document(document, owner_id)

    # Fetch the created document
    stmt = select(Document).where(Document.id == doc_id)
    result = await rag_service.session.execute(stmt)
    db_doc = result.scalar_one()

    return DocumentResponse(
        id=db_doc.id,
        title=db_doc.title,
        content=db_doc.content,
        source=db_doc.source,
        metadata=db_doc.metadata_,
        created_at=db_doc.created_at,
        updated_at=db_doc.updated_at,
    )


@router.get(
    "",
    response_model=list[DocumentResponse],
    status_code=status.HTTP_200_OK,
)
async def list_documents(
    session: SessionDep,
    current_user: CurrentUserDep,
    skip: int = 0,
    limit: int = 100,
) -> list[DocumentResponse]:
    """List all documents."""
    stmt = select(Document).offset(skip).limit(limit).order_by(Document.created_at.desc())
    result = await session.execute(stmt)
    documents = result.scalars().all()

    return [
        DocumentResponse(
            id=doc.id,
            title=doc.title,
            content=doc.content,
            source=doc.source,
            metadata=doc.metadata_,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )
        for doc in documents
    ]


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
)
async def get_document(
    document_id: uuid.UUID,
    session: SessionDep,
    current_user: CurrentUserDep,
) -> DocumentResponse:
    """Get a specific document by ID."""
    stmt = select(Document).where(Document.id == document_id)
    result = await session.execute(stmt)
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return DocumentResponse(
        id=document.id,
        title=document.title,
        content=document.content,
        source=document.source,
        metadata=document.metadata_,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    document_id: uuid.UUID,
    session: SessionDep,
    current_user: CurrentUserDep,
) -> None:
    """Delete a document and its chunks."""
    # Check if document exists
    stmt = select(Document).where(Document.id == document_id)
    result = await session.execute(stmt)
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Delete chunks first (cascade should handle this, but explicit is better)
    await session.execute(
        delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
    )

    # Delete document
    await session.delete(document)
    await session.commit()
