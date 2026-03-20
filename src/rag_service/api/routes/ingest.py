"""Document ingestion and file upload routes."""

import json
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sqlalchemy import delete, func, select

from rag_service.api.dependencies import (
    CurrentUserDep,
    PermissionServiceDep,
    RAGServiceDep,
    SessionDep,
    SettingsDep,
)
from rag_service.db.models import Document, DocumentChunk
from rag_service.models.schemas import (
    BatchUploadResponse,
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
    FileUploadResponse,
    IngestRequest,
    IngestResponse,
)
from rag_service.services.parsers import FileParserFactory

router = APIRouter(prefix="/documents", tags=["Documents"])


# =============================================================================
# Text-based Ingestion
# =============================================================================


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
    Ingest one or more documents with text content.

    Documents are chunked, embedded, and stored in the vector database.
    Allows selection of embedding provider and chunking strategy.
    """
    document_ids = []
    total_chunks = 0

    owner_id = uuid.UUID(current_user.sub) if current_user.sub else None

    for doc in request.documents:
        doc_id, chunks = await rag_service.ingest_document(
            document=doc,
            owner_id=owner_id,
            embedding_provider=request.embedding_provider,
            embedding_model=request.embedding_model,
            chunking_strategy=request.chunking_strategy,
            chunking_config=request.chunking_config,
        )
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
        file_type=db_doc.file_type,
        file_size=db_doc.file_size,
        original_filename=db_doc.original_filename,
        is_public=db_doc.is_public,
        created_at=db_doc.created_at,
        updated_at=db_doc.updated_at,
    )


# =============================================================================
# File Upload
# =============================================================================


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    embedding_provider: str = Form("openai"),
    embedding_model: str = Form("text-embedding-3-small"),
    chunking_strategy: str = Form("fixed"),
    chunking_config: str = Form("{}"),
    is_public: bool = Form(False),
    rag_service: RAGServiceDep = None,
    current_user: CurrentUserDep = None,
    settings: SettingsDep = None,
) -> FileUploadResponse:
    """
    Upload and process a single file.

    Supported formats: PDF, DOCX, TXT, Markdown, CSV, Excel, HTML, RTF

    Args:
        file: The file to upload
        title: Document title (defaults to filename)
        embedding_provider: Embedding provider (openai, azure, huggingface, ollama)
        embedding_model: Embedding model name
        chunking_strategy: Chunking strategy (fixed, recursive, semantic, token, markdown, invoice)
        chunking_config: JSON string with strategy-specific configuration
        is_public: Make document publicly accessible
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename",
        )

    # Validate file size
    if file.size and file.size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
        )

    # Validate file type
    if not FileParserFactory.is_supported(file.filename):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Supported: {FileParserFactory.list_supported_extensions()}",
        )

    # Parse chunking config
    try:
        config = json.loads(chunking_config) if chunking_config else {}
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chunking_config JSON",
        )

    owner_id = uuid.UUID(current_user.sub) if current_user.sub else None

    try:
        result = await rag_service.ingest_file(
            file=file,
            owner_id=owner_id,
            title=title,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chunking_strategy=chunking_strategy,
            chunking_config=config,
            is_public=is_public,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file: {str(e)}",
        )


@router.post(
    "/upload/batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_files(
    files: list[UploadFile] = File(...),
    embedding_provider: str = Form("openai"),
    embedding_model: str = Form("text-embedding-3-small"),
    chunking_strategy: str = Form("fixed"),
    chunking_config: str = Form("{}"),
    is_public: bool = Form(False),
    rag_service: RAGServiceDep = None,
    current_user: CurrentUserDep = None,
    settings: SettingsDep = None,
) -> BatchUploadResponse:
    """
    Upload and process multiple files.

    Returns results for each file, including any errors.
    """
    # Parse chunking config
    try:
        config = json.loads(chunking_config) if chunking_config else {}
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chunking_config JSON",
        )

    owner_id = uuid.UUID(current_user.sub) if current_user.sub else None

    results = []
    errors = []

    for file in files:
        try:
            # Validate file
            if not file.filename:
                errors.append({"file": "unknown", "error": "File must have a filename"})
                continue

            if file.size and file.size > settings.max_file_size_bytes:
                errors.append({
                    "file": file.filename,
                    "error": f"File too large. Maximum: {settings.max_file_size_mb}MB",
                })
                continue

            if not FileParserFactory.is_supported(file.filename):
                errors.append({
                    "file": file.filename,
                    "error": "Unsupported file type",
                })
                continue

            # Process file
            result = await rag_service.ingest_file(
                file=file,
                owner_id=owner_id,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                chunking_strategy=chunking_strategy,
                chunking_config=config,
                is_public=is_public,
            )
            results.append(result)

        except Exception as e:
            errors.append({
                "file": file.filename or "unknown",
                "error": str(e),
            })

    return BatchUploadResponse(
        documents=results,
        total_files=len(files),
        successful=len(results),
        failed=len(errors),
        errors=errors,
    )


# =============================================================================
# Document Management
# =============================================================================


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_documents(
    session: SessionDep,
    current_user: CurrentUserDep,
    permission_service: PermissionServiceDep,
    page: int = 1,
    page_size: int = 20,
) -> DocumentListResponse:
    """List documents accessible to the current user."""
    user_id = uuid.UUID(current_user.sub) if current_user.sub else None

    # Get accessible document IDs
    accessible_ids = await permission_service.get_accessible_document_ids(user_id, "read")

    # Query documents
    offset = (page - 1) * page_size
    stmt = (
        select(Document)
        .where(Document.id.in_(accessible_ids))
        .offset(offset)
        .limit(page_size)
        .order_by(Document.created_at.desc())
    )
    result = await session.execute(stmt)
    documents = result.scalars().all()

    # Get total count
    count_stmt = select(func.count()).select_from(Document).where(Document.id.in_(accessible_ids))
    count_result = await session.execute(count_stmt)
    total = count_result.scalar() or 0

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=doc.id,
                title=doc.title,
                content=doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                source=doc.source,
                metadata=doc.metadata_,
                file_type=doc.file_type,
                file_size=doc.file_size,
                original_filename=doc.original_filename,
                is_public=doc.is_public,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )
            for doc in documents
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
)
async def get_document(
    document_id: uuid.UUID,
    session: SessionDep,
    current_user: CurrentUserDep,
    permission_service: PermissionServiceDep,
) -> DocumentResponse:
    """Get a specific document by ID (with access check)."""
    user_id = uuid.UUID(current_user.sub) if current_user.sub else None

    # Check access
    has_access = await permission_service.check_document_access(
        user_id, document_id, "read"
    )
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this document",
        )

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
        file_type=document.file_type,
        file_size=document.file_size,
        original_filename=document.original_filename,
        is_public=document.is_public,
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
    permission_service: PermissionServiceDep,
) -> None:
    """Delete a document and its chunks (requires write access)."""
    user_id = uuid.UUID(current_user.sub) if current_user.sub else None

    # Check write access
    has_access = await permission_service.check_document_access(
        user_id, document_id, "write"
    )
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this document",
        )

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
