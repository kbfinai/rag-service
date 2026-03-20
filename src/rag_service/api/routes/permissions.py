"""Document permission management routes."""

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from rag_service.api.dependencies import (
    CurrentUserDep,
    PermissionServiceDep,
    SessionDep,
)
from rag_service.db.models import Document, DocumentPermission
from rag_service.models.schemas import (
    DocumentPermissionCreate,
    DocumentPermissionResponse,
    ShareDocumentRequest,
)

router = APIRouter(prefix="/permissions", tags=["Permissions"])


@router.post(
    "/documents/{document_id}/share",
    response_model=list[DocumentPermissionResponse],
    status_code=status.HTTP_201_CREATED,
)
async def share_document(
    document_id: uuid.UUID,
    request: ShareDocumentRequest,
    session: SessionDep,
    permission_service: PermissionServiceDep,
    current_user: CurrentUserDep,
) -> list[DocumentPermissionResponse]:
    """
    Share a document with one or more users.

    Requires write or admin permission on the document.
    """
    user_id = uuid.UUID(current_user.sub) if current_user.sub else None
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    # Check if user has write access to the document
    has_access = await permission_service.check_document_access(
        user_id, document_id, "write"
    )
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to share this document",
        )

    # Verify document exists
    stmt = select(Document).where(Document.id == document_id)
    result = await session.execute(stmt)
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Grant permissions to each user
    permissions = []
    for target_user_id in request.user_ids:
        perm = await permission_service.grant_permission(
            document_id=document_id,
            user_id=target_user_id,
            permission_level=request.permission_level,
            granted_by=user_id,
        )
        permissions.append(
            DocumentPermissionResponse(
                id=perm.id,
                document_id=perm.document_id,
                user_id=perm.user_id,
                permission_level=perm.permission_level,
                granted_by=perm.granted_by,
                created_at=perm.created_at,
            )
        )

    return permissions


@router.delete(
    "/documents/{document_id}/share/{target_user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def revoke_share(
    document_id: uuid.UUID,
    target_user_id: uuid.UUID,
    permission_service: PermissionServiceDep,
    current_user: CurrentUserDep,
) -> None:
    """
    Revoke a user's access to a document.

    Requires write or admin permission on the document.
    """
    user_id = uuid.UUID(current_user.sub) if current_user.sub else None
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    # Check if user has write access
    has_access = await permission_service.check_document_access(
        user_id, document_id, "write"
    )
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to modify sharing for this document",
        )

    # Revoke the permission
    success = await permission_service.revoke_permission(document_id, target_user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Permission not found",
        )


@router.get(
    "/documents/{document_id}",
    response_model=list[DocumentPermissionResponse],
    status_code=status.HTTP_200_OK,
)
async def list_document_permissions(
    document_id: uuid.UUID,
    session: SessionDep,
    permission_service: PermissionServiceDep,
    current_user: CurrentUserDep,
) -> list[DocumentPermissionResponse]:
    """
    List all permissions for a document.

    Requires read access to the document.
    """
    user_id = uuid.UUID(current_user.sub) if current_user.sub else None
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    # Check if user has read access
    has_access = await permission_service.check_document_access(
        user_id, document_id, "read"
    )
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this document",
        )

    # Get all permissions
    stmt = select(DocumentPermission).where(
        DocumentPermission.document_id == document_id
    )
    result = await session.execute(stmt)
    permissions = result.scalars().all()

    return [
        DocumentPermissionResponse(
            id=perm.id,
            document_id=perm.document_id,
            user_id=perm.user_id,
            permission_level=perm.permission_level,
            granted_by=perm.granted_by,
            created_at=perm.created_at,
        )
        for perm in permissions
    ]
