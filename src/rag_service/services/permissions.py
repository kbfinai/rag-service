"""Permission service for RBAC (Role-Based Access Control)."""

import uuid
from typing import Literal

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from rag_service.db.models import Document, DocumentPermission, Role, User


PermissionLevel = Literal["read", "write", "admin"]


class PermissionService:
    """Service for managing document permissions and access control."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_user_roles(self, user_id: uuid.UUID) -> list[str]:
        """Get all role names for a user."""
        stmt = (
            select(User)
            .options(selectinload(User.roles))
            .where(User.id == user_id)
        )
        result = await self.session.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            return []

        return [role.name for role in user.roles]

    async def is_admin(self, user_id: uuid.UUID) -> bool:
        """Check if user has admin role or is superuser."""
        # Check if superuser
        stmt = select(User).where(User.id == user_id)
        result = await self.session.execute(stmt)
        user = result.scalar_one_or_none()

        if user and user.is_superuser:
            return True

        # Check if has admin role
        roles = await self.get_user_roles(user_id)
        return "admin" in roles

    async def get_accessible_document_ids(
        self,
        user_id: uuid.UUID,
        permission_level: PermissionLevel = "read",
    ) -> list[uuid.UUID]:
        """
        Get all document IDs the user can access.

        Returns documents where:
        1. User is the owner
        2. Document is shared with user (with required permission level)
        3. Document is public
        4. User is admin (has access to all documents)
        """
        # Check if admin - return all documents
        if await self.is_admin(user_id):
            stmt = select(Document.id)
            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        # Permission level hierarchy: admin > write > read
        permission_levels = self._get_permission_hierarchy(permission_level)

        # Build query for accessible documents
        stmt = select(Document.id).where(
            or_(
                # User is owner
                Document.owner_id == user_id,
                # Document is public
                Document.is_public == True,  # noqa: E712
                # Document is shared with user with required permission
                Document.id.in_(
                    select(DocumentPermission.document_id).where(
                        DocumentPermission.user_id == user_id,
                        DocumentPermission.permission_level.in_(permission_levels),
                    )
                ),
            )
        )

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def check_document_access(
        self,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        required_level: PermissionLevel = "read",
    ) -> bool:
        """Check if user has required permission level on a document."""
        # Check if admin
        if await self.is_admin(user_id):
            return True

        # Get the document
        stmt = select(Document).where(Document.id == document_id)
        result = await self.session.execute(stmt)
        document = result.scalar_one_or_none()

        if not document:
            return False

        # Check if owner
        if document.owner_id == user_id:
            return True

        # Check if public (only for read access)
        if document.is_public and required_level == "read":
            return True

        # Check explicit permissions
        permission_levels = self._get_permission_hierarchy(required_level)
        stmt = select(DocumentPermission).where(
            DocumentPermission.document_id == document_id,
            DocumentPermission.user_id == user_id,
            DocumentPermission.permission_level.in_(permission_levels),
        )
        result = await self.session.execute(stmt)
        permission = result.scalar_one_or_none()

        return permission is not None

    async def grant_permission(
        self,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        permission_level: PermissionLevel,
        granted_by: uuid.UUID,
    ) -> DocumentPermission:
        """Grant a user permission on a document."""
        # Check if permission already exists
        stmt = select(DocumentPermission).where(
            DocumentPermission.document_id == document_id,
            DocumentPermission.user_id == user_id,
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing permission
            existing.permission_level = permission_level
            existing.granted_by = granted_by
            await self.session.commit()
            return existing

        # Create new permission
        permission = DocumentPermission(
            document_id=document_id,
            user_id=user_id,
            permission_level=permission_level,
            granted_by=granted_by,
        )
        self.session.add(permission)
        await self.session.commit()
        await self.session.refresh(permission)
        return permission

    async def revoke_permission(
        self,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> bool:
        """Revoke a user's permission on a document."""
        stmt = select(DocumentPermission).where(
            DocumentPermission.document_id == document_id,
            DocumentPermission.user_id == user_id,
        )
        result = await self.session.execute(stmt)
        permission = result.scalar_one_or_none()

        if not permission:
            return False

        await self.session.delete(permission)
        await self.session.commit()
        return True

    async def get_document_permissions(
        self,
        document_id: uuid.UUID,
    ) -> list[DocumentPermission]:
        """Get all permissions for a document."""
        stmt = (
            select(DocumentPermission)
            .options(selectinload(DocumentPermission.user))
            .where(DocumentPermission.document_id == document_id)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def assign_role(
        self,
        user_id: uuid.UUID,
        role_name: str,
    ) -> bool:
        """Assign a role to a user."""
        # Get user
        stmt = select(User).options(selectinload(User.roles)).where(User.id == user_id)
        result = await self.session.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            return False

        # Get role
        stmt = select(Role).where(Role.name == role_name)
        result = await self.session.execute(stmt)
        role = result.scalar_one_or_none()

        if not role:
            return False

        # Check if already assigned
        if role in user.roles:
            return True

        # Assign role
        user.roles.append(role)
        await self.session.commit()
        return True

    async def remove_role(
        self,
        user_id: uuid.UUID,
        role_name: str,
    ) -> bool:
        """Remove a role from a user."""
        # Get user
        stmt = select(User).options(selectinload(User.roles)).where(User.id == user_id)
        result = await self.session.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            return False

        # Get role
        stmt = select(Role).where(Role.name == role_name)
        result = await self.session.execute(stmt)
        role = result.scalar_one_or_none()

        if not role:
            return False

        # Check if assigned
        if role not in user.roles:
            return True

        # Remove role
        user.roles.remove(role)
        await self.session.commit()
        return True

    async def get_all_roles(self) -> list[Role]:
        """Get all available roles."""
        stmt = select(Role).order_by(Role.name)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    def _get_permission_hierarchy(self, level: PermissionLevel) -> list[str]:
        """
        Get all permission levels that satisfy the required level.

        Permission hierarchy: admin > write > read
        - If 'read' is required, 'read', 'write', and 'admin' all satisfy it
        - If 'write' is required, 'write' and 'admin' satisfy it
        - If 'admin' is required, only 'admin' satisfies it
        """
        hierarchy = {
            "read": ["read", "write", "admin"],
            "write": ["write", "admin"],
            "admin": ["admin"],
        }
        return hierarchy.get(level, ["admin"])
