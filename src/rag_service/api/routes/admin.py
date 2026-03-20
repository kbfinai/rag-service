"""Admin routes for managing roles, providers, and strategies."""

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from rag_service.api.dependencies import (
    CurrentUserDep,
    PermissionServiceDep,
    SessionDep,
)
from rag_service.db.models import (
    ChunkingStrategy,
    EmbeddingProvider,
    Role,
    User,
    UserRole,
)
from rag_service.models.schemas import (
    ChunkingStrategyResponse,
    EmbeddingProviderResponse,
    RoleAssignRequest,
    RoleResponse,
    UserResponse,
)

router = APIRouter(prefix="/admin", tags=["Admin"])


async def _require_admin(
    permission_service: PermissionServiceDep,
    current_user: CurrentUserDep,
) -> uuid.UUID:
    """Check if current user is admin and return user ID."""
    user_id = uuid.UUID(current_user.sub) if current_user.sub else None
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    is_admin = await permission_service.is_admin(user_id)
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    return user_id


# =============================================================================
# User Management
# =============================================================================


@router.get(
    "/users",
    response_model=list[UserResponse],
    status_code=status.HTTP_200_OK,
)
async def list_users(
    session: SessionDep,
    permission_service: PermissionServiceDep,
    current_user: CurrentUserDep,
) -> list[UserResponse]:
    """List all users (admin only)."""
    await _require_admin(permission_service, current_user)

    stmt = select(User).order_by(User.created_at.desc())
    result = await session.execute(stmt)
    users = result.scalars().all()

    user_responses = []
    for user in users:
        # Get user roles
        role_stmt = (
            select(Role)
            .join(UserRole, Role.id == UserRole.role_id)
            .where(UserRole.user_id == user.id)
        )
        role_result = await session.execute(role_stmt)
        roles = role_result.scalars().all()

        user_responses.append(
            UserResponse(
                id=user.id,
                email=user.email,
                is_active=user.is_active,
                roles=[role.name for role in roles],
                created_at=user.created_at,
            )
        )

    return user_responses


@router.post(
    "/users/{user_id}/roles",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
)
async def assign_role(
    user_id: uuid.UUID,
    request: RoleAssignRequest,
    session: SessionDep,
    permission_service: PermissionServiceDep,
    current_user: CurrentUserDep,
) -> UserResponse:
    """Assign a role to a user (admin only)."""
    await _require_admin(permission_service, current_user)

    # Check if user exists
    user_stmt = select(User).where(User.id == user_id)
    user_result = await session.execute(user_stmt)
    user = user_result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Get role
    role_stmt = select(Role).where(Role.name == request.role_name)
    role_result = await session.execute(role_stmt)
    role = role_result.scalar_one_or_none()

    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role '{request.role_name}' not found",
        )

    # Check if assignment already exists
    existing_stmt = select(UserRole).where(
        UserRole.user_id == user_id,
        UserRole.role_id == role.id,
    )
    existing_result = await session.execute(existing_stmt)
    existing = existing_result.scalar_one_or_none()

    if not existing:
        # Create new assignment
        user_role = UserRole(user_id=user_id, role_id=role.id)
        session.add(user_role)
        await session.commit()

    # Get all roles for response
    roles_stmt = (
        select(Role)
        .join(UserRole, Role.id == UserRole.role_id)
        .where(UserRole.user_id == user_id)
    )
    roles_result = await session.execute(roles_stmt)
    roles = roles_result.scalars().all()

    return UserResponse(
        id=user.id,
        email=user.email,
        is_active=user.is_active,
        roles=[r.name for r in roles],
        created_at=user.created_at,
    )


@router.delete(
    "/users/{user_id}/roles/{role_name}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_role(
    user_id: uuid.UUID,
    role_name: str,
    session: SessionDep,
    permission_service: PermissionServiceDep,
    current_user: CurrentUserDep,
) -> None:
    """Remove a role from a user (admin only)."""
    await _require_admin(permission_service, current_user)

    # Get role
    role_stmt = select(Role).where(Role.name == role_name)
    role_result = await session.execute(role_stmt)
    role = role_result.scalar_one_or_none()

    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role '{role_name}' not found",
        )

    # Find and delete assignment
    stmt = select(UserRole).where(
        UserRole.user_id == user_id,
        UserRole.role_id == role.id,
    )
    result = await session.execute(stmt)
    user_role = result.scalar_one_or_none()

    if not user_role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role assignment not found",
        )

    await session.delete(user_role)
    await session.commit()


# =============================================================================
# Roles
# =============================================================================


@router.get(
    "/roles",
    response_model=list[RoleResponse],
    status_code=status.HTTP_200_OK,
)
async def list_roles(
    session: SessionDep,
    permission_service: PermissionServiceDep,
    current_user: CurrentUserDep,
) -> list[RoleResponse]:
    """List all available roles (admin only)."""
    await _require_admin(permission_service, current_user)

    stmt = select(Role).order_by(Role.name)
    result = await session.execute(stmt)
    roles = result.scalars().all()

    return [
        RoleResponse(
            id=role.id,
            name=role.name,
            permissions=role.permissions,
            created_at=role.created_at,
        )
        for role in roles
    ]


# =============================================================================
# Embedding Providers
# =============================================================================


@router.get(
    "/providers",
    response_model=list[EmbeddingProviderResponse],
    status_code=status.HTTP_200_OK,
)
async def list_embedding_providers(
    session: SessionDep,
    current_user: CurrentUserDep,
) -> list[EmbeddingProviderResponse]:
    """List all available embedding providers."""
    stmt = select(EmbeddingProvider).where(EmbeddingProvider.is_active == True).order_by(EmbeddingProvider.name)
    result = await session.execute(stmt)
    providers = result.scalars().all()

    return [
        EmbeddingProviderResponse(
            id=provider.id,
            name=provider.name,
            model_name=provider.model_name,
            dimension=provider.dimension,
            is_active=provider.is_active,
            created_at=provider.created_at,
        )
        for provider in providers
    ]


# =============================================================================
# Chunking Strategies
# =============================================================================


@router.get(
    "/strategies",
    response_model=list[ChunkingStrategyResponse],
    status_code=status.HTTP_200_OK,
)
async def list_chunking_strategies(
    session: SessionDep,
    current_user: CurrentUserDep,
) -> list[ChunkingStrategyResponse]:
    """List all available chunking strategies."""
    stmt = select(ChunkingStrategy).order_by(ChunkingStrategy.name)
    result = await session.execute(stmt)
    strategies = result.scalars().all()

    return [
        ChunkingStrategyResponse(
            id=strategy.id,
            name=strategy.name,
            description=strategy.description,
            default_config=strategy.default_config,
            created_at=strategy.created_at,
        )
        for strategy in strategies
    ]
