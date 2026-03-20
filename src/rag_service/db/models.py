import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, func, BigInteger
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

if TYPE_CHECKING:
    pass


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )


# =============================================================================
# RBAC Models
# =============================================================================


class Role(Base, TimestampMixin):
    """Role model for RBAC (admin, editor, viewer)."""

    __tablename__ = "roles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(50), unique=True)
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)
    permissions: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Relationships
    users: Mapped[list["User"]] = relationship(
        secondary="user_roles", back_populates="roles"
    )


class UserRole(Base):
    """Association table for User-Role many-to-many relationship."""

    __tablename__ = "user_roles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    role_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )


class DocumentPermission(Base, TimestampMixin):
    """Document sharing permissions."""

    __tablename__ = "document_permissions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
    )
    permission_level: Mapped[str] = mapped_column(String(20))  # read, write, admin
    granted_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True,
    )

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="permissions")
    user: Mapped["User"] = relationship(
        foreign_keys=[user_id], back_populates="document_permissions"
    )
    granter: Mapped["User | None"] = relationship(foreign_keys=[granted_by])

    __table_args__ = (
        Index("ix_document_permissions_user_id", user_id),
        Index("ix_document_permissions_document_id", document_id),
    )


# =============================================================================
# Provider and Strategy Models
# =============================================================================


class EmbeddingProvider(Base, TimestampMixin):
    """Registry of available embedding providers and models."""

    __tablename__ = "embedding_providers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100))  # openai, azure, huggingface, ollama
    model_name: Mapped[str] = mapped_column(String(200))  # text-embedding-3-small
    dimension: Mapped[int] = mapped_column(Integer)  # 384, 768, 1024, 1536, 3072
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    config: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Relationships
    documents: Mapped[list["Document"]] = relationship(back_populates="embedding_provider")
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="embedding_provider")


class ChunkingStrategy(Base, TimestampMixin):
    """Registry of available chunking strategies."""

    __tablename__ = "chunking_strategies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_config: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Relationships
    documents: Mapped[list["Document"]] = relationship(back_populates="chunking_strategy")


# =============================================================================
# User Model
# =============================================================================


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=True)
    is_superuser: Mapped[bool] = mapped_column(default=False)

    # Relationships
    documents: Mapped[list["Document"]] = relationship(back_populates="owner")
    roles: Mapped[list["Role"]] = relationship(
        secondary="user_roles", back_populates="users"
    )
    document_permissions: Mapped[list["DocumentPermission"]] = relationship(
        foreign_keys=[DocumentPermission.user_id], back_populates="user"
    )


# =============================================================================
# Document Models
# =============================================================================


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )

    # New columns for file upload support
    file_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    original_filename: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)

    # Provider and strategy references
    embedding_provider_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("embedding_providers.id"),
        nullable=True,
    )
    chunking_strategy_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunking_strategies.id"),
        nullable=True,
    )

    # Relationships
    owner: Mapped[User | None] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    permissions: Mapped[list["DocumentPermission"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    embedding_provider: Mapped[EmbeddingProvider | None] = relationship(
        back_populates="documents"
    )
    chunking_strategy: Mapped[ChunkingStrategy | None] = relationship(
        back_populates="documents"
    )


class DocumentChunk(Base, TimestampMixin):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE")
    )
    content: Mapped[str] = mapped_column(Text)
    chunk_index: Mapped[int] = mapped_column()

    # Original embedding column (1536 dimensions for OpenAI default)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536), nullable=True
    )

    # Additional embedding columns for different dimensions
    embedding_384: Mapped[list[float] | None] = mapped_column(
        Vector(384), nullable=True
    )
    embedding_768: Mapped[list[float] | None] = mapped_column(
        Vector(768), nullable=True
    )
    embedding_1024: Mapped[list[float] | None] = mapped_column(
        Vector(1024), nullable=True
    )
    embedding_3072: Mapped[list[float] | None] = mapped_column(
        Vector(3072), nullable=True
    )

    # Track which dimension and provider was used
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_provider_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("embedding_providers.id"),
        nullable=True,
    )

    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    # Relationships
    document: Mapped[Document] = relationship(back_populates="chunks")
    embedding_provider: Mapped[EmbeddingProvider | None] = relationship(
        back_populates="chunks"
    )

    __table_args__ = (
        Index(
            "ix_document_chunks_embedding",
            embedding,
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# =============================================================================
# Knowledge Graph Models (existing)
# =============================================================================


class KnowledgeGraphNode(Base, TimestampMixin):
    __tablename__ = "knowledge_graph_nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(500), index=True)
    node_type: Mapped[str] = mapped_column(String(100), index=True)
    properties: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Relationships
    edges_from: Mapped[list["KnowledgeGraphEdge"]] = relationship(
        foreign_keys="KnowledgeGraphEdge.source_id",
        back_populates="source",
        cascade="all, delete-orphan",
    )
    edges_to: Mapped[list["KnowledgeGraphEdge"]] = relationship(
        foreign_keys="KnowledgeGraphEdge.target_id",
        back_populates="target",
        cascade="all, delete-orphan",
    )


class KnowledgeGraphEdge(Base, TimestampMixin):
    __tablename__ = "knowledge_graph_edges"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_graph_nodes.id", ondelete="CASCADE")
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_graph_nodes.id", ondelete="CASCADE")
    )
    relation_type: Mapped[str] = mapped_column(String(100), index=True)
    properties: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Relationships
    source: Mapped[KnowledgeGraphNode] = relationship(
        foreign_keys=[source_id], back_populates="edges_from"
    )
    target: Mapped[KnowledgeGraphNode] = relationship(
        foreign_keys=[target_id], back_populates="edges_to"
    )

    __table_args__ = (
        Index("ix_knowledge_graph_edges_source_target", source_id, target_id),
    )
