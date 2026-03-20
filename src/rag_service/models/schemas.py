import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# Base schemas
class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# User schemas
# =============================================================================


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class UserResponse(BaseSchema):
    id: uuid.UUID
    email: EmailStr
    is_active: bool
    is_superuser: bool
    created_at: datetime
    roles: list[str] = Field(default_factory=list)


class UserListResponse(BaseModel):
    users: list[UserResponse]
    total: int


# =============================================================================
# Auth schemas
# =============================================================================


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# =============================================================================
# Role & Permission schemas
# =============================================================================


class RoleResponse(BaseSchema):
    id: uuid.UUID
    name: str
    description: str | None
    permissions: dict
    created_at: datetime


class RoleAssignRequest(BaseModel):
    role_name: str


class DocumentPermissionCreate(BaseModel):
    user_id: uuid.UUID
    permission_level: Literal["read", "write", "admin"]


class DocumentPermissionResponse(BaseSchema):
    id: uuid.UUID
    document_id: uuid.UUID
    user_id: uuid.UUID
    user_email: str | None = None
    permission_level: str
    granted_by: uuid.UUID | None
    created_at: datetime


class ShareDocumentRequest(BaseModel):
    user_ids: list[uuid.UUID]
    permission_level: Literal["read", "write"] = "read"


# =============================================================================
# Document schemas
# =============================================================================


class DocumentCreate(BaseModel):
    title: str = Field(max_length=500)
    content: str
    source: str | None = None
    metadata: dict = Field(default_factory=dict)


class DocumentResponse(BaseSchema):
    id: uuid.UUID
    title: str
    content: str
    source: str | None
    metadata: dict
    file_type: str | None = None
    file_size: int | None = None
    original_filename: str | None = None
    is_public: bool = False
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    page: int
    page_size: int


class DocumentChunkResponse(BaseSchema):
    id: uuid.UUID
    document_id: uuid.UUID
    content: str
    chunk_index: int
    metadata: dict
    embedding_dimension: int | None = None


# =============================================================================
# File Upload schemas
# =============================================================================


class FileUploadConfig(BaseModel):
    """Configuration for file upload and processing."""

    embedding_provider: str = Field(
        default="openai",
        description="Embedding provider (openai, azure, huggingface, ollama)",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    chunking_strategy: str = Field(
        default="fixed",
        description="Chunking strategy (fixed, recursive, semantic, token, markdown, invoice)",
    )
    chunking_config: dict = Field(
        default_factory=dict,
        description="Strategy-specific configuration",
    )
    is_public: bool = Field(
        default=False,
        description="Make document publicly accessible",
    )


class FileUploadResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    file_type: str
    file_size: int
    chunks_created: int
    embedding_provider: str
    embedding_model: str
    chunking_strategy: str
    message: str


class BatchUploadResponse(BaseModel):
    documents: list[FileUploadResponse]
    total_files: int
    successful: int
    failed: int
    errors: list[dict] = Field(default_factory=list)


# =============================================================================
# Provider & Strategy schemas
# =============================================================================


class EmbeddingProviderResponse(BaseSchema):
    id: uuid.UUID
    name: str
    model_name: str
    dimension: int
    is_active: bool
    created_at: datetime


class ChunkingStrategyResponse(BaseSchema):
    id: uuid.UUID
    name: str
    description: str | None
    default_config: dict
    created_at: datetime


class ProviderListResponse(BaseModel):
    providers: list[EmbeddingProviderResponse]


class StrategyListResponse(BaseModel):
    strategies: list[ChunkingStrategyResponse]


# =============================================================================
# Query schemas
# =============================================================================


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict = Field(default_factory=dict)
    include_sources: bool = True
    # New fields for provider selection
    embedding_provider: str | None = Field(
        default=None,
        description="Override default embedding provider for this query",
    )
    embedding_model: str | None = Field(
        default=None,
        description="Override default embedding model for this query",
    )


class SourceReference(BaseModel):
    document_id: uuid.UUID
    document_title: str
    chunk_content: str
    relevance_score: float
    chunk_metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    metadata: dict = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Result from semantic search (without LLM generation)."""

    document_id: uuid.UUID
    document_title: str
    chunk_id: uuid.UUID
    chunk_content: str
    chunk_index: int
    similarity_score: float
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    embedding_dimension: int
    total_results: int


# =============================================================================
# Ingest schemas
# =============================================================================


class IngestRequest(BaseModel):
    documents: list[DocumentCreate]
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    chunking_strategy: str = "fixed"
    chunking_config: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_ids: list[uuid.UUID]
    chunks_created: int
    message: str


# =============================================================================
# Health schemas
# =============================================================================


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    timestamp: datetime


# =============================================================================
# Knowledge Graph schemas
# =============================================================================


class NodeCreate(BaseModel):
    name: str = Field(max_length=500)
    node_type: str = Field(max_length=100)
    properties: dict = Field(default_factory=dict)


class NodeResponse(BaseSchema):
    id: uuid.UUID
    name: str
    node_type: str
    properties: dict
    created_at: datetime


class EdgeCreate(BaseModel):
    source_id: uuid.UUID
    target_id: uuid.UUID
    relation_type: str = Field(max_length=100)
    properties: dict = Field(default_factory=dict)


class EdgeResponse(BaseSchema):
    id: uuid.UUID
    source_id: uuid.UUID
    target_id: uuid.UUID
    relation_type: str
    properties: dict
    created_at: datetime


# =============================================================================
# Admin schemas
# =============================================================================


class SystemStatsResponse(BaseModel):
    total_users: int
    total_documents: int
    total_chunks: int
    embedding_stats: dict
    storage_stats: dict
