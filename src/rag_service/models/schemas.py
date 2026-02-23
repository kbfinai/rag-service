import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# Base schemas
class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# User schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class UserResponse(BaseSchema):
    id: uuid.UUID
    email: EmailStr
    is_active: bool
    created_at: datetime


# Auth schemas
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# Document schemas
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
    created_at: datetime
    updated_at: datetime


class DocumentChunkResponse(BaseSchema):
    id: uuid.UUID
    document_id: uuid.UUID
    content: str
    chunk_index: int
    metadata: dict


# Query schemas
class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict = Field(default_factory=dict)
    include_sources: bool = True


class SourceReference(BaseModel):
    document_id: uuid.UUID
    document_title: str
    chunk_content: str
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    metadata: dict = Field(default_factory=dict)


# Ingest schemas
class IngestRequest(BaseModel):
    documents: list[DocumentCreate]


class IngestResponse(BaseModel):
    document_ids: list[uuid.UUID]
    chunks_created: int
    message: str


# Health schemas
class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    timestamp: datetime


# Knowledge Graph schemas
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
