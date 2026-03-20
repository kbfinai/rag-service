"""RBAC and embedding provider support

Revision ID: 002
Revises: 001
Create Date: 2024-01-15 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create roles table
    op.create_table(
        "roles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(50), unique=True, nullable=False),
        sa.Column("description", sa.String(255), nullable=True),
        sa.Column("permissions", postgresql.JSONB(), default=dict, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Create user_roles table (many-to-many)
    op.create_table(
        "user_roles",
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "role_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("roles.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Create document_permissions table (document sharing)
    op.create_table(
        "document_permissions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "permission_level",
            sa.String(20),
            nullable=False,
        ),  # read, write, admin
        sa.Column(
            "granted_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("document_id", "user_id", name="uq_document_user_permission"),
    )

    # Create embedding_providers table
    op.create_table(
        "embedding_providers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),  # openai, azure, huggingface, ollama
        sa.Column("model_name", sa.String(200), nullable=False),  # text-embedding-3-small
        sa.Column("dimension", sa.Integer(), nullable=False),  # 384, 768, 1536, 3072
        sa.Column("is_active", sa.Boolean(), default=True, nullable=False),
        sa.Column("config", postgresql.JSONB(), default=dict, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("name", "model_name", name="uq_provider_model"),
    )

    # Create chunking_strategies table
    op.create_table(
        "chunking_strategies",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), unique=True, nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("default_config", postgresql.JSONB(), default=dict, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Add new columns to documents table
    op.add_column(
        "documents",
        sa.Column(
            "embedding_provider_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("embedding_providers.id"),
            nullable=True,
        ),
    )
    op.add_column(
        "documents",
        sa.Column(
            "chunking_strategy_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chunking_strategies.id"),
            nullable=True,
        ),
    )
    op.add_column(
        "documents",
        sa.Column("file_type", sa.String(20), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("file_size", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("original_filename", sa.String(500), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("is_public", sa.Boolean(), default=False, server_default="false", nullable=False),
    )

    # Add new columns to document_chunks for multi-dimension embedding support
    op.add_column(
        "document_chunks",
        sa.Column("embedding_dimension", sa.Integer(), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column(
            "embedding_provider_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("embedding_providers.id"),
            nullable=True,
        ),
    )

    # Add additional embedding columns for different dimensions
    # We'll keep the original 1536 column and add others
    op.add_column(
        "document_chunks",
        sa.Column("embedding_384", Vector(384), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("embedding_768", Vector(768), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("embedding_1024", Vector(1024), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("embedding_3072", Vector(3072), nullable=True),
    )

    # Create indexes for new embedding columns
    op.execute(
        """
        CREATE INDEX ix_document_chunks_embedding_384
        ON document_chunks
        USING ivfflat (embedding_384 vector_cosine_ops)
        WITH (lists = 100)
        """
    )
    op.execute(
        """
        CREATE INDEX ix_document_chunks_embedding_768
        ON document_chunks
        USING ivfflat (embedding_768 vector_cosine_ops)
        WITH (lists = 100)
        """
    )
    op.execute(
        """
        CREATE INDEX ix_document_chunks_embedding_1024
        ON document_chunks
        USING ivfflat (embedding_1024 vector_cosine_ops)
        WITH (lists = 100)
        """
    )
    op.execute(
        """
        CREATE INDEX ix_document_chunks_embedding_3072
        ON document_chunks
        USING ivfflat (embedding_3072 vector_cosine_ops)
        WITH (lists = 100)
        """
    )

    # Create indexes for permission lookups
    op.create_index(
        "ix_document_permissions_user_id",
        "document_permissions",
        ["user_id"],
    )
    op.create_index(
        "ix_document_permissions_document_id",
        "document_permissions",
        ["document_id"],
    )

    # Insert default roles
    op.execute(
        """
        INSERT INTO roles (id, name, description, permissions) VALUES
        (gen_random_uuid(), 'admin', 'Full system access', '{"all": true}'::jsonb),
        (gen_random_uuid(), 'editor', 'Can create and edit documents', '{"documents": ["create", "read", "update", "delete"], "query": ["read"]}'::jsonb),
        (gen_random_uuid(), 'viewer', 'Read-only access', '{"documents": ["read"], "query": ["read"]}'::jsonb)
        """
    )

    # Insert default chunking strategies
    op.execute(
        """
        INSERT INTO chunking_strategies (id, name, description, default_config) VALUES
        (gen_random_uuid(), 'fixed', 'Fixed-size chunks with overlap', '{"chunk_size": 512, "overlap": 50}'::jsonb),
        (gen_random_uuid(), 'recursive', 'Recursive splitting respecting document structure', '{"chunk_size": 1000, "separators": ["\\n\\n", "\\n", ". ", " "]}'::jsonb),
        (gen_random_uuid(), 'semantic', 'Paragraph and section-aware chunking', '{"min_chunk_size": 100, "max_chunk_size": 1500}'::jsonb),
        (gen_random_uuid(), 'token', 'Token-based chunking for LLM context limits', '{"max_tokens": 512, "overlap_tokens": 50}'::jsonb),
        (gen_random_uuid(), 'markdown', 'Markdown structure-aware chunking', '{"respect_headers": true, "max_chunk_size": 1500}'::jsonb),
        (gen_random_uuid(), 'invoice', 'Invoice and billing document chunking', '{"extract_line_items": true, "extract_totals": true}'::jsonb)
        """
    )

    # Insert default embedding providers
    op.execute(
        """
        INSERT INTO embedding_providers (id, name, model_name, dimension, is_active, config) VALUES
        (gen_random_uuid(), 'openai', 'text-embedding-3-small', 1536, true, '{}'::jsonb),
        (gen_random_uuid(), 'openai', 'text-embedding-3-large', 3072, true, '{}'::jsonb),
        (gen_random_uuid(), 'openai', 'text-embedding-ada-002', 1536, true, '{}'::jsonb),
        (gen_random_uuid(), 'huggingface', 'all-MiniLM-L6-v2', 384, true, '{}'::jsonb),
        (gen_random_uuid(), 'huggingface', 'e5-large-v2', 1024, true, '{}'::jsonb),
        (gen_random_uuid(), 'huggingface', 'bge-large-en-v1.5', 1024, true, '{}'::jsonb),
        (gen_random_uuid(), 'ollama', 'nomic-embed-text', 768, true, '{}'::jsonb),
        (gen_random_uuid(), 'ollama', 'mxbai-embed-large', 1024, true, '{}'::jsonb)
        """
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_document_chunks_embedding_384")
    op.drop_index("ix_document_chunks_embedding_768")
    op.drop_index("ix_document_chunks_embedding_1024")
    op.drop_index("ix_document_chunks_embedding_3072")
    op.drop_index("ix_document_permissions_user_id")
    op.drop_index("ix_document_permissions_document_id")

    # Remove new columns from document_chunks
    op.drop_column("document_chunks", "embedding_3072")
    op.drop_column("document_chunks", "embedding_1024")
    op.drop_column("document_chunks", "embedding_768")
    op.drop_column("document_chunks", "embedding_384")
    op.drop_column("document_chunks", "embedding_provider_id")
    op.drop_column("document_chunks", "embedding_dimension")

    # Remove new columns from documents
    op.drop_column("documents", "is_public")
    op.drop_column("documents", "original_filename")
    op.drop_column("documents", "file_size")
    op.drop_column("documents", "file_type")
    op.drop_column("documents", "chunking_strategy_id")
    op.drop_column("documents", "embedding_provider_id")

    # Drop tables
    op.drop_table("chunking_strategies")
    op.drop_table("embedding_providers")
    op.drop_table("document_permissions")
    op.drop_table("user_roles")
    op.drop_table("roles")
