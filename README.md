# RAG Service

**Retrieval Augmented Generation as a Service for kbfinai**

---

## Overview

RAG Service is the centralized knowledge retrieval and generation engine powering all finance tools in the kbfinai ecosystem. It provides a unified API for intelligent document retrieval, contextual understanding, and AI-powered responses across our financial services platform.

---

## Features

| Feature | Description |
|---------|-------------|
| **Unified Knowledge Base** | Single source of truth for all financial documents, policies, and data |
| **Multi-Provider Embeddings** | Support for OpenAI, Azure OpenAI, HuggingFace, and Ollama |
| **Flexible Chunking** | Multiple strategies including semantic, markdown-aware, and invoice-specific |
| **File Upload Support** | Direct upload of PDF, DOCX, TXT, Markdown, CSV, Excel, HTML, RTF |
| **Full RBAC** | Role-based access control with document sharing and permissions |
| **Semantic Search** | Vector-based retrieval for accurate, context-aware document matching |
| **Multi-Tool Integration** | Seamless API access for all kbfinai services |
| **Scalable Architecture** | Designed to handle enterprise-grade financial queries |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Package Manager | uv |
| API Framework | FastAPI + Uvicorn |
| Database | PostgreSQL 16 + pgvector |
| ORM | SQLAlchemy 2.0 (async) |
| Embeddings | OpenAI, Azure, HuggingFace, Ollama |
| Auth | JWT (python-jose) + RBAC |
| Migrations | Alembic |
| File Parsing | pypdf, python-docx, openpyxl, beautifulsoup4 |
| Linting | Ruff |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        kbfinai Services                         │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  Analytics   │   Reports    │   Insights   │   Other Tools     │
└──────┬───────┴──────┬───────┴──────┬───────┴─────────┬─────────┘
       │              │              │                 │
       └──────────────┴──────────────┴─────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │      RAG Service       │
                 │  ┌──────────────────┐  │
                 │  │   File Parser    │  │
                 │  │ (PDF/DOCX/etc.)  │  │
                 │  └────────┬─────────┘  │
                 │           │            │
                 │  ┌────────▼─────────┐  │
                 │  │    Chunking      │  │
                 │  │  (6 strategies)  │  │
                 │  └────────┬─────────┘  │
                 │           │            │
                 │  ┌────────▼─────────┐  │
                 │  │   Embeddings     │  │
                 │  │  (4 providers)   │  │
                 │  └────────┬─────────┘  │
                 │           │            │
                 │  ┌────────▼─────────┐  │
                 │  │  Vector Store    │  │
                 │  │   (pgvector)     │  │
                 │  └────────┬─────────┘  │
                 │           │            │
                 │  ┌────────▼─────────┐  │
                 │  │  LLM Generation  │  │
                 │  └──────────────────┘  │
                 │           │            │
                 │  ┌────────▼─────────┐  │
                 │  │      RBAC        │  │
                 │  │  (permissions)   │  │
                 │  └──────────────────┘  │
                 └────────────────────────┘
```

---

## Embedding Providers

| Provider | Models | Dimensions |
|----------|--------|------------|
| **OpenAI** | text-embedding-3-small, text-embedding-3-large, ada-002 | 1536, 3072 |
| **Azure OpenAI** | text-embedding-3-small, text-embedding-3-large, ada-002 | 1536, 3072 |
| **HuggingFace** | all-MiniLM-L6-v2, e5-large-v2, bge-large-en-v1.5 | 384, 768, 1024 |
| **Ollama** | nomic-embed-text, mxbai-embed-large, all-minilm | 384, 768, 1024 |

---

## Chunking Strategies

| Strategy | Best For | Description |
|----------|----------|-------------|
| **fixed** | General text | Fixed-size chunks with overlap |
| **recursive** | Structured docs | Respects headers and sections |
| **semantic** | Natural text | Paragraph and sentence boundaries |
| **token** | LLM optimization | Token-count based chunking |
| **markdown** | MD/documentation | Preserves markdown structure |
| **invoice** | Financial docs | Extracts header, line items, summary |

---

## Supported File Formats

| Format | Extensions | Features |
|--------|------------|----------|
| PDF | `.pdf` | Text extraction, OCR fallback for scanned docs |
| Word | `.docx` | Full text and table extraction |
| Text | `.txt` | Encoding auto-detection |
| Markdown | `.md` | Structure-aware parsing |
| CSV | `.csv` | Row-based text conversion |
| Excel | `.xlsx`, `.xls` | Multi-sheet support |
| HTML | `.html`, `.htm` | Clean text extraction |
| RTF | `.rtf` | Rich text format support |

---

## RBAC System

### Roles

| Role | Permissions |
|------|-------------|
| **admin** | Full access to all documents and system settings |
| **editor** | Read/write access to own and shared documents |
| **viewer** | Read-only access to shared and public documents |

### Document Access

- **Ownership**: Document creators have full control
- **Sharing**: Grant read/write access to specific users
- **Public**: Optionally make documents accessible to all users
- **Filtering**: Queries automatically filter by user permissions

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker & Docker Compose (for local development)
- OpenAI API key (or other provider credentials)

---

## Quick Start

### Option 1: Local Development

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/kbfinai/rag-service.git
cd rag-service

# Install dependencies
uv sync

# Copy environment file and configure
cp .env.example .env
# Edit .env and add your API keys

# Start PostgreSQL with pgvector
docker compose up db -d

# Wait for database to be ready, then run migrations
uv run alembic upgrade head

# Start the service
uv run rag-service
```

The API will be available at `http://localhost:8000`

### Option 2: Docker Compose (Full Stack)

```bash
# Clone the repository
git clone https://github.com/kbfinai/rag-service.git
cd rag-service

# Start all services
OPENAI_API_KEY=your-api-key docker compose up

# Or with a .env file
cp .env.example .env
# Edit .env with your configuration
docker compose up
```

### Optional Dependencies

```bash
# Install with local HuggingFace embeddings
uv sync --extra embeddings

# Install with OCR support for scanned PDFs
uv sync --extra ocr

# Install all optional dependencies
uv sync --extra all
```

---

## Development

### Install dev dependencies

```bash
uv sync --extra dev
```

### Run tests

```bash
uv run pytest
```

### Run linting

```bash
uv run ruff check .
uv run ruff format .
```

### Type checking

```bash
uv run mypy src
```

### Setup pre-commit hooks

```bash
uv run pre-commit install
```

### Create a new migration

```bash
uv run alembic revision --autogenerate -m "description"
```

### Apply migrations

```bash
uv run alembic upgrade head
```

---

## API Endpoints

### Authentication

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/v1/auth/register` | No | Register a new user |
| POST | `/api/v1/auth/login` | No | Login and get JWT token |

### Documents

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/v1/documents` | Yes | Create and ingest document |
| POST | `/api/v1/documents/ingest` | Yes | Bulk ingest documents |
| POST | `/api/v1/documents/upload` | Yes | Upload a file |
| POST | `/api/v1/documents/upload/batch` | Yes | Upload multiple files |
| GET | `/api/v1/documents` | Yes | List accessible documents |
| GET | `/api/v1/documents/{id}` | Yes | Get document by ID |
| DELETE | `/api/v1/documents/{id}` | Yes | Delete document |

### Query

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/v1/query` | Yes | Perform RAG query |
| POST | `/api/v1/query/search` | Yes | Semantic search only |

### Permissions

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/v1/permissions/documents/{id}/share` | Yes | Share document with users |
| DELETE | `/api/v1/permissions/documents/{id}/share/{user_id}` | Yes | Revoke user access |
| GET | `/api/v1/permissions/documents/{id}` | Yes | List document permissions |

### Admin

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/api/v1/admin/users` | Admin | List all users |
| POST | `/api/v1/admin/users/{id}/roles` | Admin | Assign role to user |
| DELETE | `/api/v1/admin/users/{id}/roles/{role}` | Admin | Remove role from user |
| GET | `/api/v1/admin/roles` | Admin | List all roles |
| GET | `/api/v1/admin/providers` | Yes | List embedding providers |
| GET | `/api/v1/admin/strategies` | Yes | List chunking strategies |

### Health

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/api/v1/health` | No | Health check |
| GET | `/api/v1/ready` | No | Readiness probe |
| GET | `/api/v1/live` | No | Liveness probe |

OpenAPI documentation available at `http://localhost:8000/docs` (development only)

---

## API Usage Examples

### Register a User

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepassword"}'
```

### Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepassword"}'
```

### Upload a File

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@invoice.pdf" \
  -F "embedding_provider=openai" \
  -F "embedding_model=text-embedding-3-small" \
  -F "chunking_strategy=invoice" \
  -F "is_public=false"
```

### Ingest Documents with Custom Settings

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "documents": [
      {
        "title": "Q4 Compliance Guide",
        "content": "Your document content here...",
        "source": "compliance_q4_2024.pdf",
        "metadata": {"category": "compliance", "year": 2024}
      }
    ],
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small",
    "chunking_strategy": "semantic"
  }'
```

### Query with Provider Selection

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "query": "What are the current compliance requirements for Q4?",
    "top_k": 5,
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small"
  }'
```

### Share a Document

```bash
curl -X POST http://localhost:8000/api/v1/permissions/documents/<doc-id>/share \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "user_ids": ["<user-uuid-1>", "<user-uuid-2>"],
    "permission_level": "read"
  }'
```

### Response Format

```json
{
  "answer": "Based on the retrieved documents...",
  "sources": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "document_title": "Q4 Compliance Guide",
      "chunk_content": "Relevant excerpt from the document...",
      "relevance_score": 0.94
    }
  ],
  "metadata": {
    "chunks_retrieved": 5,
    "model": "gpt-4o-mini",
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small"
  }
}
```

---

## Configuration

### Core Settings

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `APP_NAME` | Application name | `rag-service` |
| `APP_ENV` | Environment (development/production) | `development` |
| `DEBUG` | Enable debug mode | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://...` |
| `SECRET_KEY` | JWT signing key | — |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiration | `30` |
| `CORS_ORIGINS` | Allowed CORS origins | `["http://localhost:3000"]` |

### Embedding Providers

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | — |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | — |
| `AZURE_OPENAI_API_VERSION` | Azure API version | `2024-02-01` |
| `HUGGINGFACE_API_KEY` | HuggingFace API key | — |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `DEFAULT_EMBEDDING_PROVIDER` | Default provider | `openai` |
| `DEFAULT_EMBEDDING_MODEL` | Default model | `text-embedding-3-small` |

### RAG Settings

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `LLM_MODEL` | Model for generation | `gpt-4o-mini` |
| `CHUNK_SIZE` | Default chunk size | `512` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `DEFAULT_CHUNKING_STRATEGY` | Default strategy | `fixed` |
| `TOP_K` | Default retrieved documents | `5` |
| `MIN_SIMILARITY` | Minimum similarity score | `0.5` |

### File Upload

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `MAX_FILE_SIZE_MB` | Maximum file size | `50` |
| `ALLOWED_EXTENSIONS` | Allowed file types | `["pdf","txt","docx",...]` |

### RBAC

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `ENABLE_PUBLIC_DOCUMENTS` | Allow public documents | `false` |
| `DEFAULT_USER_ROLE` | Role for new users | `viewer` |

---

## Project Structure

```
rag-service/
├── src/
│   └── rag_service/
│       ├── main.py              # FastAPI application
│       ├── api/
│       │   ├── dependencies.py  # Dependency injection
│       │   └── routes/
│       │       ├── auth.py      # Authentication endpoints
│       │       ├── query.py     # Query endpoints
│       │       ├── ingest.py    # Document & file upload endpoints
│       │       ├── permissions.py # Document sharing endpoints
│       │       ├── admin.py     # Admin endpoints
│       │       └── health.py    # Health check endpoints
│       ├── core/
│       │   ├── config.py        # Settings management
│       │   ├── security.py      # JWT authentication
│       │   └── logging.py       # Structured logging
│       ├── db/
│       │   ├── postgres.py      # Database connection
│       │   ├── models.py        # SQLAlchemy models (RBAC, providers)
│       │   └── vector.py        # pgvector operations (multi-dimension)
│       ├── models/
│       │   └── schemas.py       # Pydantic schemas
│       └── services/
│           ├── rag.py           # RAG orchestration
│           ├── permissions.py   # RBAC service
│           ├── embeddings/      # Embedding providers
│           │   ├── base.py      # Provider protocol
│           │   ├── factory.py   # Provider factory
│           │   ├── openai_provider.py
│           │   ├── huggingface_provider.py
│           │   └── ollama_provider.py
│           ├── chunking/        # Chunking strategies
│           │   ├── base.py      # Strategy protocol
│           │   ├── factory.py   # Strategy factory
│           │   ├── fixed.py
│           │   ├── recursive.py
│           │   ├── semantic.py
│           │   ├── token_based.py
│           │   ├── markdown.py
│           │   └── invoice.py
│           └── parsers/         # File parsers
│               ├── base.py      # Parser protocol
│               ├── factory.py   # Parser factory
│               ├── pdf.py
│               ├── docx.py
│               ├── txt.py
│               ├── markdown.py
│               ├── csv.py
│               ├── excel.py
│               ├── html.py
│               └── rtf.py
├── alembic/                     # Database migrations
├── tests/                       # Test suite
├── scripts/                     # Utility scripts
├── pyproject.toml               # Project configuration
├── Dockerfile                   # Container build
└── docker-compose.yml           # Local development
```

---

## Integrated Services

This RAG service powers intelligent retrieval for:

- **Financial Analytics Platform** — Market insights and trend analysis
- **Compliance Engine** — Regulatory document search and policy Q&A
- **Report Generator** — Contextual data retrieval for automated reports
- **Client Advisory Tools** — Knowledge-backed recommendations
- **Internal Documentation** — Company-wide financial knowledge base
- **Invoice Processing** — Structured extraction from billing documents

---

## Contributing

We welcome contributions from the kbfinai team. Please follow our development guidelines and submit PRs for review.

1. Create a feature branch from `main`
2. Make your changes
3. Run tests and linting
4. Submit a PR for review

---

## License

Internal use only — kbfinai © 2024

---

<p align="center">
  <strong>Built with purpose for financial intelligence</strong>
</p>
