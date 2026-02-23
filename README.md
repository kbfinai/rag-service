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
| **Semantic Search** | Vector-based retrieval for accurate, context-aware document matching |
| **Multi-Tool Integration** | Seamless API access for all kbfinai services |
| **Real-Time Retrieval** | Low-latency responses for production workloads |
| **Scalable Architecture** | Designed to handle enterprise-grade financial queries |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Package Manager | uv |
| API Framework | FastAPI + Uvicorn |
| Database | PostgreSQL 16 + pgvector |
| ORM | SQLAlchemy 2.0 (async) |
| Embeddings | OpenAI text-embedding-3-small |
| Auth | JWT (python-jose) |
| Migrations | Alembic |
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
                 │  │  Query Engine    │  │
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
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Financial Knowledge  │
                 │   • Documents          │
                 │   • Policies           │
                 │   • Market Data        │
                 │   • Regulatory Info    │
                 └────────────────────────┘
```

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker & Docker Compose (for local development)
- OpenAI API key

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
# Edit .env and add your OPENAI_API_KEY

# Start PostgreSQL with pgvector
docker compose up db -d

# Wait for database to be ready, then run migrations
uv run alembic upgrade head

# Start the service
uv run python -m rag_service.main
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

---

## Development

### Install dev dependencies

```bash
uv sync --dev
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

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/v1/auth/register` | No | Register a new user |
| POST | `/api/v1/auth/login` | No | Login and get JWT token |
| POST | `/api/v1/query` | Yes | Perform RAG query |
| POST | `/api/v1/query/search` | Yes | Semantic search only |
| POST | `/api/v1/documents` | Yes | Create and ingest document |
| POST | `/api/v1/documents/ingest` | Yes | Bulk ingest documents |
| GET | `/api/v1/documents` | Yes | List all documents |
| GET | `/api/v1/documents/{id}` | Yes | Get document by ID |
| DELETE | `/api/v1/documents/{id}` | Yes | Delete document |
| GET | `/api/v1/health` | No | Health check |
| GET | `/api/v1/ready` | No | Readiness probe |
| GET | `/api/v1/live` | No | Liveness probe |

OpenAPI documentation available at `http://localhost:8000/docs` (development only)

---

## API Usage

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

### Ingest Documents

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
    ]
  }'
```

### Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "query": "What are the current compliance requirements for Q4?",
    "top_k": 5
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
    "model": "gpt-4o-mini"
  }
}
```

---

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `APP_NAME` | Application name | `rag-service` |
| `APP_ENV` | Environment (development/production) | `development` |
| `DEBUG` | Enable debug mode | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://...` |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `EMBEDDING_MODEL` | Model for embeddings | `text-embedding-3-small` |
| `LLM_MODEL` | Model for generation | `gpt-4o-mini` |
| `CHUNK_SIZE` | Document chunk size | `512` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `TOP_K` | Default retrieved documents | `5` |
| `SECRET_KEY` | JWT signing key | — |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiration | `30` |
| `CORS_ORIGINS` | Allowed CORS origins | `["http://localhost:3000"]` |

---

## Project Structure

```
rag-service/
├── src/
│   └── rag_service/
│       ├── main.py              # FastAPI application
│       ├── api/
│       │   ├── dependencies.py  # Dependency injection
│       │   └── routes/          # API endpoints
│       ├── core/
│       │   ├── config.py        # Settings management
│       │   ├── security.py      # JWT authentication
│       │   └── logging.py       # Structured logging
│       ├── db/
│       │   ├── postgres.py      # Database connection
│       │   ├── models.py        # SQLAlchemy models
│       │   └── vector.py        # pgvector operations
│       ├── models/
│       │   └── schemas.py       # Pydantic schemas
│       └── services/
│           ├── embeddings.py    # Embedding generation
│           └── rag.py           # RAG orchestration
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
