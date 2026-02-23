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

## Quick Start

```bash
# Clone the repository
git clone https://github.com/kbfinai/rag-service.git
cd rag-service

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Run the service
python -m rag_service
```

---

## API Usage

### Query Endpoint

```python
import requests

response = requests.post(
    "https://api.kbfinai.com/rag/query",
    json={
        "query": "What are the current compliance requirements for Q4?",
        "top_k": 5,
        "filters": {"category": "compliance"}
    },
    headers={"Authorization": "Bearer <token>"}
)

print(response.json())
```

### Response Format

```json
{
  "answer": "Based on the retrieved documents...",
  "sources": [
    {"document": "compliance_q4_2024.pdf", "relevance": 0.94},
    {"document": "regulatory_updates.pdf", "relevance": 0.87}
  ],
  "metadata": {
    "retrieval_time_ms": 45,
    "generation_time_ms": 320
  }
}
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

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `VECTOR_DB_URL` | Vector database connection string | `localhost:6333` |
| `LLM_API_KEY` | API key for LLM provider | — |
| `EMBEDDING_MODEL` | Model for document embeddings | `text-embedding-3-small` |
| `CHUNK_SIZE` | Document chunk size for indexing | `512` |
| `TOP_K` | Default number of retrieved documents | `5` |

---

## Contributing

We welcome contributions from the kbfinai team. Please follow our development guidelines and submit PRs for review.

---

## License

Internal use only — kbfinai © 2024

---

<p align="center">
  <strong>Built with purpose for financial intelligence</strong>
</p>