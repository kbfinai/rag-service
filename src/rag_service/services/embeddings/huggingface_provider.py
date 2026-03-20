"""HuggingFace embedding provider using sentence-transformers."""

import asyncio
from functools import lru_cache

from rag_service.core.config import Settings, get_settings
from rag_service.services.embeddings.base import BaseEmbeddingProvider, EmbeddingResult


# HuggingFace model dimensions
HUGGINGFACE_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
    "e5-small-v2": 384,
    "e5-base-v2": 768,
    "e5-large-v2": 1024,
    "bge-small-en-v1.5": 384,
    "bge-base-en-v1.5": 768,
    "bge-large-en-v1.5": 1024,
    "gte-small": 384,
    "gte-base": 768,
    "gte-large": 1024,
}


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    """Cache loaded models to avoid reloading."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for HuggingFace embeddings. "
            "Install it with: pip install sentence-transformers"
        )


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace embedding provider using sentence-transformers (local models)."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        config: dict | None = None,
        settings: Settings | None = None,
    ):
        super().__init__(model_name, config)
        self._settings = settings or get_settings()
        self._model = None
        self._executor = None

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def dimension(self) -> int:
        return HUGGINGFACE_DIMENSIONS.get(self._model_name, 768)

    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            self._model = _get_model(self._model_name)
        return self._model

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding using local HuggingFace model."""
        normalized_text = self._normalize_text(text)

        # Run in thread pool since sentence-transformers is synchronous
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._get_model().encode(normalized_text, normalize_embeddings=True),
        )

        return EmbeddingResult(
            embedding=embedding.tolist(),
            model=self._model_name,
            dimension=len(embedding),
            token_count=None,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts using batch encoding."""
        if not texts:
            return []

        normalized_texts = [self._normalize_text(t) for t in texts]

        # Run in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._get_model().encode(
                normalized_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
        )

        results = []
        for embedding in embeddings:
            results.append(
                EmbeddingResult(
                    embedding=embedding.tolist(),
                    model=self._model_name,
                    dimension=len(embedding),
                    token_count=None,
                )
            )

        return results


class HuggingFaceAPIEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace embedding provider using the Inference API (remote)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: dict | None = None,
        settings: Settings | None = None,
    ):
        super().__init__(model_name, config)
        self._settings = settings or get_settings()
        self._api_key = self._config.get("api_key") or getattr(
            self._settings, "huggingface_api_key", None
        )

        if not self._api_key:
            raise ValueError("HuggingFace API key is required for API provider")

        import httpx

        self._client = httpx.AsyncClient(
            base_url="https://api-inference.huggingface.co",
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=60.0,
        )

    @property
    def name(self) -> str:
        return "huggingface_api"

    @property
    def dimension(self) -> int:
        # Extract model short name for lookup
        short_name = self._model_name.split("/")[-1]
        return HUGGINGFACE_DIMENSIONS.get(short_name, 768)

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding using HuggingFace Inference API."""
        normalized_text = self._normalize_text(text)

        response = await self._client.post(
            f"/pipeline/feature-extraction/{self._model_name}",
            json={"inputs": normalized_text, "options": {"wait_for_model": True}},
        )
        response.raise_for_status()

        embedding = response.json()

        # Handle nested response (sentence-transformers returns nested array)
        if isinstance(embedding, list) and len(embedding) > 0:
            if isinstance(embedding[0], list):
                embedding = embedding[0]

        return EmbeddingResult(
            embedding=embedding,
            model=self._model_name,
            dimension=len(embedding),
            token_count=None,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        normalized_texts = [self._normalize_text(t) for t in texts]

        response = await self._client.post(
            f"/pipeline/feature-extraction/{self._model_name}",
            json={"inputs": normalized_texts, "options": {"wait_for_model": True}},
        )
        response.raise_for_status()

        embeddings = response.json()

        results = []
        for embedding in embeddings:
            # Handle nested response
            if isinstance(embedding, list) and len(embedding) > 0:
                if isinstance(embedding[0], list):
                    embedding = embedding[0]

            results.append(
                EmbeddingResult(
                    embedding=embedding,
                    model=self._model_name,
                    dimension=len(embedding),
                    token_count=None,
                )
            )

        return results

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
