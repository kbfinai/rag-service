"""Ollama embedding provider for local LLM embeddings."""

import httpx

from rag_service.core.config import Settings, get_settings
from rag_service.services.embeddings.base import BaseEmbeddingProvider, EmbeddingResult


# Ollama model dimensions
OLLAMA_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-m3": 1024,
    "bge-large": 1024,
}


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider for local models."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        config: dict | None = None,
        settings: Settings | None = None,
    ):
        super().__init__(model_name, config)
        self._settings = settings or get_settings()

        # Ollama base URL
        self._base_url = self._config.get("base_url") or getattr(
            self._settings, "ollama_base_url", "http://localhost:11434"
        )

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=120.0,  # Longer timeout for local models
        )

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def dimension(self) -> int:
        return OLLAMA_DIMENSIONS.get(self._model_name, 768)

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding using Ollama API."""
        normalized_text = self._normalize_text(text)

        response = await self._client.post(
            "/api/embeddings",
            json={
                "model": self._model_name,
                "prompt": normalized_text,
            },
        )
        response.raise_for_status()

        data = response.json()
        embedding = data.get("embedding", [])

        return EmbeddingResult(
            embedding=embedding,
            model=self._model_name,
            dimension=len(embedding),
            token_count=None,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.

        Ollama doesn't have native batch support, so we process sequentially.
        For better performance with large batches, consider using asyncio.gather
        with rate limiting.
        """
        if not texts:
            return []

        results = []
        for text in texts:
            result = await self.embed(text)
            results.append(result)

        return results

    async def check_model_available(self) -> bool:
        """Check if the model is available in Ollama."""
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()

            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            return self._model_name in models
        except Exception:
            return False

    async def pull_model(self) -> bool:
        """Pull the model if not available."""
        try:
            response = await self._client.post(
                "/api/pull",
                json={"name": self._model_name},
                timeout=600.0,  # 10 minutes for model download
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
