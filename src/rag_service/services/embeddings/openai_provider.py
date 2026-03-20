"""OpenAI embedding provider."""

from openai import AsyncOpenAI

from rag_service.core.config import Settings, get_settings
from rag_service.services.embeddings.base import BaseEmbeddingProvider, EmbeddingResult


# OpenAI model dimensions
OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider using the OpenAI API."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        config: dict | None = None,
        settings: Settings | None = None,
    ):
        super().__init__(model_name, config)
        self._settings = settings or get_settings()
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def dimension(self) -> int:
        return OPENAI_DIMENSIONS.get(self._model_name, 1536)

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text using OpenAI API."""
        normalized_text = self._normalize_text(text)

        response = await self._client.embeddings.create(
            model=self._model_name,
            input=normalized_text,
        )

        embedding_data = response.data[0]

        return EmbeddingResult(
            embedding=embedding_data.embedding,
            model=self._model_name,
            dimension=len(embedding_data.embedding),
            token_count=response.usage.total_tokens if response.usage else None,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts using OpenAI batch API."""
        if not texts:
            return []

        normalized_texts = [self._normalize_text(t) for t in texts]

        response = await self._client.embeddings.create(
            model=self._model_name,
            input=normalized_texts,
        )

        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)

        results = []
        for embedding_data in sorted_data:
            results.append(
                EmbeddingResult(
                    embedding=embedding_data.embedding,
                    model=self._model_name,
                    dimension=len(embedding_data.embedding),
                    token_count=None,  # Token count is aggregate for batch
                )
            )

        return results


class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Azure OpenAI embedding provider."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        config: dict | None = None,
        settings: Settings | None = None,
    ):
        super().__init__(model_name, config)
        self._settings = settings or get_settings()

        # Azure-specific configuration
        azure_endpoint = self._config.get("azure_endpoint") or getattr(
            self._settings, "azure_openai_endpoint", None
        )
        azure_api_key = self._config.get("azure_api_key") or getattr(
            self._settings, "azure_openai_api_key", None
        )
        api_version = self._config.get("api_version", "2024-02-01")

        if not azure_endpoint or not azure_api_key:
            raise ValueError("Azure endpoint and API key are required")

        from openai import AsyncAzureOpenAI

        self._client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=api_version,
        )

    @property
    def name(self) -> str:
        return "azure"

    @property
    def dimension(self) -> int:
        # Azure uses same models as OpenAI
        return OPENAI_DIMENSIONS.get(self._model_name, 1536)

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding using Azure OpenAI."""
        normalized_text = self._normalize_text(text)

        # Azure uses deployment name instead of model name
        deployment_name = self._config.get("deployment_name", self._model_name)

        response = await self._client.embeddings.create(
            model=deployment_name,
            input=normalized_text,
        )

        embedding_data = response.data[0]

        return EmbeddingResult(
            embedding=embedding_data.embedding,
            model=self._model_name,
            dimension=len(embedding_data.embedding),
            token_count=response.usage.total_tokens if response.usage else None,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts using Azure batch API."""
        if not texts:
            return []

        normalized_texts = [self._normalize_text(t) for t in texts]
        deployment_name = self._config.get("deployment_name", self._model_name)

        response = await self._client.embeddings.create(
            model=deployment_name,
            input=normalized_texts,
        )

        sorted_data = sorted(response.data, key=lambda x: x.index)

        results = []
        for embedding_data in sorted_data:
            results.append(
                EmbeddingResult(
                    embedding=embedding_data.embedding,
                    model=self._model_name,
                    dimension=len(embedding_data.embedding),
                    token_count=None,
                )
            )

        return results
