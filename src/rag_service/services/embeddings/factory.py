"""Factory for creating embedding providers."""

from rag_service.core.config import Settings, get_settings
from rag_service.services.embeddings.base import EmbeddingProvider
from rag_service.services.embeddings.openai_provider import (
    AzureOpenAIEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from rag_service.services.embeddings.huggingface_provider import (
    HuggingFaceEmbeddingProvider,
    HuggingFaceAPIEmbeddingProvider,
)
from rag_service.services.embeddings.ollama_provider import OllamaEmbeddingProvider


# Model dimension registry
MODEL_DIMENSIONS = {
    # OpenAI
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-ada-002": 1536,
    # Azure (same as OpenAI)
    "azure/text-embedding-3-small": 1536,
    "azure/text-embedding-3-large": 3072,
    "azure/text-embedding-ada-002": 1536,
    # HuggingFace
    "huggingface/all-MiniLM-L6-v2": 384,
    "huggingface/all-MiniLM-L12-v2": 384,
    "huggingface/all-mpnet-base-v2": 768,
    "huggingface/e5-small-v2": 384,
    "huggingface/e5-base-v2": 768,
    "huggingface/e5-large-v2": 1024,
    "huggingface/bge-small-en-v1.5": 384,
    "huggingface/bge-base-en-v1.5": 768,
    "huggingface/bge-large-en-v1.5": 1024,
    # Ollama
    "ollama/nomic-embed-text": 768,
    "ollama/mxbai-embed-large": 1024,
    "ollama/all-minilm": 384,
}


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""

    _providers = {
        "openai": OpenAIEmbeddingProvider,
        "azure": AzureOpenAIEmbeddingProvider,
        "huggingface": HuggingFaceEmbeddingProvider,
        "huggingface_api": HuggingFaceAPIEmbeddingProvider,
        "ollama": OllamaEmbeddingProvider,
    }

    @classmethod
    def create(
        cls,
        provider_name: str,
        model_name: str,
        config: dict | None = None,
        settings: Settings | None = None,
    ) -> EmbeddingProvider:
        """
        Create an embedding provider instance.

        Args:
            provider_name: Provider name (openai, azure, huggingface, ollama)
            model_name: Model name (e.g., text-embedding-3-small)
            config: Optional provider-specific configuration
            settings: Optional settings object

        Returns:
            EmbeddingProvider instance

        Raises:
            ValueError: If provider name is not supported
        """
        settings = settings or get_settings()

        if provider_name not in cls._providers:
            raise ValueError(
                f"Unknown embedding provider: {provider_name}. "
                f"Supported providers: {list(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_name]
        return provider_class(model_name=model_name, config=config, settings=settings)

    @classmethod
    def get_dimension(cls, provider_name: str, model_name: str) -> int:
        """
        Get the embedding dimension for a provider/model combination.

        Args:
            provider_name: Provider name
            model_name: Model name

        Returns:
            Embedding dimension
        """
        key = f"{provider_name}/{model_name}"
        if key in MODEL_DIMENSIONS:
            return MODEL_DIMENSIONS[key]

        # Fallback to provider-specific lookup
        provider = cls.create(provider_name, model_name)
        return provider.dimension

    @classmethod
    def list_providers(cls) -> list[str]:
        """List available provider names."""
        return list(cls._providers.keys())

    @classmethod
    def list_models(cls, provider_name: str | None = None) -> list[dict]:
        """
        List available models.

        Args:
            provider_name: Optional filter by provider

        Returns:
            List of model info dicts with name, provider, and dimension
        """
        models = []
        for key, dimension in MODEL_DIMENSIONS.items():
            provider, model = key.split("/", 1)
            if provider_name is None or provider == provider_name:
                models.append({
                    "provider": provider,
                    "model": model,
                    "dimension": dimension,
                    "full_name": key,
                })
        return models

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: type,
    ) -> None:
        """
        Register a custom embedding provider.

        Args:
            name: Provider name
            provider_class: Provider class implementing EmbeddingProvider protocol
        """
        cls._providers[name] = provider_class
