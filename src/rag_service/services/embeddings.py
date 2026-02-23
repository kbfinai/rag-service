from openai import AsyncOpenAI

from rag_service.core.config import Settings, get_settings
from rag_service.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.embedding_model

    async def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # OpenAI supports batch embeddings
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        # Sort by index to maintain order
        sorted_embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_embeddings]

    def chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[str]:
        """Split text into overlapping chunks."""
        chunk_size = chunk_size or self.settings.chunk_size
        overlap = overlap or self.settings.chunk_overlap

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence or word boundary
            if end < len(text):
                # Look for sentence boundary
                for sep in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + 1
                        break
                else:
                    # Fall back to word boundary
                    last_space = text.rfind(" ", start, end)
                    if last_space > start:
                        end = last_space

            chunks.append(text[start:end].strip())
            start = end - overlap

        return [c for c in chunks if c]  # Remove empty chunks
