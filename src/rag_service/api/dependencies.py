from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.core.config import Settings, get_settings
from rag_service.core.security import TokenData, get_current_user
from rag_service.db.postgres import get_session
from rag_service.db.vector import VectorStore
from rag_service.services.embeddings import EmbeddingService
from rag_service.services.rag import RAGService

# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
CurrentUserDep = Annotated[TokenData, Depends(get_current_user)]


async def get_vector_store(
    session: SessionDep,
    settings: SettingsDep,
) -> AsyncGenerator[VectorStore, None]:
    yield VectorStore(session, settings)


async def get_embedding_service(
    settings: SettingsDep,
) -> AsyncGenerator[EmbeddingService, None]:
    yield EmbeddingService(settings)


async def get_rag_service(
    session: SessionDep,
    settings: SettingsDep,
) -> AsyncGenerator[RAGService, None]:
    vector_store = VectorStore(session, settings)
    embedding_service = EmbeddingService(settings)
    yield RAGService(session, vector_store, embedding_service, settings)


VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
