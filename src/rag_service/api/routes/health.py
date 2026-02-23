from datetime import datetime, timezone

from fastapi import APIRouter, status
from sqlalchemy import text

from rag_service.api.dependencies import SessionDep, SettingsDep
from rag_service.models.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
async def health_check(
    session: SessionDep,
    settings: SettingsDep,
) -> HealthResponse:
    """Health check endpoint for load balancers and monitoring."""
    # Check database connection
    try:
        await session.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"

    return HealthResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        version="0.1.0",
        database=db_status,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check(session: SessionDep) -> dict[str, str]:
    """Readiness probe for Kubernetes."""
    try:
        await session.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        return {"status": "not ready"}


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check() -> dict[str, str]:
    """Liveness probe for Kubernetes."""
    return {"status": "alive"}
