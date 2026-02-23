import pytest
from fastapi import status


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_liveness_check(self, client):
        """Test liveness endpoint returns alive status."""
        response = client.get("/api/v1/live")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "alive"}

    def test_readiness_check(self, client):
        """Test readiness endpoint."""
        response = client.get("/api/v1/ready")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data

    def test_health_check(self, client):
        """Test health endpoint returns full status."""
        response = client.get("/api/v1/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "database" in data
        assert "timestamp" in data


@pytest.mark.asyncio
class TestHealthEndpointsAsync:
    """Async health check tests."""

    async def test_liveness_async(self, async_client):
        """Test liveness endpoint async."""
        response = await async_client.get("/api/v1/live")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "alive"}
