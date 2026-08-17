from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    """Unauthenticated health check."""
    return {"status": "ok"}
