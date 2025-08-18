from fastapi import FastAPI
from app.api.v1.players import router as players_router
from app.api.v1.stats import router as stats_router
from app.api.v1.teams import router as teams_router
from app.core.config import settings

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    debug=settings.debug
)

# Include API routers
app.include_router(players_router, prefix="/api/v1")
app.include_router(stats_router, prefix="/api/v1")
app.include_router(teams_router, prefix="/api/v1")

@app.get("/")
def root():
    return {
        "message": "Football Fouls Analytics API",
        "version": settings.api_version,
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}