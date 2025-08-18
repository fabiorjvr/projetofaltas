from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc
from app.core.db import get_db
from src.database.models import PlayerSeasonFouls

router = APIRouter(prefix="/stats", tags=["stats"])

@router.get("/top-fouls")
def top_fouls(
    league: str = Query(...),
    season: str = Query(...),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    stmt = (
        select(
            PlayerSeasonFouls.player,
            PlayerSeasonFouls.team,
            PlayerSeasonFouls.fouls
        )
        .where(
            PlayerSeasonFouls.league == league,
            PlayerSeasonFouls.season == season,
        )
        .order_by(desc(PlayerSeasonFouls.fouls))
        .limit(limit)
    )
    rows = db.execute(stmt).all()
    return [{"player": r[0], "team": r[1], "fouls": float(r[2])} for r in rows]