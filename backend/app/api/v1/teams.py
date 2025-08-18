from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc
from app.core.db import get_db
from src.database.models import PlayerSeasonFouls

router = APIRouter(prefix="/teams", tags=["teams"])

@router.get("/fouls-summary")
def team_fouls_summary(
    league: str = Query(...),
    season: str = Query(...),
    db: Session = Depends(get_db),
):
    """Get fouls summary by team for a given league and season"""
    stmt = (
        select(
            PlayerSeasonFouls.team,
            func.sum(PlayerSeasonFouls.fouls).label("total_fouls"),
            func.sum(PlayerSeasonFouls.fouls_drawn).label("total_fouls_drawn"),
            func.sum(PlayerSeasonFouls.yellow_cards).label("total_yellow_cards"),
            func.sum(PlayerSeasonFouls.red_cards).label("total_red_cards"),
            func.count(PlayerSeasonFouls.player).label("player_count")
        )
        .where(
            PlayerSeasonFouls.league == league,
            PlayerSeasonFouls.season == season,
        )
        .group_by(PlayerSeasonFouls.team)
        .order_by(desc(func.sum(PlayerSeasonFouls.fouls)))
    )
    rows = db.execute(stmt).all()
    return [{
        "team": r[0],
        "total_fouls": float(r[1] or 0),
        "total_fouls_drawn": float(r[2] or 0),
        "total_yellow_cards": float(r[3] or 0),
        "total_red_cards": float(r[4] or 0),
        "player_count": int(r[5])
    } for r in rows]