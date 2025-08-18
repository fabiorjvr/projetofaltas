from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc
from app.core.db import get_db
from src.database.models import PlayerSeasonFouls  # ajuste o caminho
from app.schemas.players import PlayerFoulOut

router = APIRouter(prefix="/players", tags=["players"])

@router.get("", response_model=list[PlayerFoulOut])
def list_players(
    league: str | None = Query(None),
    season: str | None = Query(None),
    team: str | None = Query(None),
    min_fouls: float | None = Query(None),
    order: str = Query("fouls_desc"),  # fouls_desc | fouls_asc | name
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(PlayerSeasonFouls)
    if league: stmt = stmt.where(PlayerSeasonFouls.league == league)
    if season: stmt = stmt.where(PlayerSeasonFouls.season == season)
    if team:   stmt = stmt.where(PlayerSeasonFouls.team == team)
    if min_fouls is not None:
        stmt = stmt.where(PlayerSeasonFouls.fouls >= min_fouls)

    if order == "fouls_asc":
        stmt = stmt.order_by(PlayerSeasonFouls.fouls.asc())
    elif order == "name":
        stmt = stmt.order_by(PlayerSeasonFouls.player.asc())
    else:
        stmt = stmt.order_by(PlayerSeasonFouls.fouls.desc())

    stmt = stmt.limit(limit).offset(offset)
    rows = db.execute(stmt).scalars().all()
    return rows