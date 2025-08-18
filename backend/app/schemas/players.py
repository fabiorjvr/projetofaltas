from pydantic import BaseModel
from typing import Optional

class PlayerFoulOut(BaseModel):
    league: str
    season: str
    player: str
    team: str
    position: Optional[str] = None
    appearances: float
    fouls: float
    fouls_drawn: float
    yellow_cards: float
    red_cards: float
    minutes: float

    class Config:
        from_attributes = True