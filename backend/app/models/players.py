from sqlalchemy import Column, Integer, String, DateTime, UniqueConstraint, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class PlayerSeasonFouls(Base):
    __tablename__ = "player_season_fouls"
    
    id = Column(Integer, primary_key=True, index=True)
    player_name = Column(String(255), nullable=False, index=True)
    team = Column(String(255), nullable=False, index=True)
    league = Column(String(100), nullable=False, index=True)
    season = Column(String(20), nullable=False, index=True)
    position = Column(String(10), nullable=True)
    appearances = Column(Integer, default=0)
    fouls = Column(Integer, default=0)
    fouls_drawn = Column(Integer, default=0)
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)
    minutes = Column(Integer, default=0)
    source = Column(String(50), default="fbref")
    scraped_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('player_name', 'team', 'season', 'league', name='uq_player_team_season_league'),
        Index('idx_player_fouls', 'player_name', 'fouls'),
        Index('idx_team_season', 'team', 'season'),
        Index('idx_league_season', 'league', 'season'),
    )
    
    @property
    def fouls_per_game(self) -> float:
        """Calculate fouls per game."""
        if self.appearances == 0:
            return 0.0
        return round(self.fouls / self.appearances, 2)
    
    @property
    def cards_per_game(self) -> float:
        """Calculate total cards per game."""
        if self.appearances == 0:
            return 0.0
        total_cards = self.yellow_cards + self.red_cards
        return round(total_cards / self.appearances, 2)
    
    @property
    def minutes_per_foul(self) -> float:
        """Calculate minutes per foul."""
        if self.fouls == 0:
            return 0.0
        return round(self.minutes / self.fouls, 1)