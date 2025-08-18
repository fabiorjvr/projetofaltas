from sqlalchemy import Column, Integer, String, Float, DateTime, UniqueConstraint, Text
from sqlalchemy.sql import func
from .db import Base

class PlayerSeasonFouls(Base):
    """Modelo para armazenar dados de faltas por jogador por temporada"""
    __tablename__ = "player_season_fouls"
    
    # Chave primária
    id = Column(Integer, primary_key=True, index=True)
    
    # Dados da liga e temporada
    league = Column(String, nullable=False, index=True)
    season = Column(String, nullable=False, index=True)
    
    # Dados do jogador
    player = Column(String, nullable=False, index=True)
    team = Column(String, nullable=False, index=True)
    position = Column(String, nullable=True)
    
    # Estatísticas de jogo
    appearances = Column(Float, nullable=True)  # Jogos disputados
    fouls = Column(Float, nullable=True)        # Faltas cometidas
    fouls_drawn = Column(Float, nullable=True)  # Faltas sofridas
    yellow_cards = Column(Float, nullable=True) # Cartões amarelos
    red_cards = Column(Float, nullable=True)    # Cartões vermelhos
    minutes = Column(Float, nullable=True)      # Minutos jogados
    
    # Metadados
    source = Column(Text, nullable=False)     # Fonte dos dados (ex: "FBref (Misc)")
    scraped_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Constraint única: um jogador por time por liga por temporada
    __table_args__ = (
        UniqueConstraint('league', 'season', 'player', 'team', name='uq_player_season_team'),
    )
    
    def __repr__(self):
        return f"<PlayerSeasonFouls(player='{self.player}', team='{self.team}', league='{self.league}', season='{self.season}', fouls={self.fouls})>"