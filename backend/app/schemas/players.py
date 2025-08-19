from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import re

class PositionEnum(str, Enum):
    """Player positions enum."""
    GK = "GK"  # Goalkeeper
    DF = "DF"  # Defender
    MF = "MF"  # Midfielder
    FW = "FW"  # Forward
    DFMF = "DF,MF"  # Defender/Midfielder
    MFFW = "MF,FW"  # Midfielder/Forward
    DFFW = "DF,FW"  # Defender/Forward

class LeagueEnum(str, Enum):
    """Supported leagues enum."""
    PREMIER_LEAGUE = "Premier League"
    LA_LIGA = "La Liga"
    BUNDESLIGA = "Bundesliga"
    SERIE_A = "Serie A"
    LIGUE_1 = "Ligue 1"
    CHAMPIONS_LEAGUE = "Champions League"
    EUROPA_LEAGUE = "Europa League"
    BRASILEIRAO = "Brasileirão"

class PlayerBase(BaseModel):
    """Base player schema with common validations."""
    player_name: str = Field(
        ..., 
        min_length=2, 
        max_length=255,
        description="Player full name"
    )
    team: str = Field(
        ..., 
        min_length=2, 
        max_length=255,
        description="Team name"
    )
    league: str = Field(
        ..., 
        min_length=2, 
        max_length=100,
        description="League name"
    )
    season: str = Field(
        ..., 
        regex=r"^\d{4}-\d{2}$",
        description="Season in format YYYY-YY (e.g., 2023-24)"
    )
    position: Optional[PositionEnum] = Field(
        None,
        description="Player position"
    )
    
    @validator('player_name')
    def validate_player_name(cls, v):
        """Validate player name format."""
        if not v or v.isspace():
            raise ValueError('Player name cannot be empty or whitespace')
        
        # Remove extra whitespace and normalize
        v = ' '.join(v.split())
        
        # Check for valid characters (letters, spaces, hyphens, apostrophes)
        if not re.match(r"^[a-zA-ZÀ-ÿ\s\-']+$", v):
            raise ValueError('Player name contains invalid characters')
        
        return v.title()  # Capitalize properly
    
    @validator('team')
    def validate_team_name(cls, v):
        """Validate team name format."""
        if not v or v.isspace():
            raise ValueError('Team name cannot be empty or whitespace')
        
        # Remove extra whitespace and normalize
        v = ' '.join(v.split())
        
        return v
    
    @validator('season')
    def validate_season_format(cls, v):
        """Validate season format and logic."""
        if not re.match(r"^\d{4}-\d{2}$", v):
            raise ValueError('Season must be in format YYYY-YY (e.g., 2023-24)')
        
        year_start, year_end = v.split('-')
        year_start_int = int(year_start)
        year_end_int = int(f"20{year_end}")
        
        # Validate year sequence
        if year_end_int != year_start_int + 1:
            raise ValueError('Season end year must be consecutive to start year')
        
        # Validate reasonable year range
        current_year = datetime.now().year
        if year_start_int < 1990 or year_start_int > current_year + 1:
            raise ValueError(f'Season year must be between 1990 and {current_year + 1}')
        
        return v

class PlayerStats(BaseModel):
    """Player statistics with robust validations."""
    appearances: int = Field(
        0, 
        ge=0, 
        le=100,
        description="Number of appearances"
    )
    fouls: int = Field(
        0, 
        ge=0, 
        le=500,
        description="Total fouls committed"
    )
    fouls_drawn: int = Field(
        0, 
        ge=0, 
        le=500,
        description="Total fouls drawn"
    )
    yellow_cards: int = Field(
        0, 
        ge=0, 
        le=50,
        description="Yellow cards received"
    )
    red_cards: int = Field(
        0, 
        ge=0, 
        le=10,
        description="Red cards received"
    )
    minutes: int = Field(
        0, 
        ge=0, 
        le=10000,
        description="Total minutes played"
    )
    
    @validator('fouls')
    def validate_fouls(cls, v, values):
        """Validate fouls against appearances."""
        appearances = values.get('appearances', 0)
        if appearances > 0 and v > appearances * 20:  # Max 20 fouls per game
            raise ValueError('Fouls per game seems unrealistic (max 20 per game)')
        return v
    
    @validator('yellow_cards')
    def validate_yellow_cards(cls, v, values):
        """Validate yellow cards against appearances."""
        appearances = values.get('appearances', 0)
        if appearances > 0 and v > appearances * 3:  # Max 3 yellows per game
            raise ValueError('Yellow cards per game seems unrealistic (max 3 per game)')
        return v
    
    @validator('red_cards')
    def validate_red_cards(cls, v, values):
        """Validate red cards against appearances."""
        appearances = values.get('appearances', 0)
        if v > appearances:  # Can't have more reds than games
            raise ValueError('Red cards cannot exceed appearances')
        return v
    
    @validator('minutes')
    def validate_minutes(cls, v, values):
        """Validate minutes against appearances."""
        appearances = values.get('appearances', 0)
        if appearances > 0 and v > appearances * 120:  # Max 120 min per game (with extra time)
            raise ValueError('Minutes per game seems unrealistic (max 120 per game)')
        return v
    
    @root_validator
    def validate_stats_consistency(cls, values):
        """Validate overall statistics consistency."""
        appearances = values.get('appearances', 0)
        minutes = values.get('minutes', 0)
        fouls = values.get('fouls', 0)
        yellow_cards = values.get('yellow_cards', 0)
        red_cards = values.get('red_cards', 0)
        
        # If no appearances, other stats should be zero
        if appearances == 0:
            if any([minutes > 0, fouls > 0, yellow_cards > 0, red_cards > 0]):
                raise ValueError('Player with 0 appearances cannot have other statistics')
        
        # If minutes > 0, appearances should be > 0
        if minutes > 0 and appearances == 0:
            raise ValueError('Player cannot have minutes without appearances')
        
        # Validate cards consistency
        total_cards = yellow_cards + red_cards
        if total_cards > fouls and fouls > 0:
            # Allow some flexibility as not all cards come from fouls
            if total_cards > fouls * 2:
                raise ValueError('Total cards significantly exceeds fouls committed')
        
        return values

class PlayerSeasonFoulsBase(PlayerBase, PlayerStats):
    pass

class PlayerSeasonFoulsCreate(PlayerSeasonFoulsBase):
    """Schema for creating player season fouls data."""
    source: str = Field(
        "fbref",
        regex=r"^[a-zA-Z0-9_-]+$",
        description="Data source identifier"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "player_name": "Lionel Messi",
                "team": "Paris Saint-Germain",
                "league": "Ligue 1",
                "season": "2023-24",
                "position": "FW",
                "appearances": 25,
                "fouls": 15,
                "fouls_drawn": 45,
                "yellow_cards": 3,
                "red_cards": 0,
                "minutes": 2100,
                "source": "fbref"
            }
        }

class PlayerSeasonFoulsUpdate(BaseModel):
    """Schema for updating player season fouls data."""
    player_name: Optional[str] = Field(None, min_length=2, max_length=255)
    team: Optional[str] = Field(None, min_length=2, max_length=255)
    position: Optional[PositionEnum] = None
    appearances: Optional[int] = Field(None, ge=0, le=100)
    fouls: Optional[int] = Field(None, ge=0, le=500)
    fouls_drawn: Optional[int] = Field(None, ge=0, le=500)
    yellow_cards: Optional[int] = Field(None, ge=0, le=50)
    red_cards: Optional[int] = Field(None, ge=0, le=10)
    minutes: Optional[int] = Field(None, ge=0, le=10000)

class PlayerSeasonFoulsInDB(PlayerSeasonFoulsBase):
    """Schema for player season fouls data from database."""
    id: int
    source: str
    scraped_at: datetime
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class PlayerSeasonFouls(PlayerSeasonFoulsInDB):
    """Public schema for player season fouls data."""
    pass

class PlayerSearchFilters(BaseModel):
    """Schema for player search filters."""
    player_name: Optional[str] = Field(None, min_length=1, max_length=255)
    team: Optional[str] = Field(None, min_length=1, max_length=255)
    league: Optional[str] = Field(None, min_length=1, max_length=100)
    season: Optional[str] = Field(None, regex=r"^\d{4}-\d{2}$")
    position: Optional[PositionEnum] = None
    min_appearances: Optional[int] = Field(None, ge=0, le=100)
    max_appearances: Optional[int] = Field(None, ge=0, le=100)
    min_fouls: Optional[int] = Field(None, ge=0)
    max_fouls: Optional[int] = Field(None, ge=0)
    min_cards: Optional[int] = Field(None, ge=0)
    max_cards: Optional[int] = Field(None, ge=0)
    
    @root_validator
    def validate_ranges(cls, values):
        """Validate min/max ranges."""
        # Validate appearances range
        min_app = values.get('min_appearances')
        max_app = values.get('max_appearances')
        if min_app is not None and max_app is not None and min_app > max_app:
            raise ValueError('min_appearances cannot be greater than max_appearances')
        
        # Validate fouls range
        min_fouls = values.get('min_fouls')
        max_fouls = values.get('max_fouls')
        if min_fouls is not None and max_fouls is not None and min_fouls > max_fouls:
            raise ValueError('min_fouls cannot be greater than max_fouls')
        
        # Validate cards range
        min_cards = values.get('min_cards')
        max_cards = values.get('max_cards')
        if min_cards is not None and max_cards is not None and min_cards > max_cards:
            raise ValueError('min_cards cannot be greater than max_cards')
        
        return values

class BulkPlayerImport(BaseModel):
    """Schema for bulk player data import."""
    players: List[PlayerSeasonFoulsCreate] = Field(
        ..., 
        min_items=1, 
        max_items=1000,
        description="List of players to import"
    )
    overwrite_existing: bool = Field(
        False,
        description="Whether to overwrite existing records"
    )
    validate_only: bool = Field(
        False,
        description="Only validate data without importing"
    )
    
    @validator('players')
    def validate_unique_players(cls, v):
        """Ensure no duplicate players in the same import."""
        seen = set()
        for player in v:
            key = (player.player_name, player.team, player.season, player.league)
            if key in seen:
                raise ValueError(f'Duplicate player found: {player.player_name} ({player.team}, {player.season})')
            seen.add(key)
        return v