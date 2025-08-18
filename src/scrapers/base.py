from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class BaseScraper(ABC):
    """Classe base para scrapers de dados de futebol"""
    
    @abstractmethod
    def fetch(self, league: str, season: str) -> pd.DataFrame:
        """Método abstrato para buscar dados de uma liga e temporada específica"""
        pass

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que o DataFrame tenha as colunas padrão na ordem correta"""
    required_columns = [
        'league', 'season', 'player', 'team', 'position', 
        'appearances', 'fouls', 'fouls_drawn', 'yellow_cards', 
        'red_cards', 'minutes', 'source'
    ]
    
    # Adiciona colunas faltantes com valores None
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Reordena as colunas
    df = df[required_columns]
    
    return df