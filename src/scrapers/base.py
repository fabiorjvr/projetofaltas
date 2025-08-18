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
    """Garante que o DataFrame tenha todas as colunas necessárias na ordem correta"""
    required = [
        'league','season','player','team','position',
        'appearances','fouls','fouls_drawn','yellow_cards','red_cards','minutes',
        'source'
    ]
    out = df.copy()
    for col in required:
        if col not in out.columns:
            out[col] = None

    # preenchimentos mínimos
    for num in ['appearances','fouls','fouls_drawn','yellow_cards','red_cards','minutes']:
        out[num] = pd.to_numeric(out[num], errors='coerce').fillna(0.0)

    for txt in ['league','season','player','team','position','source']:
        out[txt] = out[txt].astype(str).str.strip().fillna('')

    # filtrar linhas inválidas (sem player/team)
    out = out[(out['player']!='') & (out['team']!='')]

    return out[required]