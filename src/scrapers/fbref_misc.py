import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from .base import BaseScraper, ensure_columns

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FBRefMiscScraper(BaseScraper):
    """Scraper para estatísticas Misc do FBref (faltas e cartões)"""
    
    def __init__(self):
        self.base_url = "https://fbref.com/en"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Mapeamento de ligas
        self.league_mapping = {
            "Premier League": {
                "comp_id": "9",
                "url_pattern": "{season}/misc/{season}-Premier-League-Stats"
            },
            "Serie A (Brazil)": {
                "comp_id": "24", 
                "url_pattern": "{season}/misc/{season}-Serie-A-Stats"
            }
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    def _fetch_page(self, url: str) -> str:
        """Busca uma página com retry e timeout"""
        logger.info(f"Acessando URL: {url}")
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.text
    
    def _sanitize_numeric(self, value) -> float:
        """Sanitiza valores numéricos"""
        if pd.isna(value) or value == '' or value is None:
            return 0.0
        
        # Converter para string se não for
        str_value = str(value)
        
        # Remover símbolos e trocar vírgula por ponto
        cleaned = re.sub(r'[^\d.,\-]', '', str_value)
        cleaned = cleaned.replace(',', '.')
        
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0
    
    def _extract_tables_from_html(self, html_content: str) -> list:
        """Extrai tabelas do HTML, incluindo as comentadas"""
        soup = BeautifulSoup(html_content, 'lxml')
        tables = []
        
        # Buscar tabelas normais
        normal_tables = soup.find_all('table')
        for table in normal_tables:
            try:
                df = pd.read_html(str(table))[0]
                tables.append(df)
            except Exception as e:
                logger.warning(f"Erro ao processar tabela normal: {e}")
        
        # Buscar tabelas em comentários
        comments = soup.find_all(string=lambda text: isinstance(text, str) and '<table' in text)
        for comment in comments:
            try:
                comment_soup = BeautifulSoup(comment, 'lxml')
                comment_tables = comment_soup.find_all('table')
                for table in comment_tables:
                    df = pd.read_html(str(table))[0]
                    tables.append(df)
            except Exception as e:
                logger.warning(f"Erro ao processar tabela comentada: {e}")
        
        return tables
    
    def _find_misc_table(self, tables: list) -> pd.DataFrame:
        """Encontra a tabela com estatísticas Misc (faltas)"""
        for i, table in enumerate(tables):
            # Verificar se a tabela tem colunas relacionadas a faltas
            columns_str = ' '.join([str(col).lower() for col in table.columns])
            
            if any(keyword in columns_str for keyword in ['fls', 'fld', 'fouls', 'yellow', 'red', 'cards']):
                logger.info(f"Tabela Misc encontrada (índice {i}): {table.shape}")
                return table
        
        raise ValueError("Tabela com estatísticas Misc não encontrada")
    
    def _normalize_dataframe(self, df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
        """Normaliza o DataFrame para o formato padrão"""
        # Flatten multi-level columns se necessário
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        
        # Mapear colunas possíveis
        column_mapping = {
            # Player
            'player': 'player',
            'jogador': 'player',
            'nome': 'player',
            
            # Team
            'squad': 'team',
            'team': 'team',
            'clube': 'team',
            'time': 'team',
            
            # Position
            'pos': 'position',
            'position': 'position',
            'posição': 'position',
            
            # Appearances
            'mp': 'appearances',
            'apps': 'appearances',
            'jogos': 'appearances',
            '90s': 'appearances',
            
            # Fouls
            'fls': 'fouls',
            'fouls': 'fouls',
            'faltas': 'fouls',
            
            # Fouls drawn
            'fld': 'fouls_drawn',
            'fouls drawn': 'fouls_drawn',
            'faltas sofridas': 'fouls_drawn',
            
            # Yellow cards
            'crdy': 'yellow_cards',
            'yellow': 'yellow_cards',
            'amarelos': 'yellow_cards',
            
            # Red cards
            'crdr': 'red_cards',
            'red': 'red_cards',
            'vermelhos': 'red_cards',
            
            # Minutes
            'min': 'minutes',
            'minutes': 'minutes',
            'minutos': 'minutes'
        }
        
        # Criar DataFrame normalizado
        normalized_data = {
            'league': league,
            'season': season,
            'source': 'FBref (Misc)'
        }
        
        # Mapear colunas existentes
        df_lower_cols = {col.lower(): col for col in df.columns}
        
        for target_col, source_patterns in column_mapping.items():
            if not isinstance(source_patterns, list):
                source_patterns = [source_patterns]
            
            found = False
            for pattern in source_patterns:
                if pattern.lower() in df_lower_cols:
                    source_col = df_lower_cols[pattern.lower()]
                    if target_col in ['appearances', 'fouls', 'fouls_drawn', 'yellow_cards', 'red_cards', 'minutes']:
                        normalized_data[target_col] = df[source_col].apply(self._sanitize_numeric)
                    else:
                        normalized_data[target_col] = df[source_col].astype(str)
                    found = True
                    break
            
            if not found and target_col not in normalized_data:
                if target_col in ['appearances', 'fouls', 'fouls_drawn', 'yellow_cards', 'red_cards', 'minutes']:
                    normalized_data[target_col] = 0.0
                else:
                    normalized_data[target_col] = ''
        
        result_df = pd.DataFrame(normalized_data)
        
        # Remover linhas vazias ou com jogador vazio
        result_df = result_df[result_df['player'].str.strip() != '']
        result_df = result_df.dropna(subset=['player'])
        
        # Remover duplicatas
        result_df = result_df.drop_duplicates(subset=['player', 'team'])
        
        logger.info(f"DataFrame normalizado: {result_df.shape}")
        return result_df
    
    def fetch(self, league: str, season: str) -> pd.DataFrame:
        """Busca dados de uma liga e temporada específica"""
        if league not in self.league_mapping:
            raise ValueError(f"Liga não suportada: {league}")
        
        league_config = self.league_mapping[league]
        url_pattern = league_config["url_pattern"]
        
        # Construir URL
        url = f"{self.base_url}/comps/{league_config['comp_id']}/{url_pattern.format(season=season)}"
        
        try:
            # Buscar página
            html_content = self._fetch_page(url)
            
            # Extrair tabelas
            tables = self._extract_tables_from_html(html_content)
            
            if not tables:
                raise ValueError("Nenhuma tabela encontrada na página")
            
            # Encontrar tabela Misc
            misc_table = self._find_misc_table(tables)
            
            # Normalizar dados
            normalized_df = self._normalize_dataframe(misc_table, league, season)
            
            # Garantir colunas padrão
            result_df = ensure_columns(normalized_df)
            
            logger.info(f"Scraping concluído: {league} {season} - {len(result_df)} linhas")
            return result_df
            
        except Exception as e:
            logger.error(f"Erro no scraping {league} {season}: {e}")
            raise