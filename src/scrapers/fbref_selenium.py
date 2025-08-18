from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import numpy as np
from .base import BaseScraper, ensure_columns

# Funções de normalização e limpeza
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Achata MultiIndex de colunas do FBref: ('Performance','CrdY') -> 'CrdY' """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    return df

def _coerce_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    s = s.replace(',', '.')
    s = re.sub(r'[^0-9\.\-]', '', s)  # tira símbolos
    try:
        return float(s) if s != '' else np.nan
    except:
        return np.nan

def _clean_fbref_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas de cabeçalho repetido / agregações de time.
    Mantém apenas jogadores.
    """
    # 1) achatar colunas e padronizar nomes possíveis
    df = _flatten_cols(df).copy()

    # renomear variações comuns
    rename_map = {
        'Player': 'player',
        'Squad': 'team',
        'Pos': 'position',
        'MP': 'appearances',
        'Fls': 'fouls',
        'Fld': 'fouls_drawn',
        'CrdY': 'yellow_cards',
        'CrdR': 'red_cards',
        'Min': 'minutes',
        '90s': 'nineties',   # opcional, se vier
        '# Pl': 'players_ct' # quando é tabela de times
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # 2) descartar linhas que são, claramente, cabeçalho repetido
    drop_if_equals = ['player', 'team', 'position']
    for col in drop_if_equals:
        if col in df.columns:
            df = df[df[col].astype(str).str.lower() != col]

    # 3) descartar linhas de agregação por time (quando dataframe de jogadores costuma ter colunas 'player')
    if 'player' in df.columns and 'team' in df.columns:
        # se a coluna player está vazia e tem 'team' preenchido, provavelmente é linha de time
        df = df[df['player'].notna() & (df['player'].astype(str).str.strip() != '')]

    # 4) tipar numéricos
    for num_col in ['appearances', 'fouls', 'fouls_drawn', 'yellow_cards', 'red_cards', 'minutes']:
        if num_col in df.columns:
            df[num_col] = df[num_col].map(_coerce_num).fillna(0.0)

    # 5) strings limpas
    for txt_col in ['player', 'team', 'position']:
        if txt_col in df.columns:
            df[txt_col] = df[txt_col].astype(str).str.strip()

    # 6) só colunas de interesse; outras serão completadas pelo ensure_columns
    keep = ['player','team','position','appearances','fouls','fouls_drawn','yellow_cards','red_cards','minutes']
    keep_existing = [c for c in keep if c in df.columns]
    df = df[keep_existing].copy()

    # 7) ainda pode ter sobrado algo "Player/Squad" (se escapou); drop explícito
    if 'player' in df.columns:
        df = df[df['player'].str.lower() != 'player']
    if 'team' in df.columns:
        df = df[df['team'].str.lower() != 'squad']

    # 8) dedup de segurança por (player, team)
    if {'player','team'}.issubset(df.columns):
        df.drop_duplicates(subset=['player','team'], inplace=True)

    return df

class FBRefSeleniumScraper(BaseScraper):
    def __init__(self):
        self.driver = None
        
    def _setup_driver(self):
        """Configura o driver Chrome com opções anti-detecção"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            # Tentar usar webdriver-manager para baixar automaticamente o ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            print("✅ ChromeDriver configurado via webdriver-manager")
        except Exception as e:
            print(f"⚠️ Erro com webdriver-manager: {e}")
            try:
                # Fallback: tentar usar ChromeDriver do PATH
                self.driver = webdriver.Chrome(options=chrome_options)
                print("✅ ChromeDriver configurado via PATH")
            except Exception as e2:
                print(f"❌ Erro ao configurar ChromeDriver: {e2}")
                raise Exception(f"Não foi possível configurar o ChromeDriver. Instale o Google Chrome e o ChromeDriver.")
        
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
    def fetch(self, league: str, season: str) -> pd.DataFrame:
        """Busca dados de faltas do FBRef usando Selenium"""
        if not self.driver:
            self._setup_driver()
            
        # URLs baseadas no mapeamento
        league_map = {
            "Premier League": ("9", "Premier-League-Stats"),
            "Serie A (Brazil)": ("24", "Serie-A-Stats"),
            "La Liga": ("12", "La-Liga-Stats")
        }
        
        if league not in league_map:
            print(f"Liga '{league}' não suportada")
            return pd.DataFrame()
            
        comp_id, url_name = league_map[league]
        # URL para estatísticas misc (contém dados de faltas)
        url = f"https://fbref.com/en/comps/{comp_id}/{season}/misc/{season}-{url_name}"
        
        try:
            print(f"Acessando: {url}")
            self.driver.get(url)
            
            # Aguardar página carregar
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            
            # Tentar clicar no link para estatísticas de jogadores se estivermos na página de times
            try:
                player_link = self.driver.find_element(By.LINK_TEXT, "Player Stats")
                if player_link:
                    print("🔄 Navegando para estatísticas de jogadores...")
                    player_link.click()
                    time.sleep(3)
            except:
                print("ℹ️ Já na página de jogadores ou link não encontrado")
            
            # Scroll para garantir carregamento completo
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            
            # Obter HTML
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Buscar tabela de estatísticas misc
            from io import StringIO
            tables = pd.read_html(StringIO(str(soup)))
            
            print(f"📊 {len(tables)} tabelas encontradas na página")
            
            # Processar tabelas procurando por dados de faltas de jogadores
            df = None
            for i, table in enumerate(tables):
                # Verificar se a tabela tem colunas relacionadas a faltas
                columns = [str(col) for col in table.columns]
                print(f"Tabela {i+1}: {len(table)} linhas, colunas: {columns[:5]}...")  # Mostrar primeiras 5 colunas
                
                # Verificar se é tabela de jogadores (não de times)
                has_player_col = any('Player' in str(col) for col in columns)
                has_squad_only = any('Squad' in str(col) for col in columns) and not has_player_col
                
                if has_squad_only:
                    print(f"⚠️ Tabela {i+1} contém dados de times, pulando...")
                    continue
                
                # Procurar por colunas de faltas em diferentes formatos
                foul_indicators = ['Fls', 'Fld', 'CrdY', 'Fouls', 'Fouled', 'Yellow', 'Miscellaneous']
                if any(indicator in str(columns) for indicator in foul_indicators) and has_player_col:
                    print(f"✅ Tabela {i+1} contém dados de faltas de jogadores")
                    df = table.copy()
                    break
                    
            if df is None or df.empty:
                print("❌ Nenhuma tabela com dados de faltas encontrada")
                print("🔍 Tentando buscar tabela por ID específico...")
                
                # Tentar buscar tabela específica por ID (estatísticas misc de jogadores)
                misc_table = soup.find('table', {'id': 'stats_misc'})
                if misc_table:
                    df = pd.read_html(StringIO(str(misc_table)))[0]
                    print(f"✅ Tabela misc encontrada por ID: {len(df)} linhas")
                else:
                    # Tentar outras tabelas de estatísticas de jogadores
                    player_tables = soup.find_all('table', {'class': 'stats_table'})
                    for table in player_tables:
                        if 'misc' in str(table.get('id', '')):
                            df = pd.read_html(StringIO(str(table)))[0]
                            print(f"✅ Tabela misc encontrada: {len(df)} linhas")
                            break
                    else:
                        print("❌ Nenhuma tabela misc encontrada")
                        return pd.DataFrame()
                
            # Aplicar limpeza robusta
            raw = df.copy()
            df = _clean_fbref_rows(raw)
            
            # Completar metadados e ordem
            df['league'] = league
            df['season'] = season
            df['source'] = "fbref_selenium"
            
            # Aplicar ensure_columns para garantir formato padrão
            df = ensure_columns(df)
            
            return df
            
        except Exception as e:
            print(f"Erro no scraping Selenium: {e}")
            return pd.DataFrame()
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None
                
    def _normalize_dataframe(self, df, league, season):
        """Normaliza o DataFrame obtido via Selenium"""
        try:
            # Lidar com MultiIndex columns se necessário
            if isinstance(df.columns, pd.MultiIndex):
                # Pegar o último nível das colunas
                df.columns = [col[-1] if isinstance(col, tuple) else col for col in df.columns]
            
            # Converter colunas para string para facilitar mapeamento
            df.columns = [str(col) for col in df.columns]
            
            # Remover linhas vazias ou com dados inválidos
            df = df.dropna(subset=['Player'] if 'Player' in df.columns else [df.columns[0]])
            
            # Filtrar apenas jogadores (remover totais/médias)
            if 'Player' in df.columns:
                df = df[~df['Player'].str.contains('Squad Total|Total|Average', na=False, case=False)]
            
            # Adicionar metadados
            df['league'] = league
            df['season'] = season
            df['source'] = 'fbref_selenium'
            df['scraped_at'] = pd.Timestamp.now()
            
            # Converter colunas numéricas
            numeric_columns = ['Fls', 'Fld', 'CrdY', 'CrdR', 'Min', 'MP']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Substituir NaN por 0 para colunas numéricas
                    df[col] = df[col].fillna(0)
            
            print(f"DataFrame normalizado: {len(df)} jogadores encontrados")
            return df
            
        except Exception as e:
            print(f"Erro na normalização: {e}")
            return pd.DataFrame()
    
    def __del__(self):
        """Cleanup do driver ao destruir o objeto"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass