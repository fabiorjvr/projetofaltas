# src/etl/fbref_comp_misc_loader.py
import asyncio
import pandas as pd
import re
from datetime import datetime
from sqlalchemy import text
from loguru import logger

# Usa a mesma engine/config da API
from backend.app.core.db import engine
from backend.app.core.config import settings

# Mapeamento de competições FBref:
# Premier League = comp_id 9, temporadas estilo 2023-2024, 2024-2025
# Série A (Brasil) = comp_id 24, temporadas ano único (2023, 2024, 2025)
COMP_MAP = {
    "Premier League": {"comp_id": 9, "seasons": ["2023-2024", "2024-2025"]},
    "Serie A (Brazil)": {"comp_id": 24, "seasons": ["2023", "2024"]},
}

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    # FBref às vezes traz MultiIndex nas colunas
    df.columns = ["_".join([c for c in map(str, col) if c and c != "nan"]).strip("_") for col in df.columns]
    return df

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Remover linhas de cabeçalho repetidas (Rk)
    if "Rk" in df.columns:
        df = df[df["Rk"].astype(str).str.match(r"^\d+$", na=False)]
    # Normalizações de nomes usuais
    rename = {
        "Player": "player_name",
        "Squad": "team",
        "Pos": "position",
        "Fls": "fouls",
        "Fld": "fouls_drawn",
        "CrdY": "yellow_cards",
        "CrdR": "red_cards",
        "Min": "minutes",
        "MP": "appearances",
    }
    for k, v in rename.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Converte numéricos
    for col in ["fouls", "fouls_drawn", "yellow_cards", "red_cards", "minutes", "appearances"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Campos obrigatórios garantidos
    for col in ["fouls", "fouls_drawn", "yellow_cards", "red_cards", "minutes", "appearances"]:
        if col not in df.columns:
            df[col] = 0

    # Remove linhas sem jogador/time
    df = df[df["player_name"].notna() & df["team"].notna()]
    return df

def _build_url(comp_id: int, season: str) -> str:
    """
    Exemplo EPL 24/25:
    `https://fbref.com/en/comps/9/2024-2025/misc/players/2024-2025-Premier-League-Stats`

    Exemplo Série A 2024:
    `https://fbref.com/en/comps/24/2024/misc/players/2024-Serie-A-Stats`
    """
    if re.match(r"^\d{4}-\d{4}$", season):
        # formato europeu (EPL)
        return f"https://fbref.com/en/comps/{comp_id}/{season}/misc/players/{season}-Premier-League-Stats"
    else:
        # formato ano único (Brasil)
        return f"https://fbref.com/en/comps/{comp_id}/{season}/misc/players/{season}-Serie-A-Stats"

async def _fetch_table_html(url: str) -> str:
    # Playwright headless para evitar bloqueio
    from playwright.async_api import async_playwright
    logger.info(f"Fetching: {url}")
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        page = await context.new_page()
        
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            # Tenta aguardar por qualquer tabela, não apenas stats_misc
            try:
                await page.wait_for_selector("table", timeout=15_000)
                logger.info("Tabela encontrada, aguardando carregamento completo...")
                await page.wait_for_timeout(3000)  # Aguarda 3s para carregamento completo
            except:
                logger.warning("Timeout aguardando tabela, continuando...")
            
            html = await page.content()
            logger.success(f"HTML obtido com sucesso: {len(html)} caracteres")
            return html
        finally:
            await context.close()
            await browser.close()

def _parse_misc_table(html: str) -> pd.DataFrame:
    # Lê a(s) tabela(s) com pandas
    tables = pd.read_html(html)
    # Heurística: pegar a tabela que contenha colunas de faltas/cartões
    candidates = []
    for t in tables:
        cols = set(map(str, t.columns))
        score = sum(int(any(key in c for c in cols)) for key in [["Fls"], ["Fld"], ["CrdY"], ["CrdR"], ["Min"], ["MP"]])
        candidates.append((score, t))
    candidates.sort(key=lambda x: x[0], reverse=True)
    df = candidates[0][1] if candidates else tables[0]
    df = _flatten_cols(df)
    # Se após flatten ainda tiver cabeçalhos esquisitos, tenta normalizar nomes simples:
    simple_cols = {c.split("_")[-1]: c for c in df.columns}
    for need in ["Player", "Squad", "Pos", "Fls", "Fld", "CrdY", "CrdR", "Min", "MP"]:
        if need not in df.columns and need in simple_cols:
            df.rename(columns={simple_cols[need]: need}, inplace=True)
    return _clean_df(df)

def _upsert_psf(df: pd.DataFrame, league: str, season: str):
    df = df.copy()
    df["league"] = league
    df["season"] = season
    df["source"] = "fbref"
    df["scraped_at"] = datetime.utcnow()

    insert_sql = text("""
    INSERT INTO player_season_fouls
    (player_name, team, league, season, position, appearances, fouls, fouls_drawn, yellow_cards, red_cards, minutes, source, scraped_at)
    VALUES
    (:player_name, :team, :league, :season, :position, :appearances, :fouls, :fouls_drawn, :yellow_cards, :red_cards, :minutes, :source, :scraped_at)
    ON CONFLICT (player_name, team, season, league)
    DO UPDATE SET
      position = EXCLUDED.position,
      appearances = EXCLUDED.appearances,
      fouls = EXCLUDED.fouls,
      fouls_drawn = EXCLUDED.fouls_drawn,
      yellow_cards = EXCLUDED.yellow_cards,
      red_cards = EXCLUDED.red_cards,
      minutes = EXCLUDED.minutes,
      source = EXCLUDED.source,
      scraped_at = EXCLUDED.scraped_at;
    """)

    rows = df[["player_name","team","league","season","position","appearances","fouls","fouls_drawn","yellow_cards","red_cards","minutes","source","scraped_at"]].to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(insert_sql, rows)
    logger.success(f"Upsert OK: {league} {season} -> {len(rows)} jogadores")

async def run_load():
    for league, meta in COMP_MAP.items():
        comp_id = meta["comp_id"]
        for season in meta["seasons"]:
            url = _build_url(comp_id, season)
            try:
                html = await _fetch_table_html(url)
                df = _parse_misc_table(html)
                _upsert_psf(df, league=league, season=season)
            except Exception as e:
                logger.exception(f"Falhou {league} {season}: {e}")

if __name__ == "__main__":
    asyncio.run(run_load())