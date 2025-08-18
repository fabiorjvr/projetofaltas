#!/usr/bin/env python3
"""
CLI para coletar dados do FBref e inserir no banco PostgreSQL
"""

import click
import sys
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import logging

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.db import Base, get_session
from src.database.models import PlayerSeasonFouls
from .scrapers.fbref_misc import FBRefMiscScraper
from .scrapers.fbref_selenium import FBRefSeleniumScraper

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upsert_df(session: Session, df: pd.DataFrame, league: str, season: str):
    """
    Faz upsert atômico por (league,season,player,team) usando INSERT .. ON CONFLICT DO UPDATE.
    """
    if df.empty:
        logger.warning("DataFrame vazio - nenhum dado para upsert")
        return 0
    
    # Limpar dados problemáticos
    df_clean = df.copy()
    df_clean = df_clean.where(pd.notnull(df_clean), None)
    
    # Converter para lista de dicionários
    records = df_clean.to_dict(orient="records")
    
    # Preparar statement de upsert
    stmt = insert(PlayerSeasonFouls).values(records)
    
    # Definir colunas para atualizar em caso de conflito
    update_cols = {
        'position': stmt.excluded.position,
        'appearances': stmt.excluded.appearances,
        'fouls': stmt.excluded.fouls,
        'fouls_drawn': stmt.excluded.fouls_drawn,
        'yellow_cards': stmt.excluded.yellow_cards,
        'red_cards': stmt.excluded.red_cards,
        'minutes': stmt.excluded.minutes,
        'source': stmt.excluded.source,
    }
    
    stmt = stmt.on_conflict_do_update(
        index_elements=['league', 'season', 'player', 'team'],  # chaves únicas
        set_=update_cols
    )
    
    # Executar upsert
    session.execute(stmt)
    session.commit()
    
    logger.info(f"Upsert concluído para {len(records)} registros de {league} {season}")
    return len(records)

def setup_database():
    """Configura o banco de dados"""
    load_dotenv()
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL não encontrada no arquivo .env")
    
    engine = create_engine(database_url, pool_pre_ping=True)
    Base.metadata.create_all(bind=engine)
    return engine

def upsert_data(engine, df: pd.DataFrame, league: str, season: str):
    """Faz upsert dos dados no banco"""
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    
    with SessionLocal() as session:
        try:
            # Limpar duplicatas por garantia
            df_clean = df.drop_duplicates(subset=["league", "season", "player", "team"])
            logger.info(f"Após remoção de duplicatas: {len(df_clean)} registros (eram {len(df)})")
            
            # Usar upsert atômico
            total_upserted = upsert_df(session, df_clean, league, season)
            logger.info(f"✅ upsert concluído: {league} {season} → {total_upserted} linhas")
                
        except Exception as e:
            session.rollback()
            logger.error(f"Erro durante upsert: {e}")
            raise

def fetch_fbref_data(league: str, season: str):
    """Busca dados do FBref"""
    logger.info(f"Iniciando coleta para {league} {season}")
    
    try:
        scraper = FBRefMiscScraper()
        df = scraper.fetch(league, season)
        
        if df.empty:
            logger.warning(f"Nenhum dado encontrado para {league} {season}")
            return df
            
        logger.info(f"Coletados {len(df)} registros para {league} {season}")
        
        # Verificar colunas obrigatórias
        required_columns = ['league', 'season', 'player', 'team', 'position', 
                          'appearances', 'fouls', 'fouls_drawn', 'yellow_cards', 
                          'red_cards', 'minutes', 'source']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Colunas obrigatórias ausentes: {missing_columns}")
            return pd.DataFrame()
            
        return df
        
    except Exception as e:
        logger.error(f"Erro ao buscar dados: {e}")
        return pd.DataFrame()

@click.group()
def cli():
    """Football Fouls Analytics CLI"""
    pass

@cli.command()
def hello():
    """Comando de teste"""
    click.echo("Hello from Football Fouls Analytics!")

@cli.command('fetch-fbref')
@click.option('--league', required=True, help='Nome da liga')
@click.option('--season', required=True, help='Temporada')
def fetch_fbref(league: str, season: str):
    """Buscar dados do FBref e inserir no banco"""
    try:
        # Configurar banco
        engine = setup_database()
        
        # Buscar dados
        df = fetch_fbref_data(league, season)
        
        if not df.empty:
            # Fazer upsert
            upsert_data(engine, df, league, season)
            click.echo("Processo concluído com sucesso!")
        else:
            click.echo("Nenhum dado para inserir", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Erro durante execução: {e}", err=True)
        sys.exit(1)

@cli.command('fetch-fbref-selenium')
@click.option('--league', required=True, help='Nome da liga')
@click.option('--season', required=True, help='Temporada')
def fetch_fbref_selenium(league: str, season: str):
    """Buscar dados do FBref usando Selenium (bypass anti-bot)"""
    try:
        # Configurar banco
        engine = setup_database()
        
        # Buscar dados com Selenium
        scraper = FBRefSeleniumScraper()
        df = scraper.fetch(league, season)
        
        if not df.empty:
            # Debug: verificar formato dos dados
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"Colunas: {list(df.columns)}")
            logger.info(f"Primeiras 3 linhas:\n{df.head(3)}")
            # Fazer upsert
            upsert_data(engine, df, league, season)
            click.echo("Processo concluído com sucesso (via Selenium)!")
        else:
            click.echo("Nenhum dado para inserir", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Erro durante execução: {e}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()