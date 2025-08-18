#!/usr/bin/env python3
"""
Script para criar as tabelas no banco de dados PostgreSQL
"""

import sys
import os

# Adiciona o diretório src ao path para importar os módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.db import Base, engine
from database.models import PlayerSeasonFouls
from dotenv import load_dotenv

def create_tables():
    """Cria todas as tabelas no banco de dados"""
    print("Carregando variáveis de ambiente...")
    load_dotenv()
    
    print(f"Conectando ao banco: {os.getenv('DATABASE_URL')}")
    
    try:
        print("Criando tabelas...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tabelas criadas com sucesso!")
        
        # Lista as tabelas criadas
        print("\nTabelas criadas:")
        for table_name in Base.metadata.tables.keys():
            print(f"  - {table_name}")
            
    except Exception as e:
        print(f"❌ Erro ao criar tabelas: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_tables()