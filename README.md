# ⚽ Football Fouls Analytics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37.1-red.svg)](https://streamlit.io/)

Um sistema completo de análise de faltas no futebol que coleta dados através de web scraping, armazena em PostgreSQL e fornece uma API REST e dashboard interativo para visualização e análise dos dados.

## 🚀 Funcionalidades

### 📊 Web Scraping
- **Coleta automatizada** de dados de faltas do FBRef
- **Suporte a múltiplas ligas**: Premier League, La Liga, Serie A (Brasil)
- **Limpeza robusta** de dados com filtros anti-header
- **Deduplicação automática** para evitar registros duplicados
- **Upsert atômico** com PostgreSQL para alta performance

### 🔌 API REST (FastAPI)
- **Endpoints RESTful** para consulta de dados
- **Filtros avançados** por liga, temporada, time e estatísticas
- **Paginação** e ordenação personalizável
- **Documentação automática** com Swagger UI
- **Validação** com Pydantic schemas

### 📈 Dashboard (Streamlit)
- **Interface interativa** para visualização de dados
- **Gráficos dinâmicos** com Plotly
- **Filtros em tempo real**
- **Análises estatísticas** detalhadas

## 🏗️ Arquitetura

```
football-fouls-analytics/
├── src/
│   ├── scrapers/           # Web scrapers
│   │   ├── base.py         # Classe base para scrapers
│   │   └── fbref_selenium.py # Scraper do FBRef
│   ├── database/           # Modelos e configuração do banco
│   │   ├── models.py       # Modelos SQLAlchemy
│   │   └── connection.py   # Configuração de conexão
│   └── cli.py             # Interface de linha de comando
├── backend/
│   └── app/               # API FastAPI
│       ├── main.py        # Aplicação principal
│       ├── core/          # Configurações
│       ├── api/v1/        # Endpoints da API
│       └── schemas/       # Schemas Pydantic
├── dashboard/             # Dashboard Streamlit
└── requirements.txt       # Dependências
```

## 🛠️ Instalação

### Pré-requisitos
- Python 3.8+
- PostgreSQL 13+
- Chrome/Chromium (para web scraping)

### 1. Clone o repositório
```bash
git clone https://github.com/fabiorjvr/projetofaltas.git
cd football-fouls-analytics
```

### 2. Crie um ambiente virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Unix/Linux/macOS
python -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure o banco de dados

Crie um arquivo `.env` na raiz do projeto:
```env
DATABASE_URL=postgresql://usuario:senha@localhost:5432/football_fouls
```

### 5. Execute as migrações
```bash
python -c "from src.database.models import create_tables; create_tables()"
```

## 🚀 Uso

### Web Scraping

#### Coletar dados de uma liga específica:
```bash
python -m src.cli --league "Premier League" --season "2024-2025"
python -m src.cli --league "La Liga" --season "2024-2025"
python -m src.cli --league "Serie A" --season "2024-2025"
```

### API REST

#### Iniciar o servidor da API:
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

A API estará disponível em: http://localhost:8000

**Documentação interativa:** http://localhost:8000/docs

#### Endpoints principais:

- **GET /api/v1/players** - Lista jogadores com filtros
- **GET /api/v1/stats/top-fouls** - Top jogadores com mais faltas
- **GET /api/v1/teams/fouls-summary** - Resumo de faltas por time

#### Exemplos de uso:

```bash
# Listar jogadores da Premier League 2024-2025
curl "http://localhost:8000/api/v1/players?league=Premier League&season=2024-2025&limit=10"

# Top 5 jogadores com mais faltas
curl "http://localhost:8000/api/v1/stats/top-fouls?league=Premier League&season=2024-2025&limit=5"

# Resumo de faltas por time
curl "http://localhost:8000/api/v1/teams/fouls-summary?league=Premier League&season=2024-2025"
```

### Dashboard

#### Iniciar o dashboard:
```bash
streamlit run dashboard/app.py
```

O dashboard estará disponível em: http://localhost:8501

## 📊 Dados Coletados

Para cada jogador, o sistema coleta:

- **Informações básicas**: Nome, time, posição
- **Estatísticas de jogo**: Aparições, minutos jogados
- **Estatísticas de faltas**: Faltas cometidas, faltas sofridas
- **Cartões**: Cartões amarelos e vermelhos
- **Metadados**: Liga, temporada, fonte dos dados

## 🔧 Configuração Avançada

### Variáveis de Ambiente

```env
# Banco de dados
DATABASE_URL=postgresql://usuario:senha@localhost:5432/football_fouls

# API
API_TITLE=Football Fouls Analytics API
API_VERSION=0.1.0
DEBUG=false

# Scraping
SCRAPE_DELAY=2.0
MAX_RETRIES=3
```

### Configuração do Chrome para Scraping

O sistema usa Chrome em modo headless. Para configurar:

1. Instale o Chrome/Chromium
2. O webdriver-manager baixará automaticamente o ChromeDriver
3. Para servidores sem interface gráfica, use: `--headless --no-sandbox --disable-dev-shm-usage`

## 🧪 Testes

### Testar a coleta de dados:
```bash
python -m src.cli --league "Premier League" --season "2024-2025" --limit 10
```

### Testar a API:
```bash
# Health check
curl http://localhost:8000/health

# Listar jogadores
curl "http://localhost:8000/api/v1/players?limit=5"
```

### Testar o dashboard:
```bash
streamlit hello
```

## 📈 Performance

- **Upsert atômico**: Inserção/atualização em lote para alta performance
- **Deduplicação**: Evita registros duplicados automaticamente
- **Índices otimizados**: Consultas rápidas por liga, temporada e jogador
- **Connection pooling**: Gerenciamento eficiente de conexões do banco

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- [FBRef](https://fbref.com/) pelos dados de futebol
- [FastAPI](https://fastapi.tiangolo.com/) pelo framework web
- [Streamlit](https://streamlit.io/) pelo framework de dashboard
- [Selenium](https://selenium.dev/) pela automação web

## 📞 Contato

**Fábio Vieira** - [@fabiorjvr](https://github.com/fabiorjvr)

**Link do Projeto:** [https://github.com/fabiorjvr/projetofaltas](https://github.com/fabiorjvr/projetofaltas)

---

⭐ Se este projeto te ajudou, considere dar uma estrela!