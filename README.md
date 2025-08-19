# ⚽ Football Fouls Analytics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg)](https://www.typescriptlang.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)

Um sistema completo de análise de faltas no futebol com **Machine Learning**, **PWA** e **API REST**. Coleta dados através de web scraping, processa com algoritmos de ML para predições e oferece uma interface moderna React com animações e funcionalidades offline.

## 🚀 Funcionalidades

### 🤖 Machine Learning & IA
- **Pipeline ML completo** com XGBoost para predição de faltas e cartões
- **Clustering de perfis** de jogadores com algoritmos não-supervisionados
- **Detecção de anomalias** em comportamento de jogadores
- **MLflow** para versionamento e serving de modelos
- **Predições em tempo real** via API REST
- **Métricas de performance** e monitoramento de modelos

### 📊 Web Scraping & ETL
- **Coleta automatizada** de dados do FBRef com Selenium
- **Suporte a múltiplas ligas**: Premier League, La Liga, Serie A (Brasil)
- **Pipeline ETL robusto** com limpeza e validação de dados
- **Deduplicação automática** e upsert atômico
- **Tratamento de erros** e retry automático

### 🔌 API REST (FastAPI)
- **Autenticação JWT** com middleware de segurança
- **Rate limiting** e validações Pydantic robustas
- **Endpoints ML** para predições e análises
- **Documentação automática** com Swagger UI
- **CORS configurado** para integração frontend
- **Logging estruturado** e monitoramento

### 💻 Frontend PWA (React 18 + TypeScript)
- **Progressive Web App** com funcionalidades offline
- **Interface moderna** com Tailwind CSS e animações Framer Motion
- **Gerenciamento de estado** com Zustand e React Query
- **Componentes reutilizáveis** e design system
- **Responsivo** para desktop, tablet e mobile
- **Notificações push** e sincronização em background
- **Instalável** como app nativo

### 📈 Dashboard & Analytics
- **Visualizações interativas** com gráficos dinâmicos
- **Filtros em tempo real** e busca avançada
- **Análises estatísticas** detalhadas
- **Exportação de dados** em múltiplos formatos
- **Dashboards personalizáveis** por usuário

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
│   ├── ml/                # Pipeline Machine Learning
│   │   ├── models/        # Modelos ML (XGBoost, Clustering)
│   │   ├── preprocessing/ # Preprocessamento de dados
│   │   ├── training/      # Scripts de treinamento
│   │   └── inference/     # Inferência e predições
│   └── cli.py             # Interface de linha de comando
├── backend/
│   └── app/               # API FastAPI
│       ├── main.py        # Aplicação principal
│       ├── core/          # Configurações e segurança
│       ├── api/v1/        # Endpoints da API
│       ├── schemas/       # Schemas Pydantic
│       ├── auth/          # Autenticação JWT
│       └── ml/            # Endpoints ML
├── frontend/              # PWA React + TypeScript
│   ├── src/
│   │   ├── components/    # Componentes React
│   │   ├── pages/         # Páginas da aplicação
│   │   ├── store/         # Gerenciamento de estado (Zustand)
│   │   ├── hooks/         # Custom hooks
│   │   └── utils/         # Utilitários
│   ├── public/            # Assets estáticos
│   └── package.json       # Dependências frontend
├── dashboard/             # Dashboard Streamlit (legacy)
├── mlflow/               # Experimentos e modelos ML
└── requirements.txt       # Dependências Python
```

## 🛠️ Instalação

### Pré-requisitos
- Python 3.8+
- Node.js 18+ e npm
- PostgreSQL 13+
- Chrome/Chromium (para web scraping)
- MLflow (para ML pipeline)

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

### 3. Instale as dependências do backend
```bash
pip install -r requirements.txt
```

### 4. Instale as dependências do frontend
```bash
cd frontend
npm install
cd ..
```

### 5. Configure o banco de dados

Crie um arquivo `.env` na raiz do projeto:
```env
DATABASE_URL=postgresql://usuario:senha@localhost:5432/football_fouls
```

### 6. Execute as migrações
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

### API REST (Backend)

#### Iniciar o servidor da API:
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

A API estará disponível em: http://localhost:8000

**Documentação interativa:** http://localhost:8000/docs

### Frontend PWA

#### Iniciar o frontend:
```bash
cd frontend
npm run dev
```

O frontend estará disponível em: http://localhost:3000

**Funcionalidades PWA:**
- Interface moderna e responsiva
- Funcionalidades offline
- Instalável como app nativo
- Animações fluidas com Framer Motion
- Gerenciamento de estado com Zustand

#### Endpoints principais:

**Autenticação:**
- **POST /api/v1/auth/login** - Login e obtenção de token JWT
- **POST /api/v1/auth/register** - Registro de novo usuário

**Dados:**
- **GET /api/v1/players** - Lista jogadores com filtros
- **GET /api/v1/stats/top-fouls** - Top jogadores com mais faltas
- **GET /api/v1/teams/fouls-summary** - Resumo de faltas por time

**Machine Learning:**
- **POST /api/v1/ml/predict/fouls** - Predição de faltas
- **POST /api/v1/ml/predict/cards** - Predição de cartões
- **GET /api/v1/ml/clusters/players** - Clustering de jogadores

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

## 🚀 Tecnologias Utilizadas

### Backend
- **Python 3.8+** - Linguagem principal
- **FastAPI** - Framework web moderno e rápido
- **SQLAlchemy** - ORM para PostgreSQL
- **Pydantic** - Validação de dados
- **JWT** - Autenticação segura
- **XGBoost** - Machine Learning
- **MLflow** - MLOps e versionamento de modelos
- **Selenium** - Web scraping

### Frontend
- **React 18** - Biblioteca UI moderna
- **TypeScript** - Tipagem estática
- **Vite** - Build tool rápido
- **Tailwind CSS** - Framework CSS utilitário
- **Framer Motion** - Animações fluidas
- **Zustand** - Gerenciamento de estado
- **React Query** - Cache e sincronização de dados
- **PWA** - Progressive Web App

### Infraestrutura
- **PostgreSQL** - Banco de dados relacional
- **Docker** - Containerização
- **Nginx** - Proxy reverso
- **GitHub Actions** - CI/CD

## 📞 Contato

**Fábio Rosestolato Ferreira**
- 📧 Email: fabiorjvr@gmail.com
- 📱 Telefone: (21) 98030-6189
- 🐙 GitHub: [@fabiorjvr](https://github.com/fabiorjvr)

**Link do Projeto:** [https://github.com/fabiorjvr/projetofaltas](https://github.com/fabiorjvr/projetofaltas)

---

*Desenvolvido com ❤️ para análise avançada de dados esportivos*

⭐ Se este projeto te ajudou, considere dar uma estrela!