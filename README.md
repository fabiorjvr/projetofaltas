# Football Fouls Analytics

Este projeto analisa dados de faltas de futebol, coletando informações de jogadores através de web scraping e fornecendo uma API e dashboard para visualização dos dados. O sistema utiliza PostgreSQL para armazenamento, FastAPI para a API REST e Streamlit para o dashboard interativo.

## Configuração do Ambiente

### 1. Criar ambiente virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Unix/Linux/macOS
python -m venv venv
source venv/bin/activate
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Testar ambiente
```bash
streamlit hello
```

Se tudo estiver configurado corretamente, o Streamlit abrirá uma página de demonstração no navegador.