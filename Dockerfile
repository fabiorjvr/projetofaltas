# ---- Base de execução ----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependências de sistema para psycopg2 e builds leves
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Copia dependências primeiro (melhor cache)
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Copia o código
COPY . .

# Faz a API "enxergar" tanto backend/ quanto src/
ENV PYTHONPATH=/app:/app/src:/app/backend

# Usuário não-root por segurança
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# Healthcheck do container: bate no /health
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Comando padrão (dev/prod simples)
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]