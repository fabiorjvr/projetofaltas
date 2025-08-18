from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.db import get_db

# Common dependencies for API endpoints
def get_database() -> Session:
    return Depends(get_db)