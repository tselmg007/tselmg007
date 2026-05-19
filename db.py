from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# ─────────────────────────────────────────────────────────────────────────────
# MySQL Database Configuration
# ─────────────────────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "guitar_analyzer")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "root")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,        # Connection-ийг автоматаар шалгана
    pool_recycle=3600,         # 1 цаг тутамд холболтыг шинэчилнэ
    echo=False,                # SQL log харахыг хүсвэл True болго
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─────────────────────────────────────────────────────────────────────────────
# User Model (MySQL Table)
# ─────────────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username   = Column(String(50), unique=True, nullable=False, index=True)
    email      = Column(String(255), unique=True, nullable=False, index=True)
    full_name  = Column(String(100), nullable=True)
    first_name = Column(String(100), nullable=True)
    last_name  = Column(String(100), nullable=True)
    phone      = Column(String(20),  nullable=True)
    birth_date = Column(String(20),  nullable=True)
    hashed_password = Column(String(255), nullable=False)
    level      = Column(Integer, default=1, nullable=False)
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<User id={self.id} username={self.username}>"


# ─────────────────────────────────────────────────────────────────────────────
# DB Session Dependency (FastAPI-д inject хийхэд ашиглана)
# ─────────────────────────────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Бүх таблуудыг үүсгэнэ (эхний ажиллуулалтад)"""
    Base.metadata.create_all(bind=engine)