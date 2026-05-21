from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# MySQL — MYSQL_URL-аас шууд уншина (Railway-н хамгийн найдвартай арга)
# ─────────────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("MYSQL_URL")

if DATABASE_URL:
    # mysql:// → mysql+pymysql:// болгоно
    if DATABASE_URL.startswith("mysql://"):
        DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)
    print(f"[DB] Connecting → {DATABASE_URL.split('@')[-1]}", flush=True)

else:
    # DATABASE_URL байхгүй бол хувь хувийн variables-аас үүсгэнэ
    DB_HOST = os.getenv("DB_HOST") or os.getenv("MYSQLHOST")
    DB_PORT = os.getenv("DB_PORT") or os.getenv("MYSQLPORT", "3306")
    DB_NAME = os.getenv("DB_NAME") or os.getenv("MYSQLDATABASE")
    DB_USER = os.getenv("DB_USER") or os.getenv("MYSQLUSER")
    DB_PASS = os.getenv("DB_PASS") or os.getenv("MYSQLPASSWORD")

    missing = [k for k, v in {
        "DB_HOST": DB_HOST,
        "DB_NAME": DB_NAME,
        "DB_USER": DB_USER,
        "DB_PASS": DB_PASS,
    }.items() if not v]

    if missing:
        print(f"[DB] АЛДАА: Тохируулагдаагүй variables: {missing}", flush=True)
        print("[DB] Railway → Variables дээр DATABASE_URL=${{MySQL.MYSQL_URL}} нэмнэ үү.", flush=True)
        sys.exit(1)

    DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print(f"[DB] Connecting → {DB_HOST}:{DB_PORT}/{DB_NAME} (user={DB_USER})", flush=True)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─────────────────────────────────────────────────────────────────────────────
# User Model (MySQL Table)
# ─────────────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username   = Column(String(50),  unique=True, nullable=False, index=True)
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
    print("[DB] Таблуудыг амжилттай үүсгэлээ.", flush=True)