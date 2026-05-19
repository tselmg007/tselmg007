from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
import bcrypt
import os

# ─────────────────────────────────────────────────────────────────────────────
# JWT Тохиргоо — .env файлд хадгалахыг зөвлөнө
# ─────────────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_THIS_TO_A_LONG_RANDOM_SECRET_KEY_IN_PRODUCTION")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# ─────────────────────────────────────────────────────────────────────────────
# Password Hashing
# ─────────────────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# JWT Token
# ─────────────────────────────────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username:   str            = Field(..., min_length=3, max_length=50, example="boldoo")
    email:      EmailStr       = Field(...,                               example="boldoo@example.mn")
    password:   str            = Field(..., min_length=6,                 example="SecurePass123")
    firstName:  Optional[str]  = Field(None, max_length=100,              example="Болд")
    lastName:   Optional[str]  = Field(None, max_length=100,              example="Баатар")
    phone:      Optional[str]  = Field(None, max_length=20,               example="99001122")
    birthDate:  Optional[str]  = Field(None,                              example="2000-01-01")

class LoginRequest(BaseModel):
    username_or_email: str  = Field(..., example="boldoo")
    password:          str  = Field(..., example="SecurePass123")

class TokenResponse(BaseModel):
    token:      str
    id:         int
    username:   str
    email:      str
    firstName:  str
    lastName:   str
    phone:      str
    birthDate:  str
    type:       str = "Bearer"
    level:      int

class UserResponse(BaseModel):
    id:         int
    username:   str
    email:      str
    full_name:  Optional[str]
    level:      int
    is_active:  bool
    created_at: datetime

    class Config:
        from_attributes = True  # SQLAlchemy model-оос шууд унших

class RegisterResponse(BaseModel):
    message: str
    user:    UserResponse

class GoogleLoginRequest(BaseModel):
    id_token: str

class UpdateProfileRequest(BaseModel):
    birthDate:  Optional[str]  = Field(None, example="2000-01-01")
    phone:      Optional[str]  = Field(None, max_length=20)
    firstName:  Optional[str]  = Field(None, max_length=100)
    lastName:   Optional[str]  = Field(None, max_length=100)

class ProfileResponse(BaseModel):
    id:         int
    email:      str
    firstName:  str
    lastName:   str
    phone:      str
    birthDate:  str
    level:      int