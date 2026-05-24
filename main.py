"""
Guitar Skill Level Analyzer — Pure Local Backend v4
====================================================
Бүх шинжилгээ librosa + numpy + дүрмэд суурилсан алгоритмаар
серверийн дотоодод гүйцэтгэгдэнэ.
"""
import io
import os
import shutil
import subprocess
import tempfile
import warnings
import smtplib
import random
import string
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel

import requests as http_requests

from db import get_db, create_tables, User, PracticeSession, LevelHistory
from auth import (
    hash_password, verify_password,
    create_access_token, decode_access_token,
    RegisterRequest, LoginRequest, TokenResponse,
    UserResponse, RegisterResponse,
    UpdateProfileRequest, ProfileResponse,
    GoogleLoginRequest, ForgotPasswordRequest, SetPasswordRequest,
    VerifyCodeRequest, ResetPasswordRequest,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Guitar Skill Analyzer",
    description="Бүх шинжилгээ librosa + алгоритмаар. User auth багтсан.",
    version="4.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.on_event("startup")
def startup():
    create_tables()
    ffmpeg = _get_ffmpeg()
    print(f"[STARTUP] ffmpeg path: {ffmpeg}")

MAX_FILE_MB = 50
SUPPORTED_CT = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mp4", "audio/m4a", "audio/ogg", "audio/webm",
    "audio/flac", "audio/x-flac", "application/octet-stream",
    "audio/aac",
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
LEVEL_LABEL = {1: "Beginner", 2: "Intermediate", 3: "Advanced", 4: "Professional"}


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL — Нууц үг сэргээх код илгээх
# ─────────────────────────────────────────────────────────────────────────────
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")        # код илгээх Gmail хаяг
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")  # Gmail App Password

# Баталгаажуулах кодуудыг түр хадгалах (memory)
# Бүтэц: { "email": {"code": "123456", "expires": datetime} }
_reset_codes: dict = {}


def _generate_code() -> str:
    """6 оронтой санамсаргүй код үүсгэнэ"""
    return "".join(random.choices(string.digits, k=6))


def _send_reset_email(to_email: str, code: str) -> bool:
    """Хэрэглэгчийн email рүү баталгаажуулах код илгээнэ"""
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print("[EMAIL] АЛДАА: SMTP_EMAIL эсвэл SMTP_PASSWORD тохируулагдаагүй")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_EMAIL
        msg["To"] = to_email
        msg["Subject"] = "Guitar App — Нууц үг сэргээх код"

        body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto;">
            <h2 style="color: #6C5CE7;">Guitar Skill Analyzer</h2>
            <p>Сайн байна уу,</p>
            <p>Таны нууц үг сэргээх код:</p>
            <div style="background: #f4f4f8; border-radius: 12px; padding: 20px;
                        text-align: center; margin: 20px 0;">
                <span style="font-size: 32px; font-weight: bold; letter-spacing: 8px;
                             color: #6C5CE7;">{code}</span>
            </div>
            <p style="color: #888; font-size: 13px;">
                Энэ код 10 минутын дараа хүчингүй болно.<br>
                Хэрэв та энэ хүсэлтийг илгээгээгүй бол энэ имэйлийг үл тоомсорлоно уу.
            </p>
        </div>
        """
        msg.attach(MIMEText(body, "html"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"[EMAIL] Код амжилттай илгээгдлээ → {to_email}")
        return True

    except Exception as exc:
        print(f"[EMAIL] Илгээхэд алдаа гарлаа: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Auth Helper
# ─────────────────────────────────────────────────────────────────────────────
def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Токен хүчингүй эсвэл дууссан.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if not auth_header:
        raise credentials_exception
    token = auth_header.removeprefix("Bearer ").removeprefix("bearer ").strip()
    if not token:
        raise credentials_exception
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    user_id: int = payload.get("sub")
    if user_id is None:
        raise credentials_exception
    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None or not user.is_active:
        raise credentials_exception
    return user


# ─────────────────────────────────────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=TokenResponse,
          status_code=status.HTTP_201_CREATED,
          summary="Шинэ хэрэглэгч бүртгэх", tags=["Auth"])
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"'{req.username}' username аль хэдийн бүртгэлтэй.")
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"'{req.email}' email аль хэдийн бүртгэлтэй.")
    new_user = User(
        username=req.username, email=req.email,
        first_name=req.firstName, last_name=req.lastName,
        phone=req.phone, birth_date=req.birthDate,
        hashed_password=hash_password(req.password),
    )
    db.add(new_user); db.commit(); db.refresh(new_user)
    token = create_access_token(data={"sub": str(new_user.id)},
                                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return TokenResponse(token=token, id=new_user.id, username=new_user.username,
                         email=new_user.email, firstName=new_user.first_name or "",
                         lastName=new_user.last_name or "", phone=new_user.phone or "",
                         birthDate=new_user.birth_date or "", level=new_user.level)


@app.post("/auth/login", response_model=TokenResponse,
          summary="Нэвтрэх (JWT токен авах)", tags=["Auth"])
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = (db.query(User).filter(User.username == req.username_or_email).first()
            or db.query(User).filter(User.email == req.username_or_email).first())
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Нэвтрэх нэр эсвэл нууц үг буруу.",
                            headers={"WWW-Authenticate": "Bearer"})
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Таны бүртгэл идэвхгүй байна.")
    token = create_access_token(data={"sub": str(user.id)},
                                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return TokenResponse(token=token, id=user.id, username=user.username,
                         email=user.email, firstName=user.first_name or "",
                         lastName=user.last_name or "", phone=user.phone or "",
                         birthDate=user.birth_date or "", level=user.level)


@app.get("/auth/me", response_model=UserResponse,
         summary="Одоогийн хэрэглэгчийн мэдээлэл", tags=["Auth"])
def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse.from_orm(current_user)


@app.post("/auth/google", response_model=TokenResponse,
          summary="Google-ээр нэвтрэх", tags=["Auth"])
def google_login(req: GoogleLoginRequest, db: Session = Depends(get_db)):
    resp = http_requests.get("https://oauth2.googleapis.com/tokeninfo",
                             params={"id_token": req.id_token}, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Google токен хүчингүй.")
    info = resp.json()
    email = info.get("email")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Email авч чадсангүй.")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        username = email.split("@")[0]
        base, i = username, 1
        while db.query(User).filter(User.username == username).first():
            username = f"{base}{i}"; i += 1
        user = User(username=username, email=email,
                    first_name=info.get("given_name", ""),
                    last_name=info.get("family_name", ""),
                    hashed_password=hash_password(info.get("sub", "")))
        db.add(user); db.commit(); db.refresh(user)
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Бүртгэл идэвхгүй.")
    token = create_access_token(data={"sub": str(user.id)},
                                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return TokenResponse(token=token, id=user.id, username=user.username,
                         email=user.email, firstName=user.first_name or "",
                         lastName=user.last_name or "", phone=user.phone or "",
                         birthDate=user.birth_date or "", level=user.level)


@app.post("/auth/set-password",
          summary="Нууц үг тавих (Google бүртгэлд)", tags=["Auth"])
def set_password(
    req: SetPasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current_user.hashed_password = hash_password(req.new_password)
    db.commit()
    return {"message": "Нууц үг амжилттай тохируулагдлаа."}


# ── FORGOT PASSWORD — Email-ээр код илгээх 3 алхам ──────────────────────────

@app.post("/auth/forgot-password",
          summary="Нууц үг мартсан — код илгээх", tags=["Auth"])
def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Алхам 1: Хэрэглэгчийн email рүү 6 оронтой баталгаажуулах код илгээнэ"""
    user = db.query(User).filter(User.email == req.email).first()

    # Аюулгүй байдлын үүднээс — хэрэглэгч байхгүй ч "илгээсэн" гэж хариулна
    if not user:
        return {"message": "Хэрэв энэ email бүртгэлтэй бол код илгээгдсэн."}

    code = _generate_code()
    _reset_codes[req.email] = {
        "code": code,
        "expires": datetime.utcnow() + timedelta(minutes=10),
    }

    sent = _send_reset_email(req.email, code)
    if not sent:
        if SMTP_EMAIL and SMTP_PASSWORD:
            # SMTP тохируулагдсан ч илгээхэд алдаа гарсан
            raise HTTPException(
                status_code=500,
                detail="Email илгээхэд алдаа гарлаа. Дараа дахин оролдоно уу.",
            )
        # SMTP тохируулагдаагүй үед (dev) — кодыг log-д хэвлэж, амжилттай хариулна
        print(f"[EMAIL-DEV] {req.email} → код: {code}  (SMTP тохируулаагүй)")

    return {"message": "Баталгаажуулах код таны email рүү илгээгдлээ."}


@app.post("/auth/verify-code",
          summary="Баталгаажуулах код шалгах", tags=["Auth"])
def verify_code(req: VerifyCodeRequest):
    """Алхам 2: Хэрэглэгчийн оруулсан код зөв эсэхийг шалгана"""
    entry = _reset_codes.get(req.email)

    if not entry:
        raise HTTPException(status_code=400,
                            detail="Код олдсонгүй. Дахин код авна уу.")

    if datetime.utcnow() > entry["expires"]:
        del _reset_codes[req.email]
        raise HTTPException(status_code=400,
                            detail="Кодын хугацаа дууссан. Дахин код авна уу.")

    if req.code != entry["code"]:
        raise HTTPException(status_code=400, detail="Код буруу байна.")

    return {"message": "Код зөв байна.", "verified": True}


@app.post("/auth/reset-password",
          summary="Шинэ нууц үг тавих", tags=["Auth"])
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Алхам 3: Код баталгаажуулсны дараа шинэ нууц үг хадгална"""
    entry = _reset_codes.get(req.email)

    if not entry:
        raise HTTPException(status_code=400,
                            detail="Код олдсонгүй. Дахин код авна уу.")

    if datetime.utcnow() > entry["expires"]:
        del _reset_codes[req.email]
        raise HTTPException(status_code=400,
                            detail="Кодын хугацаа дууссан. Дахин код авна уу.")

    if req.code != entry["code"]:
        raise HTTPException(status_code=400, detail="Код буруу байна.")

    user = db.query(User).filter(User.email == req.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Хэрэглэгч олдсонгүй.")

    user.hashed_password = hash_password(req.new_password)
    db.commit()

    del _reset_codes[req.email]

    return {"message": "Нууц үг амжилттай шинэчлэгдлээ."}


@app.patch("/users/me", response_model=ProfileResponse,
           summary="Профайл мэдээлэл шинэчлэх", tags=["Auth"])
def update_me(req: UpdateProfileRequest,
              current_user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    if req.birthDate is not None: current_user.birth_date = req.birthDate
    if req.phone is not None: current_user.phone = req.phone
    if req.firstName is not None: current_user.first_name = req.firstName
    if req.lastName is not None: current_user.last_name = req.lastName
    db.commit(); db.refresh(current_user)
    return ProfileResponse(id=current_user.id, email=current_user.email,
                           firstName=current_user.first_name or "",
                           lastName=current_user.last_name or "",
                           phone=current_user.phone or "",
                           birthDate=current_user.birth_date or "",
                           level=current_user.level)


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE STATS ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/users/me/stats", summary="Хэрэглэгчийн нийт статистик", tags=["Profile"])
def get_my_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sessions = db.query(PracticeSession).filter(
        PracticeSession.user_id == current_user.id
    ).all()

    total_sessions = len(sessions)
    total_seconds  = sum(s.duration_seconds or 0 for s in sessions)
    total_minutes  = round(total_seconds / 60, 1)

    scores = [s.score for s in sessions if s.score is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0

    days_since_register = (datetime.utcnow() - current_user.created_at).days if current_user.created_at else 0

    week_ago = datetime.utcnow() - timedelta(days=7)
    weekly_sessions = sum(1 for s in sessions if s.created_at and s.created_at >= week_ago)

    return {
        "total_sessions":       total_sessions,
        "total_minutes":        total_minutes,
        "avg_score":            avg_score,
        "current_level":        current_user.level,
        "current_level_label":  LEVEL_LABEL.get(current_user.level, "Beginner"),
        "days_using_app":       days_since_register,
        "weekly_sessions":      weekly_sessions,
        "member_since":         current_user.created_at.isoformat() if current_user.created_at else None,
    }


@app.get("/users/me/history", summary="Дадлагын түүх (сүүлийн 20)", tags=["Profile"])
def get_my_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sessions = (
        db.query(PracticeSession)
        .filter(PracticeSession.user_id == current_user.id)
        .order_by(PracticeSession.created_at.desc())
        .limit(20)
        .all()
    )
    return {
        "history": [
            {
                "id":               s.id,
                "date":             s.created_at.isoformat() if s.created_at else None,
                "score":            s.score,
                "level":            s.level,
                "duration_seconds": s.duration_seconds,
                "session_type":     s.session_type,
            }
            for s in sessions
        ]
    }


@app.get("/users/me/level-history", summary="Түвшний өөрчлөлтийн түүх", tags=["Profile"])
def get_level_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    history = (
        db.query(LevelHistory)
        .filter(LevelHistory.user_id == current_user.id)
        .order_by(LevelHistory.changed_at.asc())
        .all()
    )
    return {
        "current_level":       current_user.level,
        "current_level_label": LEVEL_LABEL.get(current_user.level, "Beginner"),
        "level_history": [
            {
                "date":       h.changed_at.isoformat() if h.changed_at else None,
                "from_level": h.from_level,
                "to_level":   h.to_level,
                "from_label": h.from_label,
                "to_label":   h.to_label,
            }
            for h in history
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# SONGS / ASSESSMENT ROUTES
# ─────────────────────────────────────────────────────────────────────────────

SONGS = [
    {"id": 1, "title": "Riptide", "artist": "Vance Joy", "bpm": 93, "difficulty": "Beginner"},
    {"id": 2, "title": "As Long As You Love Me", "artist": "Justin Bieber", "bpm": 100, "difficulty": "Intermediate"},
]


@app.get("/songs/assessment", summary="Хэрэглэгчийн түвшин болон дуунуудын жагсаалт",
         tags=["Songs"])
def get_assessment(request: Request, db: Session = Depends(get_db)):
    user_level = None
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    token = auth.removeprefix("Bearer ").removeprefix("bearer ").strip() if auth else None
    if token:
        payload = decode_access_token(token)
        if payload:
            user_id = payload.get("sub")
            if user_id:
                user = db.query(User).filter(User.id == int(user_id)).first()
                if user:
                    user_level = {"level": user.level, "label": LEVEL_LABEL.get(user.level, "Beginner")}
    return {"user_level": user_level, "songs": SONGS}


# ─────────────────────────────────────────────────────────────────────────────
# GUITAR ANALYSIS ROUTE
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_MAP = {"Beginner": 1, "Intermediate": 2, "Advanced": 3, "Professional": 4}


@app.post("/analyze", summary="Guitar audio шинжилгээ", tags=["Analysis"])
async def analyze_guitar(
    request: Request,
    audio: UploadFile = File(..., description="Guitar audio файл (wav/mp3/m4a/flac/ogg)"),
    db: Session = Depends(get_db),
):
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    token = auth.removeprefix("Bearer ").removeprefix("bearer ").strip() if auth else None
    ct = (audio.content_type or "").lower().split(";")[0].strip()
    print(f"[ANALYZE] content_type={ct!r} filename={audio.filename!r}")
    if ct not in SUPPORTED_CT:
        raise HTTPException(status_code=415, detail=f"Дэмжигдэхгүй төрөл: '{ct}'.")
    audio_bytes = await audio.read()
    size_mb = len(audio_bytes) / (1024 * 1024)
    print(f"[ANALYZE] size={len(audio_bytes)} bytes ({size_mb:.2f} MB)")
    if size_mb > MAX_FILE_MB:
        raise HTTPException(status_code=413,
                            detail=f"Файл хэт том: {size_mb:.1f} MB. Хамгийн их {MAX_FILE_MB} MB.")
    if len(audio_bytes) < 2000:
        raise HTTPException(status_code=400, detail="Файл хэт жижиг эсвэл хоосон.")
    try:
        result = analyze_pipeline(audio_bytes)
    except Exception as exc:
        print(f"[ANALYZE] ERROR: {exc}")
        raise HTTPException(status_code=500, detail=f"Шинжилгээний алдаа: {exc}")

    if token:
        payload = decode_access_token(token)
        if payload:
            user_id = payload.get("sub")
            if user_id:
                user = db.query(User).filter(User.id == int(user_id)).first()
                if user:
                    new_level = LEVEL_MAP.get(result.skill_level, 1)

                    if user.level != new_level:
                        history = LevelHistory(
                            user_id    = user.id,
                            from_level = user.level,
                            to_level   = new_level,
                            from_label = LEVEL_LABEL.get(user.level, "Beginner"),
                            to_label   = LEVEL_LABEL.get(new_level, "Beginner"),
                        )
                        db.add(history)

                    session = PracticeSession(
                        user_id          = user.id,
                        duration_seconds = int(result.duration_seconds),
                        score            = result.skill_score,
                        level            = result.skill_level,
                        session_type     = "analyze",
                    )
                    db.add(session)

                    user.level = new_level
                    db.commit()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CHORD DETECTION ROUTE
# ─────────────────────────────────────────────────────────────────────────────

def _build_chord_profiles() -> dict:
    profiles = {}
    interval_map = {
        "":     [(0, 2.0), (4, 0.8), (7, 1.2)],
        "m":    [(0, 2.0), (3, 0.8), (7, 1.2)],
        "7":    [(0, 1.5), (4, 0.8), (7, 1.0), (10, 1.5)],
        "maj7": [(0, 1.5), (4, 0.8), (7, 1.0), (11, 1.5)],
        "min7": [(0, 1.5), (3, 0.8), (7, 1.0), (10, 1.5)],
        "sus2": [(0, 2.0), (2, 1.0), (7, 1.2)],
        "sus4": [(0, 2.0), (5, 1.0), (7, 1.2)],
        "dim":  [(0, 2.0), (3, 0.8), (6, 1.0)],
        "aug":  [(0, 2.0), (4, 0.8), (8, 1.0)],
    }
    for suffix, weighted_intervals in interval_map.items():
        for i, root in enumerate(NOTE_NAMES):
            profile = np.zeros(12)
            for interval, weight in weighted_intervals:
                profile[(i + interval) % 12] = weight
            profile /= profile.sum() + 1e-9
            profiles[f"{root}{suffix}"] = profile
    return profiles

CHORD_PROFILES = _build_chord_profiles()


@app.post("/chords/detect", summary="Гитарын chord таних", tags=["Chords"])
async def detect_chord(
    audio: UploadFile = File(..., description="Audio файл (m4a/wav/mp3)"),
):
    ct = (audio.content_type or "").lower().split(";")[0].strip()
    print(f"[CHORD] content_type={ct!r} filename={audio.filename!r}")
    audio_bytes = await audio.read()
    if len(audio_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Файл хэт жижиг.")
    try:
        f = extract_features(audio_bytes)
        chroma = np.array(f.chroma_vector)
        best_chord = "Unknown"
        best_score = -1.0
        for chord_name, profile in CHORD_PROFILES.items():
            score = _cosine(chroma, profile)
            root_name = None
            for n in ["C#", "D#", "F#", "G#", "A#", "C", "D", "E", "F", "G", "A", "B"]:
                if chord_name.startswith(n):
                    root_name = n
                    break
            if root_name and root_name in NOTE_NAMES:
                root_idx = NOTE_NAMES.index(root_name)
                root_energy = float(chroma[root_idx])
                score = score * (1.0 + root_energy * 0.5)
            if score > best_score:
                best_score = score
                best_chord = chord_name
        confidence = round(float(best_score) * 100, 1)
        print(f"[CHORD] detected={best_chord} confidence={confidence}%")
        return {
            "chord": best_chord,
            "chord_name": best_chord,
            "confidence": confidence,
            "dominant_note": f.dominant_note,
            "tempo_bpm": f.tempo_bpm,
        }
    except Exception as exc:
        print(f"[CHORD] ERROR: {exc}")
        raise HTTPException(status_code=500, detail=f"Chord detection алдаа: {exc}")


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "4.0.0", "mode": "fully-local"}


@app.post("/debug", summary="Chroma + similarity debug", tags=["System"])
async def debug():
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Response загвар
# ─────────────────────────────────────────────────────────────────────────────
class SkillAnalysis(BaseModel):
    skill_level: str
    skill_score: int
    song_detected: str
    duration_seconds: float
    tempo_bpm: float
    rhythm_score: int
    dynamics_score: int
    clarity_score: int
    consistency_score: int
    tempo_feel: str
    rhythm_accuracy: str
    chord_clarity: str
    dynamic_range: str
    strengths: list[str]
    areas_to_improve: list[str]
    practice_tips: list[str]
    overall_feedback: str


# ─────────────────────────────────────────────────────────────────────────────
# 1. RAW FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RawFeatures:
    duration_sec: float
    sr: int
    tempo_bpm: float
    rhythm_stability: float
    onset_regularity: float
    onset_rate: float
    dynamic_range_db: float
    rms_std_db: float
    silence_ratio: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    zero_crossing_rate: float
    dominant_note: str
    active_note_count: int
    chroma_entropy: float
    mfcc_variance: float
    chroma_vector: list[float] = None


def _detect_format(header: bytes) -> str:
    if header[:4] == b'RIFF': return 'wav'
    if header[:3] == b'ID3' or header[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'): return 'mp3'
    if header[:2] in (b'\xff\xf1', b'\xff\xf9'): return 'aac'
    if header[4:8] in (b'ftyp', b'moov') or header[:4] in (b'\x00\x00\x00\x18', b'\x00\x00\x00\x20'): return 'm4a'
    if header[:4] == b'OggS': return 'ogg'
    if header[:4] == b'fLaC': return 'flac'
    return 'wav'


def _get_ffmpeg():
    path = shutil.which("ffmpeg")
    if path: return path
    local = os.path.join(os.path.dirname(__file__), "ffmpeg",
                         "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe")
    if os.path.isfile(local): return local
    try:
        result = subprocess.run(["find", "/nix/store", "-name", "ffmpeg", "-type", "f"],
                                capture_output=True, text=True, timeout=10)
        paths = [p for p in result.stdout.strip().split("\n") if p and "/bin/ffmpeg" in p]
        if paths: return paths[0]
    except Exception:
        pass
    return None

_FFMPEG = _get_ffmpeg()
print(f"[FFMPEG] path={_FFMPEG}")


def _load_audio(audio_bytes: bytes):
    fmt = _detect_format(audio_bytes[:16])
    print(f"[LOAD] detected format={fmt}")
    if _FFMPEG:
        tmp_in = tmp_out = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
                f.write(audio_bytes); tmp_in = f.name
            tmp_out = tmp_in + "_out.wav"
            result = subprocess.run(
                [_FFMPEG, "-y", "-i", tmp_in, "-ar", "22050", "-ac", "1", "-f", "wav", tmp_out],
                capture_output=True)
            if result.returncode == 0:
                y, sr = librosa.load(tmp_out, mono=True, sr=16000, duration=120)
                print(f"[LOAD] ffmpeg OK sr={sr}")
                return y, sr
            print(f"[LOAD] ffmpeg error: {result.stderr[-200:].decode(errors='ignore')}")
        except Exception as e:
            print(f"[LOAD] ffmpeg failed: {e}")
        finally:
            for p in [tmp_in, tmp_out]:
                if p and os.path.exists(p): os.unlink(p)
    try:
        import soundfile as sf
        y, sr = sf.read(io.BytesIO(audio_bytes))
        y = y.mean(axis=1) if y.ndim > 1 else y
        print(f"[LOAD] soundfile OK sr={sr}")
        return y, sr
    except Exception as e:
        print(f"[LOAD] soundfile failed: {e}")
    y, sr = librosa.load(io.BytesIO(audio_bytes), mono=True, sr=16000, duration=120)
    print(f"[LOAD] librosa OK sr={sr}")
    return y, sr


def extract_features(audio_bytes: bytes) -> RawFeatures:
    y, sr = _load_audio(audio_bytes)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo_arr, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo_arr)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    if len(beat_times) > 2:
        ivs = np.diff(beat_times)
        rhythm_stability = float(1.0 - min(np.std(ivs) / (np.mean(ivs) + 1e-6), 1.0))
    else:
        rhythm_stability = 0.4

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_rate = len(onset_times) / max(duration, 1.0)
    if len(onset_times) > 2:
        gaps = np.diff(onset_times)
        onset_regularity = float(1.0 - min(np.std(gaps) / (np.mean(gaps) + 1e-6), 1.0))
    else:
        onset_regularity = 0.3

    rms = librosa.feature.rms(y=y)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    dynamic_range_db = float(np.max(db) - np.min(db))
    rms_std_db = float(np.std(db))
    silence_ratio = float(np.mean(rms < 0.005))

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    note_energy = {NOTE_NAMES[i]: float(np.mean(chroma[i])) for i in range(12)}
    dominant_note = max(note_energy, key=note_energy.get)
    active_notes = sum(1 for v in note_energy.values() if v > 0.25)
    vals = np.array(list(note_energy.values())) + 1e-9
    vals /= vals.sum()
    chroma_entropy = float(-np.sum(vals * np.log2(vals)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_variance = float(np.mean(np.var(mfcc, axis=1)))

    chroma_mean = np.array([float(np.mean(chroma[i])) for i in range(12)])
    norm = chroma_mean.sum() + 1e-9
    chroma_vector = [round(float(v / norm), 4) for v in chroma_mean]

    return RawFeatures(
        duration_sec=round(duration, 1), sr=int(sr),
        tempo_bpm=round(tempo_bpm, 1),
        rhythm_stability=round(rhythm_stability, 3),
        onset_regularity=round(onset_regularity, 3),
        onset_rate=round(onset_rate, 2),
        dynamic_range_db=round(dynamic_range_db, 2),
        rms_std_db=round(rms_std_db, 2),
        silence_ratio=round(silence_ratio, 3),
        spectral_centroid_hz=round(float(np.mean(sc)), 1),
        spectral_bandwidth_hz=round(float(np.mean(sbw)), 1),
        zero_crossing_rate=round(float(np.mean(zcr)), 5),
        dominant_note=dominant_note,
        active_note_count=active_notes,
        chroma_entropy=round(chroma_entropy, 3),
        mfcc_variance=round(mfcc_variance, 2),
        chroma_vector=chroma_vector,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. RULE-BASED SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> int:
    return int(max(lo, min(hi, val)))

def score_rhythm(f: RawFeatures) -> int:
    base = (f.rhythm_stability * 0.6 + f.onset_regularity * 0.4) * 100
    tempo_penalty = 15.0 if f.tempo_bpm < 40 or f.tempo_bpm > 220 else 0.0
    return clamp(base - tempo_penalty)

def score_dynamics(f: RawFeatures) -> int:
    dr = f.dynamic_range_db
    if dr < 5: range_score = 20.0
    elif dr < 15: range_score = 40.0 + (dr - 5) * 3.0
    elif dr < 40: range_score = 70.0 + (dr - 15) * 0.8
    else: range_score = 90.0
    return clamp(range_score - min(f.silence_ratio * 80, 30.0))

def score_clarity(f: RawFeatures) -> int:
    if f.active_note_count < 2: note_score = 20.0
    elif f.active_note_count <= 6: note_score = 50.0 + f.active_note_count * 8.0
    else: note_score = max(30.0, 98.0 - (f.active_note_count - 6) * 10)
    ent = f.chroma_entropy
    if ent < 1.5: ent_score = 30.0
    elif ent < 2.5: ent_score = 50.0 + (ent - 1.5) * 20.0
    elif ent <= 3.3: ent_score = 70.0 + (ent - 2.5) * 25.0
    else: ent_score = 85.0
    return clamp(note_score * 0.5 + ent_score * 0.5)

def score_consistency(f: RawFeatures) -> int:
    reg_score = f.onset_regularity * 100
    rate = f.onset_rate
    if rate < 1.0: rate_bonus = -20.0
    elif rate <= 6.0: rate_bonus = 10.0
    elif rate <= 10.0: rate_bonus = 0.0
    else: rate_bonus = -15.0
    return clamp(reg_score + rate_bonus)


def _chord_to_chroma(notes: list[str]) -> np.ndarray:
    NOTE_IDX = {n: i for i, n in enumerate(NOTE_NAMES)}
    vec = np.zeros(12)
    for n in notes: vec[NOTE_IDX[n]] += 1.0
    return vec / (vec.sum() + 1e-9)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


RIPTIDE_PROFILE = (
    _chord_to_chroma(["A","C","E"]) * 0.40 +
    _chord_to_chroma(["G","B","D"]) * 0.35 +
    _chord_to_chroma(["C","E","G"]) * 0.20 +
    _chord_to_chroma(["F","A","C","E"]) * 0.05
)
RIPTIDE_PROFILE /= RIPTIDE_PROFILE.sum() + 1e-9

ALAYLM_PROFILE = (
    _chord_to_chroma(["B","D","F#"]) * 0.30 +
    _chord_to_chroma(["A","C#","E"]) * 0.25 +
    _chord_to_chroma(["D","F#","A"]) * 0.30 +
    _chord_to_chroma(["G","B","D"])  * 0.15
)
ALAYLM_PROFILE /= ALAYLM_PROFILE.sum() + 1e-9

RIPTIDE_BPM = 93.0
ALAYLM_BPM  = 100.0

def _bpm_score(detected: float, target: float, tol: float = 12.0) -> float:
    for mult in (1.0, 0.5, 2.0):
        if abs(detected - target * mult) <= tol:
            return 1.0 - abs(detected - target * mult) / tol
    return 0.0

def detect_song(f: RawFeatures) -> str:
    if not f.chroma_vector: return "Тодорхойгүй"
    cv = np.array(f.chroma_vector)
    r_c = _cosine(cv, RIPTIDE_PROFILE); a_c = _cosine(cv, ALAYLM_PROFILE)
    r_b = _bpm_score(f.tempo_bpm, RIPTIDE_BPM); a_b = _bpm_score(f.tempo_bpm, ALAYLM_BPM)
    r_t = r_c * 0.70 + r_b * 0.30; a_t = a_c * 0.70 + a_b * 0.30
    if max(r_t, a_t) < 0.45: return "Тодорхойгүй"
    return "Riptide — Vance Joy" if r_t >= a_t else "As Long As You Love Me — Justin Bieber"

def level_from_score(score: int) -> str:
    if score >= 82: return "Professional"
    if score >= 62: return "Advanced"
    if score >= 38: return "Intermediate"
    return "Beginner"


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEEDBACK GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def make_feedback(f: RawFeatures, r: int, d: int, cl: int, co: int, total: int) -> dict:
    level = level_from_score(total)
    song  = detect_song(f)
    bpm   = f.tempo_bpm

    if bpm < 60: tempo_feel = f"Удаан ({bpm:.0f} BPM)"
    elif bpm < 80: tempo_feel = f"Зөөлөн удаан ({bpm:.0f} BPM)"
    elif bpm < 100: tempo_feel = f"Дундаж хурд ({bpm:.0f} BPM)"
    elif bpm < 130: tempo_feel = f"Хурдан ({bpm:.0f} BPM)"
    else: tempo_feel = f"Маш хурдан ({bpm:.0f} BPM)"

    if r >= 80: rhythm_accuracy = "Маш сайн — хэмнэл тогтвортой, найдвартай"
    elif r >= 60: rhythm_accuracy = "Сайн — жижиг хэлбэлзэл байгаа ч тогтвортой"
    elif r >= 40: rhythm_accuracy = "Дундаж — хэмнэл тогтворгүй хэсгүүд бий"
    else: rhythm_accuracy = "Хангалтгүй — хэмнэл ихээхэн хэлбэлзэлтэй"

    if cl >= 75: chord_clarity = "Маш тод — аккорд цэвэр дарагдаж байна"
    elif cl >= 55: chord_clarity = "Тод — ерөнхийдөө тодорхой, бага зэрэг бузрал бий"
    elif cl >= 35: chord_clarity = "Бүдэг — аккорд бузартсан хэсгүүд байна"
    else: chord_clarity = "Тодорхойгүй — аккорд нарийвчлал хэрэгтэй"

    if d >= 75: dynamic_range = f"Өргөн ({f.dynamic_range_db:.0f} dB) — сайн динамик хяналт"
    elif d >= 50: dynamic_range = f"Дундаж ({f.dynamic_range_db:.0f} dB)"
    else: dynamic_range = f"Нарийн ({f.dynamic_range_db:.0f} dB) — динамик хяналт хэрэгтэй"

    strengths = []
    if r >= 65: strengths.append("Хэмнэл тогтвортой, beat алдагдахгүй тоглож байна")
    if d >= 65: strengths.append("Динамик хэлбэлзэл сайн — чанга/намхан хэсгийг ялган тоглодог")
    if cl >= 65: strengths.append("Аккорд тодорхой, string мулталт бага")
    if co >= 65: strengths.append("Цохилт тогтмол, strumming/picking жигд")
    if f.silence_ratio < 0.05: strengths.append("Тасралтгүй тоглолт — дуу урсгалтай")
    if not strengths: strengths.append("Тоглох хүсэл эрмэлзэл харагдаж байна — дадлага үргэлжлүүлнэ үү")

    areas = []
    if r < 60: areas.append("Хэмнэлийн тогтвортой байдал — метроном ашиглан дадлага хий")
    if cl < 55: areas.append("Аккордын дарлага — хуруу байрлал, дарах хүч нарийвчлах")
    if d < 50: areas.append("Динамик хяналт — нам ба чанга хэсгийг ухамсартайгаар ялгах")
    if co < 55: areas.append("Strumming/picking тогтмол байдал — хэв маяг дагаж дадлага хий")
    if f.silence_ratio > 0.15: areas.append("Аккорд шилжилтийн хурд — тасралтыг багасгах")
    if not areas: areas.append("Илүү нарийн техник (vibrato, fingerpicking) сурах")

    tips = []
    tips.append("Метроном 70 BPM-с эхлээд дуртай дуугаа тоглох дадлага хий." if r < 65
                else "Метроном ашиглаж байгаа бол BPM-ийг аажмаар ихэсгэж хурдыг нэмэгдүүл.")
    tips.append("Аккорд бүрийг тусад нь дарж, string бүр тод дуугарч байгааг шалга." if cl < 60
                else "Аккорд шилжилтийн хурдыг нэмэх — Am→F→C→G дараалалд timer тавьж дадла.")
    tips.append("Намхан болон чанга хэсгийг ухамсартайгаар тоглох дадлага хий." if d < 55
                else "Fingerpicking техник нэмж, ганц нотоор melody тоглох дадлагыг туршиж үз.")

    level_desc = {"Beginner": "Эхлэгч түвшний тоглогч.",
                  "Intermediate": "Дундаж түвшний тоглогч.",
                  "Advanced": "Дэвшилтэт түвшний тоглогч.",
                  "Professional": "Мэргэжлийн түвшний тоглогч."}[level]
    weak = []
    if r < 60: weak.append("хэмнэл")
    if cl < 55: weak.append("аккордын тодрол")
    if d < 50: weak.append("динамик")
    if co < 55: weak.append("цохилтын тогтвортой байдал")

    if weak:
        overall = (f"{level_desc} Нийт оноо {total}/100. "
                   f"{'、'.join(weak)}-ыг сайжруулбал дараагийн түвшинд гарна. "
                   f"Өдөр бүр 20-30 минут тогтмол дадлага хийх нь хамгийн үр дүнтэй.")
    else:
        overall = (f"{level_desc} Нийт оноо {total}/100. "
                   f"Бүх үзүүлэлт сайн байна — илүү нарийн техник сурах цаг боллоо.")

    return dict(skill_level=level, skill_score=total, song_detected=song,
                duration_seconds=f.duration_sec, tempo_bpm=f.tempo_bpm,
                tempo_feel=tempo_feel, rhythm_accuracy=rhythm_accuracy,
                chord_clarity=chord_clarity, dynamic_range=dynamic_range,
                strengths=strengths, areas_to_improve=areas,
                practice_tips=tips, overall_feedback=overall)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANALYZE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def analyze_pipeline(audio_bytes: bytes) -> SkillAnalysis:
    f  = extract_features(audio_bytes)
    r  = score_rhythm(f)
    d  = score_dynamics(f)
    cl = score_clarity(f)
    co = score_consistency(f)
    total = clamp(r * 0.40 + d * 0.15 + cl * 0.25 + co * 0.20)
    feedback = make_feedback(f, r, d, cl, co, total)
    feedback["rhythm_score"] = r; feedback["dynamics_score"] = d
    feedback["clarity_score"] = cl; feedback["consistency_score"] = co
    return SkillAnalysis(**feedback)


@app.get("/")
async def root():
    return {"name": "Guitar Skill Analyzer", "version": "4.0.0",
            "endpoints": {
                "POST /analyze":             "Бүрэн шинжилгээ",
                "POST /chords/detect":       "Chord таних (108 chord)",
                "POST /auth/forgot-password": "Нууц үг сэргээх код илгээх",
                "POST /auth/verify-code":    "Код шалгах",
                "POST /auth/reset-password": "Шинэ нууц үг тавих",
                "GET  /users/me/stats":      "Профайл статистик",
                "GET  /users/me/history":    "Дадлагын түүх",
                "GET  /users/me/level-history": "Түвшний өөрчлөлтийн түүх",
            },
            "formats": ["wav", "mp3", "m4a", "flac", "ogg", "webm"],
            "max_mb": MAX_FILE_MB, "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)