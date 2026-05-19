"""
Guitar Skill Level Analyzer — Pure Local Backend v3
====================================================
Бүх шинжилгээ librosa + numpy + дүрмэд суурилсан алгоритмаар
серверийн дотоодод гүйцэтгэгдэнэ.

Суулгах:
    pip install -r requirements.txt

Ажиллуулах:
    uvicorn main:app --reload --port 8000
"""
import io
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, asdict

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel

import requests as http_requests

from db import get_db, create_tables, User
from auth import (
    hash_password, verify_password,
    create_access_token, decode_access_token,
    RegisterRequest, LoginRequest, TokenResponse,
    UserResponse, RegisterResponse,
    UpdateProfileRequest, ProfileResponse,
    GoogleLoginRequest,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from datetime import timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Guitar Skill Analyzer — Local",
    description="Гадаад API-гүй, бүх шинжилгээ librosa + алгоритмаар. User auth багтсан.",
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

# DB таблуудыг эхлүүлэхэд үүсгэнэ
@app.on_event("startup")
def startup():
    create_tables()

MAX_FILE_MB = 50
SUPPORTED_CT = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mp4", "audio/m4a", "audio/ogg", "audio/webm",
    "audio/flac", "audio/x-flac", "application/octet-stream",
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


# ─────────────────────────────────────────────────────────────────────────────
# Auth Helper — одоогийн хэрэглэгчийг токеноос авна
# ─────────────────────────────────────────────────────────────────────────────
def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
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

@app.post(
    "/auth/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Шинэ хэрэглэгч бүртгэх",
    tags=["Auth"],
)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"'{req.username}' username аль хэдийн бүртгэлтэй.",
        )
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"'{req.email}' email аль хэдийн бүртгэлтэй.",
        )

    new_user = User(
        username=req.username,
        email=req.email,
        first_name=req.firstName,
        last_name=req.lastName,
        phone=req.phone,
        birth_date=req.birthDate,
        hashed_password=hash_password(req.password),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token(
        data={"sub": str(new_user.id)},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenResponse(
        token=token,
        id=new_user.id,
        username=new_user.username,
        email=new_user.email,
        firstName=new_user.first_name or "",
        lastName=new_user.last_name or "",
        phone=new_user.phone or "",
        birthDate=new_user.birth_date or "",
        level=new_user.level,
    )


@app.post(
    "/auth/login",
    response_model=TokenResponse,
    summary="Нэвтрэх (JWT токен авах)",
    tags=["Auth"],
)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    # Username эсвэл email-ээр хайх
    user = (
        db.query(User).filter(User.username == req.username_or_email).first()
        or db.query(User).filter(User.email == req.username_or_email).first()
    )

    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Нэвтрэх нэр эсвэл нууц үг буруу.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Таны бүртгэл идэвхгүй байна.",
        )

    token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenResponse(
        token=token,
        id=user.id,
        username=user.username,
        email=user.email,
        firstName=user.first_name or "",
        lastName=user.last_name or "",
        phone=user.phone or "",
        birthDate=user.birth_date or "",
        level=user.level,
    )


@app.get(
    "/auth/me",
    response_model=UserResponse,
    summary="Одоогийн хэрэглэгчийн мэдээлэл",
    tags=["Auth"],
)
def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse.from_orm(current_user)


@app.post(
    "/auth/google",
    response_model=TokenResponse,
    summary="Google-ээр нэвтрэх",
    tags=["Auth"],
)
def google_login(req: GoogleLoginRequest, db: Session = Depends(get_db)):
    # Google токеныг шалгана
    resp = http_requests.get(
        "https://oauth2.googleapis.com/tokeninfo",
        params={"id_token": req.id_token},
        timeout=10,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google токен хүчингүй.")

    info = resp.json()
    email = info.get("email")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Email авч чадсангүй.")

    # Хэрэглэгч байгаа эсэхийг шалгана, байхгүй бол үүсгэнэ
    user = db.query(User).filter(User.email == email).first()
    if not user:
        first_name = info.get("given_name", "")
        last_name  = info.get("family_name", "")
        username   = email.split("@")[0]
        # Username давхардал байвал дугаар нэмнэ
        base, i = username, 1
        while db.query(User).filter(User.username == username).first():
            username = f"{base}{i}"; i += 1
        user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
            hashed_password=hash_password(info.get("sub", "")),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Бүртгэл идэвхгүй.")

    token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenResponse(
        token=token,
        id=user.id,
        username=user.username,
        email=user.email,
        firstName=user.first_name or "",
        lastName=user.last_name or "",
        phone=user.phone or "",
        birthDate=user.birth_date or "",
        level=user.level,
    )


@app.patch(
    "/users/me",
    response_model=ProfileResponse,
    summary="Профайл мэдээлэл шинэчлэх",
    tags=["Auth"],
)
def update_me(
    req: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if req.birthDate is not None:
        current_user.birth_date = req.birthDate
    if req.phone is not None:
        current_user.phone = req.phone
    if req.firstName is not None:
        current_user.first_name = req.firstName
    if req.lastName is not None:
        current_user.last_name = req.lastName
    db.commit()
    db.refresh(current_user)
    return ProfileResponse(
        id=current_user.id,
        email=current_user.email,
        firstName=current_user.first_name or "",
        lastName=current_user.last_name or "",
        phone=current_user.phone or "",
        birthDate=current_user.birth_date or "",
        level=current_user.level,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SONGS / ASSESSMENT ROUTES
# ─────────────────────────────────────────────────────────────────────────────

SONGS = [
    {"id": 1, "title": "Riptide", "artist": "Vance Joy",        "bpm": 93,  "difficulty": "Beginner"},
    {"id": 2, "title": "As Long As You Love Me", "artist": "Justin Bieber", "bpm": 100, "difficulty": "Intermediate"},
]

LEVEL_LABEL = {1: "Beginner", 2: "Intermediate", 3: "Advanced", 4: "Professional"}


@app.get(
    "/songs/assessment",
    summary="Хэрэглэгчийн одоогийн түвшин болон шинжлэх дуунуудын жагсаалт",
    tags=["Songs"],
)
def get_assessment(
    request: Request,
    db: Session = Depends(get_db),
):
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

    return {
        "user_level": user_level,
        "songs": SONGS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GUITAR ANALYSIS ROUTES
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_MAP = {
    "Beginner":     1,
    "Intermediate": 2,
    "Advanced":     3,
    "Professional": 4,
}


@app.post(
    "/analyze",
    summary="Guitar audio шинжилгээ (100% дотоод)",
    description="Librosa + алгоритмаар гитарын ур чадварыг тодорхойлж хэрэглэгчийн түвшинг хадгална.",
    tags=["Analysis"],
)
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
        raise HTTPException(
            status_code=415,
            detail=f"Дэмжигдэхгүй төрөл: '{ct}'. wav/mp3/m4a/flac/ogg/webm.",
        )

    audio_bytes = await audio.read()
    size_mb = len(audio_bytes) / (1024 * 1024)
    print(f"[ANALYZE] size={len(audio_bytes)} bytes ({size_mb:.2f} MB)")

    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Файл хэт том: {size_mb:.1f} MB. Хамгийн их {MAX_FILE_MB} MB.",
        )
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
                new_level = LEVEL_MAP.get(result.skill_level, 1)
                db.query(User).filter(User.id == int(user_id)).update({"level": new_level})
                db.commit()

    return result


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "4.0.0", "mode": "fully-local"}


@app.post("/debug", summary="Chroma + similarity debug", tags=["System"])
async def debug():
    pass  # Таны одоогийн debug логик энд байна


# ─────────────────────────────────────────────────────────────────────────────
# Response загвар
# ─────────────────────────────────────────────────────────────────────────────
class SkillAnalysis(BaseModel):
    # ── Үндсэн үнэлгээ ───────────────────────────────────────────────────────
    skill_level: str          # Beginner / Intermediate / Advanced / Professional
    skill_score: int          # 0–100
    song_detected: str        # Илэрсэн дуу

    # ── Бичлэгийн мэдээлэл ───────────────────────────────────────────────────
    duration_seconds: float
    tempo_bpm: float

    # ── Дэд оноонууд (0–100) ─────────────────────────────────────────────────
    rhythm_score: int         # Хэмнэлийн тогтвортой байдал
    dynamics_score: int       # Динамик хяналт
    clarity_score: int        # Аккордын тодрол / бузрал
    consistency_score: int    # Тогтмол цохилт (onset regularity)

    # ── Тайлбар текстүүд ─────────────────────────────────────────────────────
    tempo_feel: str
    rhythm_accuracy: str
    chord_clarity: str
    dynamic_range: str

    # ── Дэлгэрэнгүй санал ────────────────────────────────────────────────────
    strengths: list[str]
    areas_to_improve: list[str]
    practice_tips: list[str]
    overall_feedback: str


# ─────────────────────────────────────────────────────────────────────────────
# 1. RAW FEATURE EXTRACTION  (librosa)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RawFeatures:
    duration_sec: float
    sr: int
    tempo_bpm: float
    rhythm_stability: float   # 0–1, 1 = төгс тогтвортой
    onset_regularity: float   # 0–1
    onset_rate: float         # onset/sec
    dynamic_range_db: float
    rms_std_db: float
    silence_ratio: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    zero_crossing_rate: float
    dominant_note: str
    active_note_count: int    # хэдэн нот идэвхтэй байна
    chroma_entropy: float     # нот тархалтын жигд байдал
    mfcc_variance: float      # тембрийн хэлбэлзэл
    # 12 нотын дундаж chroma эрчим (C C# D D# E F F# G G# A A# B)
    chroma_vector: list[float] = None


def _detect_format(header: bytes) -> str:
    """Файлын header-оор аудио форматыг тодорхойлно."""
    if header[:4] == b'RIFF':
        return 'wav'
    if header[:3] == b'ID3' or header[:2] == b'\xff\xfb' or header[:2] == b'\xff\xf3' or header[:2] == b'\xff\xf2':
        return 'mp3'
    if header[:2] in (b'\xff\xf1', b'\xff\xf9'):
        return 'aac'
    if header[4:8] in (b'ftyp', b'moov') or header[:4] in (b'\x00\x00\x00\x18', b'\x00\x00\x00\x20'):
        return 'm4a'
    if header[:4] == b'OggS':
        return 'ogg'
    if header[:4] == b'fLaC':
        return 'flac'
    return 'wav'


def _load_audio(audio_bytes: bytes):
    """librosa-р шууд уншина — ffmpeg байвал ашиглана."""
    fmt = _detect_format(audio_bytes[:16])
    print(f"[LOAD] detected format={fmt}")

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path:
        tmp_in = tmp_out = None
        try:
            suffix = f".{fmt}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_bytes)
                tmp_in = f.name
            tmp_out = tmp_in + "_out.wav"
            result = subprocess.run(
                [ffmpeg_path, "-y", "-i", tmp_in, "-ar", "22050", "-ac", "1", "-f", "wav", tmp_out],
                capture_output=True,
            )
            if result.returncode == 0:
                y, sr = librosa.load(tmp_out, mono=True, sr=None)
                print(f"[LOAD] ffmpeg OK sr={sr}")
                return y, sr
            print(f"[LOAD] ffmpeg error: {result.stderr[-200:].decode(errors='ignore')}")
        except Exception as e:
            print(f"[LOAD] ffmpeg failed: {e}")
        finally:
            for p in [tmp_in, tmp_out]:
                if p and os.path.exists(p):
                    os.unlink(p)

    # ffmpeg байхгүй бол librosa-р шууд уншина
    y, sr = librosa.load(io.BytesIO(audio_bytes), mono=True, sr=None)
    print(f"[LOAD] librosa OK sr={sr}")
    return y, sr


def extract_features(audio_bytes: bytes) -> RawFeatures:
    y, sr = _load_audio(audio_bytes)

    duration = librosa.get_duration(y=y, sr=sr)

    # ── Темп & хэмнэл ────────────────────────────────────────────────────────
    tempo_arr, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo_arr)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    if len(beat_times) > 2:
        ivs = np.diff(beat_times)
        rhythm_stability = float(1.0 - min(np.std(ivs) / (np.mean(ivs) + 1e-6), 1.0))
    else:
        rhythm_stability = 0.4

    # ── Onset (цохилт) ────────────────────────────────────────────────────────
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_rate = len(onset_times) / max(duration, 1.0)
    if len(onset_times) > 2:
        gaps = np.diff(onset_times)
        onset_regularity = float(1.0 - min(np.std(gaps) / (np.mean(gaps) + 1e-6), 1.0))
    else:
        onset_regularity = 0.3

    # ── Динамик ───────────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]
    db  = librosa.amplitude_to_db(rms, ref=np.max)
    dynamic_range_db = float(np.max(db) - np.min(db))
    rms_std_db       = float(np.std(db))
    silence_ratio    = float(np.mean(rms < 0.005))

    # ── Spectral ──────────────────────────────────────────────────────────────
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # ── Chroma (аккорд/нот) ───────────────────────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    note_energy = {note_names[i]: float(np.mean(chroma[i])) for i in range(12)}
    dominant_note   = max(note_energy, key=note_energy.get)
    active_notes    = sum(1 for v in note_energy.values() if v > 0.25)
    # Shannon entropy → нотын тархалт жигд үү?
    vals = np.array(list(note_energy.values())) + 1e-9
    vals /= vals.sum()
    chroma_entropy  = float(-np.sum(vals * np.log2(vals)))

    # ── MFCC тембр ────────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_variance = float(np.mean(np.var(mfcc, axis=1)))

    # ── Нормчилсон chroma vector (12 нот: C C# D D# E F F# G G# A A# B) ──────
    chroma_mean = np.array([float(np.mean(chroma[i])) for i in range(12)])
    norm = chroma_mean.sum() + 1e-9
    chroma_vector = [round(float(v / norm), 4) for v in chroma_mean]

    return RawFeatures(
        duration_sec=round(duration, 1),
        sr=int(sr),
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
    """
    Хэмнэлийн тогтвортой байдал — rhythm_stability + onset_regularity хосолсон.
    Acoustic guitar-д хэмнэл хамгийн чухал.
    """
    base = (f.rhythm_stability * 0.6 + f.onset_regularity * 0.4) * 100
    # Хэт удаан эсвэл хэт хурдан tempo → хасна
    tempo_penalty = 0.0
    if f.tempo_bpm < 40 or f.tempo_bpm > 220:
        tempo_penalty = 15.0
    return clamp(base - tempo_penalty)


def score_dynamics(f: RawFeatures) -> int:
    """
    Динамик хяналт — dynamic_range өргөн = сайн,
    silence_ratio хэт их = тоглолт тасарсан.
    """
    # 20–45 dB хүрээ = дундаж-сайн; <10 = монотон; >55 = хэт их хэлбэлзэл
    dr = f.dynamic_range_db
    if dr < 5:
        range_score = 20.0
    elif dr < 15:
        range_score = 40.0 + (dr - 5) * 3.0
    elif dr < 40:
        range_score = 70.0 + (dr - 15) * 0.8
    else:
        range_score = 90.0

    # Тасралт их байвал хасна
    silence_penalty = min(f.silence_ratio * 80, 30.0)
    return clamp(range_score - silence_penalty)


def score_clarity(f: RawFeatures) -> int:
    """
    Аккордын тодрол — chroma_entropy + active_note_count.
    Гитарын аккорд ихэвчлэн 3–6 нот → active_note_count 3-6 = сайн.
    Энтропи өндөр = нотууд тэгш тархсан = тод аккорд.
    """
    # Идэвхтэй нотын тоо 3–6 = 100 оноо руу ойртоно
    if f.active_note_count < 2:
        note_score = 20.0
    elif f.active_note_count <= 6:
        note_score = 50.0 + f.active_note_count * 8.0
    else:
        # Хэт олон нот = будлиантай дуу
        note_score = max(30.0, 98.0 - (f.active_note_count - 6) * 10)

    # Chroma entropy 2.5–3.2 bit = сайн (12 нотын дотор тэнцвэртэй)
    ent = f.chroma_entropy
    if ent < 1.5:
        ent_score = 30.0
    elif ent < 2.5:
        ent_score = 50.0 + (ent - 1.5) * 20.0
    elif ent <= 3.3:
        ent_score = 70.0 + (ent - 2.5) * 25.0
    else:
        ent_score = 85.0

    return clamp(note_score * 0.5 + ent_score * 0.5)


def score_consistency(f: RawFeatures) -> int:
    """
    Onset тогтмол байдал — цохилтын хурд, тогтвортой байдал.
    Acoustic guitar-д onset_rate 2–6/sec ердийн.
    """
    reg_score = f.onset_regularity * 100

    # onset rate оновчтой хүрээ 2–6/sec
    rate = f.onset_rate
    if rate < 1.0:
        rate_bonus = -20.0
    elif rate <= 6.0:
        rate_bonus = 10.0
    elif rate <= 10.0:
        rate_bonus = 0.0
    else:
        rate_bonus = -15.0

    return clamp(reg_score + rate_bonus)


def _chord_to_chroma(notes: list[str]) -> np.ndarray:
    """Нотын нэрсийн жагсаалтыг 12-хэмжээст chroma vector болгоно."""
    NOTE_IDX = {"C":0,"C#":1,"D":2,"D#":3,"E":4,"F":5,
                "F#":6,"G":7,"G#":8,"A":9,"A#":10,"B":11}
    vec = np.zeros(12)
    for n in notes:
        vec[NOTE_IDX[n]] += 1.0
    return vec / (vec.sum() + 1e-9)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── Дууны chord профайл ───────────────────────────────────────────────────────
#
# RIPTIDE — Vance Joy
#   Бүтэц: Am  G  C  (бараг бүх хэсэг)  +  Bridge: Fmaj7
#   Am  = A C E
#   G   = G B D
#   C   = C E G
#   Fmaj7 = F A C E   (bridge дээр л)
#
#   Нийт нотын давтамжийн жин (chord chart-аас):
#     Am: 40% үүссэн  → A,C,E   × 0.40
#     G:  35%         → G,B,D   × 0.35
#     C:  20%         → C,E,G   × 0.20
#     Fmaj7: 5%       → F,A,C,E × 0.05
#
RIPTIDE_PROFILE = (
    _chord_to_chroma(["A","C","E"]) * 0.40 +   # Am
    _chord_to_chroma(["G","B","D"]) * 0.35 +   # G
    _chord_to_chroma(["C","E","G"]) * 0.20 +   # C
    _chord_to_chroma(["F","A","C","E"]) * 0.05  # Fmaj7
)
RIPTIDE_PROFILE /= RIPTIDE_PROFILE.sum() + 1e-9

# AS LONG AS YOU LOVE ME — Justin Bieber
#   Capo 2 дээр: Am G C F  (relative: Bm A D G)
#   Bm = B D F#
#   A  = A C# E
#   D  = D F# A
#   G  = G B D
#
ALAYLM_PROFILE = (
    _chord_to_chroma(["B","D","F#"]) * 0.30 +  # Bm
    _chord_to_chroma(["A","C#","E"]) * 0.25 +  # A
    _chord_to_chroma(["D","F#","A"]) * 0.30 +  # D
    _chord_to_chroma(["G","B","D"])  * 0.15     # G
)
ALAYLM_PROFILE /= ALAYLM_PROFILE.sum() + 1e-9

# Хүлээгдэж буй tempo (BPM) — librosa half/double tempo алдаа нөхөхийн тулд
# octave tolerance-тэй шалгана
RIPTIDE_BPM  = 93.0
ALAYLM_BPM   = 100.0   # Bieber cover ихэвчлэн 96–104 BPM


def _bpm_score(detected: float, target: float, tol: float = 12.0) -> float:
    """
    Detected BPM-г target-тэй харьцуулна.
    librosa заримдаа tempo-г 2 дахин бага/их тоолдог тул
    half (target/2) болон double (target*2) хувилбарыг ч шалгана.
    0.0–1.0 буцаана.
    """
    for mult in (1.0, 0.5, 2.0):
        if abs(detected - target * mult) <= tol:
            # Ойр байх тусам өндөр оноо
            return 1.0 - abs(detected - target * mult) / tol
    return 0.0


def detect_song(f: RawFeatures) -> str:
    """
    Chord profile cosine similarity + BPM ойролцоо байдлаар дуу таана.

    Chord chart-д суурилсан RIPTIDE_PROFILE болон ALAYLM_PROFILE-тэй
    хэрэглэгчийн chroma vector-ийг харьцуулна.
    Cosine similarity 70%, BPM оноо 30% жинтэй.
    """
    if not f.chroma_vector:
        return "Тодорхойгүй"

    cv = np.array(f.chroma_vector)

    # Chroma similarity
    riptide_chroma  = _cosine(cv, RIPTIDE_PROFILE)
    alaylm_chroma   = _cosine(cv, ALAYLM_PROFILE)

    # BPM оноо
    riptide_bpm  = _bpm_score(f.tempo_bpm, RIPTIDE_BPM)
    alaylm_bpm   = _bpm_score(f.tempo_bpm, ALAYLM_BPM)

    # Нийт оноо: chroma 70% + bpm 30%
    riptide_total = riptide_chroma * 0.70 + riptide_bpm * 0.30
    alaylm_total  = alaylm_chroma  * 0.70 + alaylm_bpm  * 0.30

    THRESHOLD = 0.45   # энэ доороос "Тодорхойгүй"

    best_score = max(riptide_total, alaylm_total)
    if best_score < THRESHOLD:
        return "Тодорхойгүй"
    if riptide_total >= alaylm_total:
        return "Riptide — Vance Joy"
    return "As Long As You Love Me — Justin Bieber"


def level_from_score(score: int) -> str:
    if score >= 82:
        return "Professional"
    if score >= 62:
        return "Advanced"
    if score >= 38:
        return "Intermediate"
    return "Beginner"


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEEDBACK GENERATOR  (дүрмэд суурилсан текст)
# ─────────────────────────────────────────────────────────────────────────────
def make_feedback(f: RawFeatures, r: int, d: int, cl: int, co: int, total: int) -> dict:
    level = level_from_score(total)
    song  = detect_song(f)

    # ── Темп тайлбар ─────────────────────────────────────────────────────────
    bpm = f.tempo_bpm
    if bpm < 60:
        tempo_feel = f"Удаан ({bpm:.0f} BPM) — suranзалтай темп"
    elif bpm < 80:
        tempo_feel = f"Зөөлөн удаан ({bpm:.0f} BPM)"
    elif bpm < 100:
        tempo_feel = f"Дундаж хурд ({bpm:.0f} BPM) — ердийн acoustic guitar темп"
    elif bpm < 130:
        tempo_feel = f"Хурдан ({bpm:.0f} BPM)"
    else:
        tempo_feel = f"Маш хурдан ({bpm:.0f} BPM)"

    # ── Хэмнэлийн үнэлгээ текст ──────────────────────────────────────────────
    if r >= 80:
        rhythm_accuracy = "Маш сайн — хэмнэл тогтвортой, найдвартай"
    elif r >= 60:
        rhythm_accuracy = "Сайн — жижиг хэлбэлзэл байгаа ч ерөнхийдөө тогтвортой"
    elif r >= 40:
        rhythm_accuracy = "Дундаж — хэмнэл тогтворгүй хэсгүүд бий"
    else:
        rhythm_accuracy = "Хангалтгүй — хэмнэл ихээхэн хэлбэлзэлтэй"

    # ── Аккордын тодрол ───────────────────────────────────────────────────────
    if cl >= 75:
        chord_clarity = "Маш тод — аккорд цэвэр дарагдаж байна"
    elif cl >= 55:
        chord_clarity = "Тод — ерөнхийдөө тодорхой, бага зэрэг бузрал бий"
    elif cl >= 35:
        chord_clarity = "Бүдэг — аккорд бузартсан (buzzing) хэсгүүд байна"
    else:
        chord_clarity = "Тодорхойгүй — аккорд нарийвчлал хэрэгтэй"

    # ── Динамик хүрээ ─────────────────────────────────────────────────────────
    if d >= 75:
        dynamic_range = f"Өргөн ({f.dynamic_range_db:.0f} dB) — сайн динамик хяналт"
    elif d >= 50:
        dynamic_range = f"Дундаж ({f.dynamic_range_db:.0f} dB)"
    else:
        dynamic_range = f"Нарийн ({f.dynamic_range_db:.0f} dB) — динамик хяналт хэрэгтэй"

    # ── Давуу талууд ─────────────────────────────────────────────────────────
    strengths = []
    if r >= 65:
        strengths.append("Хэмнэл тогтвортой, beat алдагдахгүй тоглож байна")
    if d >= 65:
        strengths.append("Динамик хэлбэлзэл сайн — чанга/намхан хэсгийг ялган тоглодог")
    if cl >= 65:
        strengths.append("Аккорд тодорхой, string мулталт бага")
    if co >= 65:
        strengths.append("Цохилт тогтмол, strumming/picking жигд")
    if f.silence_ratio < 0.05:
        strengths.append("Тасралтгүй тоглолт — дуу урсгалтай")
    if not strengths:
        strengths.append("Тоглох хүсэл эрмэлзэл харагдаж байна — дадлага үргэлжлүүлнэ үү")

    # ── Сайжруулах хэсгүүд ───────────────────────────────────────────────────
    areas = []
    if r < 60:
        areas.append("Хэмнэлийн тогтвортой байдал — метроном ашиглан дадлага хий")
    if cl < 55:
        areas.append("Аккордын дарлага — хуруу байрлал, дарах хүч нарийвчлах")
    if d < 50:
        areas.append("Динамик хяналт — нам ба чанга хэсгийг ухамсартайгаар ялгах")
    if co < 55:
        areas.append("Strumming/picking тогтмол байдал — хэв маяг дагаж дадлага хий")
    if f.silence_ratio > 0.15:
        areas.append("Аккорд шилжилтийн хурд — тасралтыг багасгах")
    if not areas:
        areas.append("Илүү нарийн техник (vibrato, fingerpicking) сурах")

    # ── Дадлагын зөвлөмж ─────────────────────────────────────────────────────
    tips = []

    if r < 65:
        tips.append(
            "Метроном 70 BPM-с эхлээд дуртай дуугаа тоглох дадлага хий. "
            "Хэмнэл тогтвортой болсны дараа BPM нэмэ."
        )
    else:
        tips.append(
            "Метроном ашиглаж байгаа бол BPM-ийг аажмаар ихэсгэж хурдыг нэмэгдүүл."
        )

    if cl < 60:
        tips.append(
            "Аккорд бүрийг тусад нь дарж, string бүр тод дуугарч байгааг шалга. "
            "Хуруу байрлалаа тохируулаад дараа нь аккорд шилжилтийн дадлага хий."
        )
    else:
        tips.append(
            "Аккорд шилжилтийн хурдыг нэмэх — Am→F→C→G дараалалд timer тавьж дадла."
        )

    if d < 55:
        tips.append(
            "Намхан (piano) болон чанга (forte) хэсгийг ухамсартайгаар тоглох дадлага хий. "
            "Ялгаа их байх тусам сонсогдох чанар сайжирна."
        )
    else:
        tips.append(
            "Fingerpicking техник нэмж, ганц нотоор melody тоглох дадлагыг туршиж үз."
        )

    # ── Ерөнхий үнэлгээ ──────────────────────────────────────────────────────
    level_desc = {
        "Beginner":      "Эхлэгч түвшний тоглогч.",
        "Intermediate":  "Дундаж түвшний тоглогч.",
        "Advanced":      "Дэвшилтэт түвшний тоглогч.",
        "Professional":  "Мэргэжлийн түвшний тоглогч.",
    }[level]

    weak = []
    if r < 60:     weak.append("хэмнэл")
    if cl < 55:    weak.append("аккордын тодрол")
    if d < 50:     weak.append("динамик")
    if co < 55:    weak.append("цохилтын тогтвортой байдал")

    if weak:
        weak_str = "、".join(weak)
        overall = (
            f"{level_desc} Нийт оноо {total}/100. "
            f"{weak_str}-ыг сайжруулбал дараагийн түвшинд гарна. "
            f"Өдөр бүр 20-30 минут тогтмол дадлага хийх нь хамгийн үр дүнтэй."
        )
    else:
        overall = (
            f"{level_desc} Нийт оноо {total}/100. "
            f"Бүх үзүүлэлт сайн байна — илүү нарийн техник болон хэлбэр сурах цаг боллоо. "
            f"Бичлэг хийж өөрийгөө сонсох нь ахиц дэвшлийг мэдрэхэд тусална."
        )

    return dict(
        skill_level=level,
        skill_score=total,
        song_detected=song,
        duration_seconds=f.duration_sec,
        tempo_bpm=f.tempo_bpm,
        tempo_feel=tempo_feel,
        rhythm_accuracy=rhythm_accuracy,
        chord_clarity=chord_clarity,
        dynamic_range=dynamic_range,
        strengths=strengths,
        areas_to_improve=areas,
        practice_tips=tips,
        overall_feedback=overall,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANALYZE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def analyze_pipeline(audio_bytes: bytes) -> SkillAnalysis:
    # a) Feature задлах
    f = extract_features(audio_bytes)

    # b) Дэд оноонууд
    r  = score_rhythm(f)
    d  = score_dynamics(f)
    cl = score_clarity(f)
    co = score_consistency(f)

    # c) Нийт оноо — хэмнэл хамгийн чухал (40%), дараа нь clarity (25%)
    total = clamp(r * 0.40 + d * 0.15 + cl * 0.25 + co * 0.20)

    # d) Текст санал
    feedback = make_feedback(f, r, d, cl, co, total)

    # e) Дэд оноонуудыг response-д нэмэх
    feedback["rhythm_score"]      = r
    feedback["dynamics_score"]    = d
    feedback["clarity_score"]     = cl
    feedback["consistency_score"] = co

    return SkillAnalysis(**feedback)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

async def debug_song(
    audio: UploadFile = File(...),
):
    """
    Chroma vector болон Riptide/ALAYLM similarity оноог задлан харуулна.
    Алгоритм тохируулахад ашиглана.
    """
    audio_bytes = await audio.read()
    try:
        f = extract_features(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    cv = np.array(f.chroma_vector) if f.chroma_vector else np.zeros(12)

    riptide_chroma = _cosine(cv, RIPTIDE_PROFILE)
    alaylm_chroma  = _cosine(cv, ALAYLM_PROFILE)
    riptide_bpm    = _bpm_score(f.tempo_bpm, RIPTIDE_BPM)
    alaylm_bpm     = _bpm_score(f.tempo_bpm, ALAYLM_BPM)
    riptide_total  = riptide_chroma * 0.70 + riptide_bpm * 0.30
    alaylm_total   = alaylm_chroma  * 0.70 + alaylm_bpm  * 0.30

    return {
        "tempo_bpm": f.tempo_bpm,
        "dominant_note": f.dominant_note,
        "active_note_count": f.active_note_count,
        "chroma_vector": {NOTE_NAMES[i]: f.chroma_vector[i] for i in range(12)},
        "riptide_profile": {NOTE_NAMES[i]: round(float(RIPTIDE_PROFILE[i]), 4) for i in range(12)},
        "alaylm_profile":  {NOTE_NAMES[i]: round(float(ALAYLM_PROFILE[i]), 4) for i in range(12)},
        "scores": {
            "riptide": {
                "chroma_similarity": round(riptide_chroma, 4),
                "bpm_score":         round(riptide_bpm, 4),
                "total":             round(riptide_total, 4),
            },
            "alaylm": {
                "chroma_similarity": round(alaylm_chroma, 4),
                "bpm_score":         round(alaylm_bpm, 4),
                "total":             round(alaylm_total, 4),
            },
        },
        "detected": detect_song(f),
        "threshold": 0.45,
    }


@app.get("/")
async def root():
    return {
        "name": "Guitar Skill Analyzer — Local",
        "version": "3.1.0",
        "description": "Гадаад API-гүй — librosa + chord-profile cosine similarity",
        "endpoints": {
            "POST /analyze": "Бүрэн шинжилгээ + ур чадварын үнэлгээ",
            "POST /debug":   "Chroma vector + similarity оноо харах (тохиргоонд)",
        },
        "formats": ["wav", "mp3", "m4a", "flac", "ogg", "webm"],
        "max_mb": MAX_FILE_MB,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)