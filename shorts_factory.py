#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 SHORTS FACTORY  ·  Autonomous YouTube Shorts Production & Scheduling Engine
==============================================================================
 Author : Elite Senior Video Automation Engineer
 Target : 5 videos/day · Mon–Fri · 25 videos/week  (3 daily shifts)
 Stack  : moviepy 1.0.3 · opencv-python (cv2) · openai-whisper · numpy

 WHAT THIS SCRIPT DOES
 ---------------------
 1. Scans an input folder of raw creator downloads (or a URL list).
 2. Classifies each source by creator (EiNerd, Felipe Neto, Cariani,
    Alanzoka, Paulinho o Loko) using filename keyword matching.
 3. Builds a full weekly plan of 25 slots (5 days x 5 slots) mapped to the
    3-shift schedule, guaranteeing every creator is covered.
 4. For each slot it extracts a "viral hook" sub-clip and renders a
    9:16 1080x1920 @30fps Short with:
        - OpenCV face tracking + rolling-average "stabilizer motion" crop
        - Whisper word-by-word pop-up subtitles (#FFDB58 + black outline)
 5. Emits publication_queue.json (the scheduling database) and validates
    that no two publication timestamps overlap.

 DEPENDENCIES (tested versions)
 ------------------------------
    pip install moviepy==1.0.3 opencv-python==4.9.0.80 openai-whisper numpy
    # + ffmpeg on PATH  (moviepy)
    # + ImageMagick     (moviepy TextClip)  ->  set IMAGEMAGICK_BINARY below
    # + a font named "Impact" available to ImageMagick

 USAGE
 -----
    # Plan only (no rendering) — safe to run anywhere:
    python shorts_factory.py --input-dir ./raw --output-dir ./out --dry-run

    # Full production run:
    python shorts_factory.py --input-dir ./raw --output-dir ./out \
                             --whisper-model small --font Impact

    # Render only the first 3 slots (smoke test):
    python shorts_factory.py --input-dir ./raw --output-dir ./out --limit 3
==============================================================================
"""

import os
import sys
import json
import glob
import random
import logging
import argparse
import datetime as dt
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
#  Optional / heavy imports are guarded so that --dry-run works on any machine #
# --------------------------------------------------------------------------- #
try:
    import cv2
    _CV2_OK = True
except Exception as _e:                                    # pragma: no cover
    cv2 = None
    _CV2_OK = False
    _CV2_ERR = _e

try:
    from moviepy.editor import (VideoFileClip, TextClip,
                                CompositeVideoClip)
    _MOVIEPY_OK = True
except Exception as _e:                                    # pragma: no cover
    _MOVIEPY_OK = False
    _MOVIEPY_ERR = _e

try:
    import whisper
    _WHISPER_OK = True
except Exception as _e:                                    # pragma: no cover
    whisper = None
    _WHISPER_OK = False
    _WHISPER_ERR = _e


# --------------------------------------------------------------------------- #
#  LOGGING                                                                     #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shorts_factory")


# =========================================================================== #
#  SECTION 1 — GLOBAL CONFIGURATION                                           #
# =========================================================================== #

# Output video spec (vertical 9:16)
OUT_W, OUT_H = 1080, 1920
OUT_FPS = 30
OUT_BITRATE = "12000k"          # high bitrate for crisp Shorts
OUT_AUDIO_BITRATE = "192k"
OUT_CODEC = "libx264"
OUT_AUDIO_CODEC = "aac"
OUT_PRESET = "medium"

# Subtitle styling
SUB_COLOR = "#FFDB58"           # mustard yellow
SUB_STROKE_COLOR = "black"
SUB_STROKE_WIDTH = 5
SUB_FONT_SIZE = 92
SUB_Y_POS = int(OUT_H * 0.64)   # lower-middle third
SUB_MAX_WORDS_ON_SCREEN = 1     # word-by-word

# Hook extraction
HOOK_MIN_SEC = 28.0
HOOK_MAX_SEC = 58.0             # keep < 60s (Shorts limit)

# Face tracking
FACE_SAMPLE_EVERY = 3           # detect every N frames (interpolate the rest)
FACE_SMOOTH_WINDOW = 15         # rolling-average window (frames)
FACE_MIN_SIZE = 60              # min face size (px)

# ImageMagick binary (Windows users usually must set this explicitly).
# Leave as None to rely on the system default / moviepy auto-detection.
IMAGEMAGICK_BINARY = os.environ.get("IMAGEMAGICK_BINARY", None)

# The 5 fixed daily slots mapped to the 3 shifts.
#   Morning  -> 1 video
#   Afternoon-> 2 videos
#   Evening  -> 2 videos
DAILY_SLOTS = [
    ("morning",   "09:00"),
    ("afternoon", "13:00"),
    ("afternoon", "16:00"),
    ("evening",   "19:00"),
    ("evening",   "21:30"),
]

# The 5 publication weekdays (Mon–Fri).
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Minimum gap between two scheduled publications (overlap guard).
MIN_GAP_MINUTES = 5


# =========================================================================== #
#  SECTION 2 — CREATOR PROFILES (content prioritization pool)                 #
# =========================================================================== #

@dataclass
class CreatorProfile:
    key: str
    display_name: str
    keywords: list          # used to classify raw filenames
    hashtags: list          # niche hashtags for metadata
    title_templates: list   # rotated per publication


CREATORS = [
    CreatorProfile(
        key="einerd",
        display_name="EiNerd / Peter Aqui",
        keywords=["einerd", "peteraqui", "peter aqui", "breno",
                  "brenoaldonerd", "nerd"],
        hashtags=["#einerd", "#brenoaldonerd", "#popculture",
                  "#curiosidadesnerd"],
        title_templates=[
            "Você NÃO sabia disso sobre {topic}! 🤯 #shorts",
            "A curiosidade nerd que quebrou a internet 🎬",
            "Isso mudou TUDO na cultura pop 😱",
        ],
    ),
    CreatorProfile(
        key="felipeneto",
        display_name="Felipe Neto",
        keywords=["felipeneto", "felipe neto", "netolab", "neto"],
        hashtags=["#felipeneto", "#netolab", "#cortesgossip",
                  "#reacaofelipeneto"],
        title_templates=[
            "Felipe Neto NÃO segurou e falou TUDO 😳 #shorts",
            "A reação que ninguém esperava 🔥",
            "Ele não deixou barato... 👀",
        ],
    ),
    CreatorProfile(
        key="cariani",
        display_name="Renato Cariani",
        keywords=["cariani", "renatocariani", "maromba",
                  "bodybuilding", "monstro"],
        hashtags=["#renatocariani", "#maromba", "#cortesmaromba",
                  "#bodybuildingbr"],
        title_templates=[
            "O SEGREDO da maromba que ninguém te conta 💪 #shorts",
            "Cariani DETONOU esse mito do treino 🔥",
            "Faça ISSO e mude seu shape 😤",
        ],
    ),
    CreatorProfile(
        key="alanzoka",
        display_name="Alanzoka",
        keywords=["alanzoka", "alan", "gameplay", "cortesdoalan"],
        hashtags=["#alanzoka", "#cortesdoalan", "#gameplaybr",
                  "#momentosengracados"],
        title_templates=[
            "Alanzoka SURTOU nesse momento 😂 #shorts",
            "O corte mais engraçado do Alan 🤣",
            "Não teve como segurar a risada 😭",
        ],
    ),
    CreatorProfile(
        key="paulinho",
        display_name="Paulinho o Loko",
        keywords=["paulinho", "oloko", "o loko", "gta5", "gta 5",
                  "gtarp", "gta rp", "rp"],
        hashtags=["#paulinhooloko", "#gta5rp", "#cortesrp",
                  "#gta5engracado"],
        title_templates=[
            "Paulinho fez a MAIOR treta no GTA RP 😂 #shorts",
            "Esse momento no GTA 5 é surreal 🎮",
            "O caos do Paulinho no RP 🤣",
        ],
    ),
]

CREATOR_BY_KEY = {c.key: c for c in CREATORS}

# Generic hashtags appended to every video.
GENERIC_HASHTAGS = ["#shorts", "#viral", "#cortes", "#fyp", "#brasil"]

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


# =========================================================================== #
#  SECTION 3 — CONTENT LIBRARY (scan + classify raw downloads)               #
# =========================================================================== #

class ContentLibrary:
    """Scans the input folder and buckets each source file by creator."""

    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.by_creator = {c.key: [] for c in CREATORS}
        self.unclassified = []
        self._round_robin = {c.key: 0 for c in CREATORS}
        self._scan()

    def _scan(self):
        if not self.input_dir or not os.path.isdir(self.input_dir):
            log.warning("Input dir '%s' not found — library is empty.",
                        self.input_dir)
            return

        files = []
        for ext in VIDEO_EXTS:
            files.extend(glob.glob(os.path.join(self.input_dir, "**", "*" + ext),
                                   recursive=True))
        files = sorted(set(files))

        for path in files:
            name = os.path.basename(path).lower()
            matched = None
            for creator in CREATORS:
                if any(kw in name for kw in creator.keywords):
                    matched = creator.key
                    break
            if matched:
                self.by_creator[matched].append(path)
            else:
                self.unclassified.append(path)

        total = sum(len(v) for v in self.by_creator.values())
        log.info("Scanned %d source files (%d classified, %d unclassified).",
                 len(files), total, len(self.unclassified))
        for c in CREATORS:
            log.info("   %-22s : %d file(s)",
                     c.display_name, len(self.by_creator[c.key]))

    def all_sources(self):
        pool = list(self.unclassified)
        for v in self.by_creator.values():
            pool.extend(v)
        return pool

    def pick_source(self, creator_key: str):
        """
        Return a source path for the given creator using round-robin so a
        creator's clips are spread evenly across the week. Falls back to
        unclassified pool, then to the global pool.
        """
        bucket = self.by_creator.get(creator_key, [])
        if not bucket:
            bucket = self.unclassified or self.all_sources()
        if not bucket:
            return None
        idx = self._round_robin.get(creator_key, 0) % len(bucket)
        self._round_robin[creator_key] = idx + 1
        return bucket[idx]


# =========================================================================== #
#  SECTION 4 — WEEKLY SCHEDULING ENGINE                                       #
# =========================================================================== #

def next_monday(reference: dt.date) -> dt.date:
    """Return the Monday of the upcoming week (today if today is Monday)."""
    days_ahead = (0 - reference.weekday()) % 7      # Monday == 0
    if days_ahead == 0 and reference.weekday() != 0:
        days_ahead = 7
    return reference + dt.timedelta(days=days_ahead)


def build_creator_rotation(seed: int = None) -> list:
    """
    Produce a randomized list of exactly 25 creator keys where each of the
    5 creators appears exactly 5 times — guarantees full weekly coverage
    while still randomizing the order.
    """
    rng = random.Random(seed)
    pool = [c.key for c in CREATORS] * 5      # 25 entries, 5 each
    rng.shuffle(pool)
    return pool


def sec_to_timestr(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def build_metadata(creator: CreatorProfile, day_name: str,
                   slot_index: int, rng: random.Random) -> dict:
    """Generate title, description and hashtag set for one publication."""
    template = rng.choice(creator.title_templates)
    title = template.replace("{topic}", creator.display_name.split("/")[0].strip())

    hashtags = list(dict.fromkeys(creator.hashtags + GENERIC_HASHTAGS))
    tag_line = " ".join(hashtags)

    description = (
        f"🎬 {creator.display_name} | Melhores momentos e cortes!\n\n"
        f"🔥 Se inscreva e ative o sino para não perder nenhum corte.\n"
        f"👍 Deixe seu like e comente o que achou!\n\n"
        f"{tag_line}\n\n"
        f"#{creator.key} #cortes #shorts\n"
        f"—\n"
        f"⚠️ Conteúdo de cortes/curadoria. Créditos ao criador original: "
        f"{creator.display_name}."
    )
    return {"title": title, "description": description, "hashtags": hashtags}


def build_weekly_plan(library: ContentLibrary,
                      week_start: dt.date,
                      output_dir: str,
                      seed: int = None) -> list:
    """
    Build the full 25-slot plan. Each entry is a dict ready to be rendered
    and serialized into publication_queue.json.
    """
    rng = random.Random(seed)
    rotation = build_creator_rotation(seed)
    plan = []
    slot_counter = 0

    for d, day_name in enumerate(WEEKDAYS):
        pub_date = week_start + dt.timedelta(days=d)
        for (shift, time_str) in DAILY_SLOTS:
            creator_key = rotation[slot_counter]
            creator = CREATOR_BY_KEY[creator_key]

            hh, mm = (int(x) for x in time_str.split(":"))
            publish_at = dt.datetime.combine(pub_date, dt.time(hh, mm))

            source = library.pick_source(creator_key)
            meta = build_metadata(creator, day_name, slot_counter, rng)

            slot_id = f"{pub_date.isoformat()}_{shift}_{slot_counter:02d}"
            out_name = f"{pub_date.isoformat()}_{creator_key}_{slot_counter:02d}.mp4"
            out_path = os.path.join(output_dir, out_name)

            entry = {
                "id": slot_id,
                "publish_at": publish_at.isoformat(),
                "day": day_name,
                "shift": shift,
                "slot_index": slot_counter,
                "creator_key": creator_key,
                "creator": creator.display_name,
                "source_file": source,
                "hook": None,               # filled at render time
                "output_file": out_path,
                "title": meta["title"],
                "description": meta["description"],
                "hashtags": meta["hashtags"],
                "status": "scheduled",
                "render_error": None,
            }
            plan.append(entry)
            slot_counter += 1

    log.info("Weekly plan built: %d slots (%s → %s).",
             len(plan), plan[0]["publish_at"], plan[-1]["publish_at"])
    return plan


def build_day_plan(library: ContentLibrary, target_date: dt.date,
                   output_dir: str, seed: int = None) -> list:
    """
    Build just the 5 slots for ONE day (GitHub Actions daily mode): one video
    per shift, one creator each (shuffled), scheduled at that day's shift times.
    """
    rng = random.Random(seed)
    creators = [c.key for c in CREATORS]
    rng.shuffle(creators)
    day_name = target_date.strftime("%A")
    plan = []
    for i, (shift, time_str) in enumerate(DAILY_SLOTS):
        creator_key = creators[i % len(creators)]
        creator = CREATOR_BY_KEY[creator_key]
        hh, mm = (int(x) for x in time_str.split(":"))
        publish_at = dt.datetime.combine(target_date, dt.time(hh, mm))
        source = library.pick_source(creator_key)
        meta = build_metadata(creator, day_name, i, rng)
        plan.append({
            "id": f"{target_date.isoformat()}_{shift}_{i:02d}",
            "publish_at": publish_at.isoformat(),
            "day": day_name,
            "shift": shift,
            "slot_index": i,
            "creator_key": creator_key,
            "creator": creator.display_name,
            "source_file": source,
            "hook": None,
            "output_file": os.path.join(
                output_dir, f"{target_date.isoformat()}_{creator_key}_{i:02d}.mp4"),
            "title": meta["title"],
            "description": meta["description"],
            "hashtags": meta["hashtags"],
            "status": "scheduled",
            "render_error": None,
        })
    log.info("Day plan built: %d slots for %s.", len(plan), target_date.isoformat())
    return plan


def validate_no_overlap(plan: list, min_gap_minutes: int = MIN_GAP_MINUTES):
    """
    Assert that all publication timestamps are strictly increasing with at
    least `min_gap_minutes` between consecutive entries. Raises ValueError
    on any overlap/duplicate.
    """
    times = sorted(dt.datetime.fromisoformat(e["publish_at"]) for e in plan)
    gap = dt.timedelta(minutes=min_gap_minutes)
    for i in range(1, len(times)):
        delta = times[i] - times[i - 1]
        if delta < gap:
            raise ValueError(
                f"Schedule overlap detected between "
                f"{times[i-1]} and {times[i]} (gap {delta} < {gap})."
            )
    log.info("Schedule validation OK — %d unique slots, no overlaps.", len(times))


# =========================================================================== #
#  SECTION 5 — FACE TRACKING ("Stabilizer Motion")                           #
# =========================================================================== #

def _rolling_average(values: np.ndarray, window: int) -> np.ndarray:
    """Length-preserving rolling average (edge-padded)."""
    n = len(values)
    if n == 0 or window < 2 or n < window:
        return values.astype(float)
    cumsum = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    ra = (cumsum[window:] - cumsum[:-window]) / float(window)   # len = n-window+1
    pad_front = window // 2
    pad_back = n - len(ra) - pad_front
    front = np.full(max(pad_front, 0), ra[0])
    back = np.full(max(pad_back, 0), ra[-1])
    out = np.concatenate([front, ra, back])
    return out[:n]


def detect_face_track(video_path: str, start_s: float, end_s: float,
                      src_fps: float) -> np.ndarray:
    """
    Frame-by-frame face detection over [start_s, end_s]. Returns a per-frame
    array of smoothed horizontal face-center X positions (in source pixels).
    Missing detections carry the last known position forward.
    """
    if not _CV2_OK:
        return np.array([])

    cascade_path = os.path.join(cv2.data.haarcascades,
                                "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("cv2 could not open %s for tracking.", video_path)
        return np.array([])

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or OUT_W
    start_frame = int(round(start_s * src_fps))
    end_frame = int(round(end_s * src_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    positions = []
    last_cx = src_w / 2.0
    fidx = start_frame

    while fidx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (fidx - start_frame) % FACE_SAMPLE_EVERY == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE),
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                last_cx = x + w / 2.0
        positions.append(last_cx)
        fidx += 1

    cap.release()

    if not positions:
        return np.array([])

    arr = np.array(positions, dtype=float)
    smoothed = _rolling_average(arr, FACE_SMOOTH_WINDOW)
    log.info("   Face track: %d frames, X range [%.0f..%.0f] smoothed.",
             len(smoothed), smoothed.min(), smoothed.max())
    return smoothed


def make_vertical_processor(positions: np.ndarray, fps: float,
                            src_w: int, src_h: int):
    """
    Return a moviepy frame filter fun(get_frame, t) -> RGB uint8 frame of
    size (OUT_H, OUT_W, 3). Landscape sources are cropped 9:16 and tracked
    on the face; narrow sources get a blurred-background fit.
    """
    target_ar = OUT_W / OUT_H
    src_ar = src_w / float(src_h)

    if src_ar >= target_ar:
        # Wide/landscape -> crop a 9:16 window centered on the tracked face.
        crop_w = int(round(src_h * target_ar))
        crop_w = min(crop_w, src_w)

        def process(get_frame, t):
            frame = get_frame(t)
            i = int(round(t * fps))
            if len(positions) > 0:
                i = min(max(i, 0), len(positions) - 1)
                cx = positions[i]
            else:
                cx = src_w / 2.0
            x1 = int(round(cx - crop_w / 2.0))
            x1 = max(0, min(x1, src_w - crop_w))
            cropped = frame[:, x1:x1 + crop_w]
            return cv2.resize(cropped, (OUT_W, OUT_H),
                              interpolation=cv2.INTER_LANCZOS4)
    else:
        # Narrow/portrait -> blurred, darkened background + centered fit.
        def process(get_frame, t):
            frame = get_frame(t)
            bg = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
            bg = cv2.GaussianBlur(bg, (0, 0), 25)
            bg = (bg.astype(np.float32) * 0.55).astype(np.uint8)

            scale = OUT_W / float(src_w)
            fh = int(round(src_h * scale))
            fg = cv2.resize(frame, (OUT_W, fh), interpolation=cv2.INTER_LANCZOS4)

            canvas = bg.copy()
            if fh <= OUT_H:
                y0 = (OUT_H - fh) // 2
                canvas[y0:y0 + fh, 0:OUT_W] = fg
            else:
                cy = (fh - OUT_H) // 2
                canvas[:, :] = fg[cy:cy + OUT_H, 0:OUT_W]
            return canvas

    return process


# =========================================================================== #
#  SECTION 6 — WHISPER WORD-BY-WORD SUBTITLES                                 #
# =========================================================================== #

def transcribe_words(audio_path: str, model, language: str = "pt") -> list:
    """
    Run Whisper with word timestamps. Returns a list of
    {word, start, end} dicts (times relative to the audio clip).
    """
    if model is None:
        return []
    result = model.transcribe(audio_path, language=language,
                              word_timestamps=True, verbose=False)
    words = []
    for seg in result.get("segments", []):
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                token = (w.get("word") or "").strip()
                if token:
                    words.append({
                        "word": token,
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                    })
        else:
            # Fallback: spread the segment text evenly across its span.
            tokens = (seg.get("text") or "").split()
            if not tokens:
                continue
            s, e = float(seg["start"]), float(seg["end"])
            step = (e - s) / len(tokens)
            for k, tok in enumerate(tokens):
                words.append({
                    "word": tok.strip(),
                    "start": s + k * step,
                    "end": s + (k + 1) * step,
                })
    return words


def _popup_scale(t: float) -> float:
    """Quick pop-in from 60% to 100% over ~0.12s (subtitle appearance)."""
    return min(1.0, 0.60 + 3.5 * t)


def make_word_clip(word: str, start: float, end: float,
                   duration_cap: float, font: str):
    """
    Build one styled, animated TextClip for a single word.
    Returns None if TextClip could not be created (e.g. ImageMagick missing).
    """
    dur = max(0.10, min(end - start, duration_cap - start))
    if dur <= 0:
        return None

    def _try(font_name):
        return TextClip(
            word.upper(),
            fontsize=SUB_FONT_SIZE,
            font=font_name,
            color=SUB_COLOR,
            stroke_color=SUB_STROKE_COLOR,
            stroke_width=SUB_STROKE_WIDTH,
            method="label",
        )

    try:
        tc = _try(font)
    except Exception as e:
        # Fallback to a font that ships with most ImageMagick installs.
        log.debug("Font '%s' failed (%s) — falling back.", font, e)
        try:
            tc = _try("DejaVu-Sans-Bold")
        except Exception as e2:
            log.warning("TextClip failed for word '%s': %s", word, e2)
            return None

    tc = (tc
          .set_start(start)
          .set_duration(dur)
          .set_position(("center", SUB_Y_POS))
          .resize(_popup_scale))
    return tc


# =========================================================================== #
#  SECTION 7 — RENDER ONE SHORT                                               #
# =========================================================================== #

def load_hints(path: str) -> dict:
    """Load hooks_hints.json (most-replayed peaks written by sourcing.py)."""
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def choose_hook(clip_duration: float, rng: random.Random,
                source_path: str = None, hints: dict = None) -> tuple:
    """
    Pick the hook window. If sourcing.py left a 'most replayed' hint for this
    source (matched by filename stem, e.g. 'einerd_<videoid>'), cut a few
    seconds BEFORE the viral peak. Otherwise fall back to a random window.
    """
    hook_len = min(HOOK_MAX_SEC, clip_duration)
    hook_len = max(min(hook_len, HOOK_MAX_SEC), min(HOOK_MIN_SEC, clip_duration))
    if clip_duration <= hook_len:
        return 0.0, clip_duration
    max_start = clip_duration - hook_len

    if source_path and hints:
        stem = os.path.splitext(os.path.basename(source_path))[0]
        hint = hints.get(stem)
        if hint and hint.get("peak_start") is not None:
            lead = float(hint.get("lead", 4.0))
            start = max(0.0, min(float(hint["peak_start"]) - lead, max_start))
            log.info("   Hook @ replay peak %.0fs (intensity %.2f).",
                     hint["peak_start"], hint.get("intensity", 0.0))
            return start, start + hook_len

    start = rng.uniform(0.0, max_start)
    return start, start + hook_len


def render_short(entry: dict, whisper_model, font: str,
                 scratch_dir: str, rng: random.Random,
                 hints: dict = None) -> bool:
    """
    Full render pipeline for a single publication entry.
    Returns True on success. All clips are explicitly .close()d in a
    finally block to avoid file-handle / memory leaks in the batch loop.
    """
    source = entry["source_file"]
    out_path = entry["output_file"]

    if not source or not os.path.isfile(source):
        entry["status"] = "failed"
        entry["render_error"] = f"source not found: {source}"
        log.error("   Missing source for %s", entry["id"])
        return False

    src_clip = None
    hook_clip = None
    base = None
    final = None
    word_clips = []
    tmp_audio = os.path.join(scratch_dir, f"{entry['slot_index']:02d}_audio.wav")

    try:
        # 1) Load source and choose a hook.
        src_clip = VideoFileClip(source)
        src_fps = src_clip.fps or OUT_FPS
        start, end = choose_hook(src_clip.duration, rng, source, hints)
        entry["hook"] = {
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "start_ts": sec_to_timestr(start),
            "end_ts": sec_to_timestr(end),
        }
        hook_clip = src_clip.subclip(start, end)
        src_w, src_h = src_clip.w, src_clip.h

        # 2) Face track over the hook range ("stabilizer motion").
        positions = detect_face_track(source, start, end, src_fps)

        # 3) Build the vertical 9:16 base clip (crop+track or fit).
        processor = make_vertical_processor(positions, src_fps, src_w, src_h)
        base = hook_clip.fl(processor, apply_to=[], keep_duration=True)
        base.size = (OUT_W, OUT_H)          # correct stale size metadata
        base = base.set_duration(hook_clip.duration)

        # 4) Transcribe + build word-by-word subtitles.
        if whisper_model is not None and hook_clip.audio is not None:
            hook_clip.audio.write_audiofile(
                tmp_audio, fps=16000, nbytes=2,
                codec="pcm_s16le", logger=None,
            )
            words = transcribe_words(tmp_audio, whisper_model)
            for w in words:
                if w["start"] >= base.duration:
                    continue
                wc = make_word_clip(w["word"], w["start"], w["end"],
                                    base.duration, font)
                if wc is not None:
                    word_clips.append(wc)
            log.info("   Subtitles: %d word clips.", len(word_clips))
        else:
            log.info("   Skipping subtitles (no whisper model / no audio).")

        # 5) Composite and export.
        layers = [base] + word_clips
        final = CompositeVideoClip(layers, size=(OUT_W, OUT_H))
        final = final.set_duration(base.duration)
        if hook_clip.audio is not None:
            final = final.set_audio(hook_clip.audio)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        final.write_videofile(
            out_path,
            fps=OUT_FPS,
            codec=OUT_CODEC,
            bitrate=OUT_BITRATE,
            audio_codec=OUT_AUDIO_CODEC,
            audio_bitrate=OUT_AUDIO_BITRATE,
            preset=OUT_PRESET,
            threads=os.cpu_count() or 4,
            temp_audiofile=os.path.join(scratch_dir,
                                        f"{entry['slot_index']:02d}_mux.m4a"),
            remove_temp=True,
            logger=None,
        )

        entry["status"] = "rendered"
        log.info("   ✔ Rendered %s", os.path.basename(out_path))
        return True

    except Exception as e:                                 # pragma: no cover
        entry["status"] = "failed"
        entry["render_error"] = repr(e)
        log.exception("   [X] Render failed for %s: %s", entry["id"], e)
        return False

    finally:
        # Explicit cleanup — critical for a long batch loop (no leaks).
        for wc in word_clips:
            try:
                wc.close()
            except Exception:
                pass
        for c in (final, base, hook_clip, src_clip):
            try:
                if c is not None:
                    c.close()
            except Exception:
                pass
        if os.path.exists(tmp_audio):
            try:
                os.remove(tmp_audio)
            except Exception:
                pass


# =========================================================================== #
#  SECTION 8 — QUEUE PERSISTENCE                                              #
# =========================================================================== #

def write_queue(plan: list, path: str):
    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "spec": {
            "resolution": f"{OUT_W}x{OUT_H}",
            "fps": OUT_FPS,
            "bitrate": OUT_BITRATE,
            "videos_per_day": len(DAILY_SLOTS),
            "days": WEEKDAYS,
            "shifts": {
                "morning": 1, "afternoon": 2, "evening": 2,
            },
            "total_per_week": len(plan),
        },
        "queue": plan,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("Queue written -> %s", path)


# =========================================================================== #
#  SECTION 9 — MAIN                                                           #
# =========================================================================== #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Autonomous YouTube Shorts factory (25 videos/week).")
    p.add_argument("--input-dir", default="./raw",
                   help="Folder with raw creator downloads.")
    p.add_argument("--output-dir", default="./out",
                   help="Folder for rendered Shorts + queue JSON.")
    p.add_argument("--queue-file", default=None,
                   help="Path to publication_queue.json "
                        "(default: <output-dir>/publication_queue.json).")
    p.add_argument("--week-start", default=None,
                   help="Monday of the target week (YYYY-MM-DD). "
                        "Default: next Monday.")
    p.add_argument("--whisper-model", default="small",
                   help="Whisper model size (tiny/base/small/medium/large).")
    p.add_argument("--font", default="Impact",
                   help="Font name for subtitles (must be known to ImageMagick).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible planning.")
    p.add_argument("--limit", type=int, default=None,
                   help="Render only the first N slots (smoke test).")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan + write JSON only; do not render any video.")
    p.add_argument("--hints-file", default="./hooks_hints.json",
                   help="Most-replayed peaks from sourcing.py (hook cutting).")
    p.add_argument("--single-day", action="store_true",
                   help="Render only 5 slots for ONE day (GitHub Actions mode).")
    p.add_argument("--date", default=None,
                   help="Target date YYYY-MM-DD for --single-day (default: today).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Resolve target week.
    if args.week_start:
        week_start = dt.date.fromisoformat(args.week_start)
        if week_start.weekday() != 0:
            log.warning("Given --week-start is not a Monday; using it anyway.")
    else:
        week_start = next_monday(dt.date.today())

    os.makedirs(args.output_dir, exist_ok=True)
    scratch_dir = os.path.join(args.output_dir, "_scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    queue_file = args.queue_file or os.path.join(args.output_dir,
                                                 "publication_queue.json")

    log.info("=" * 70)
    log.info("SHORTS FACTORY — target week starting %s (Mon)", week_start)
    log.info("=" * 70)

    # 1) Build library + plan.
    library = ContentLibrary(args.input_dir)
    if args.single_day:
        target = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
        plan = build_day_plan(library, target, args.output_dir, seed=args.seed)
    else:
        plan = build_weekly_plan(library, week_start, args.output_dir, seed=args.seed)

    # 2) Validate schedule BEFORE any heavy work.
    validate_no_overlap(plan)

    # 3) Persist the initial (scheduled) queue immediately.
    write_queue(plan, queue_file)

    # 4) Decide whether we can render.
    render_possible = _CV2_OK and _MOVIEPY_OK
    if args.dry_run:
        log.info("DRY RUN — skipping all rendering. Plan is ready.")
        _print_summary(plan)
        return 0
    if not render_possible:
        missing = []
        if not _CV2_OK:
            missing.append("opencv-python")
        if not _MOVIEPY_OK:
            missing.append("moviepy")
        log.error("Rendering unavailable (missing: %s). "
                  "Queue JSON was still written.", ", ".join(missing))
        _print_summary(plan)
        return 1

    # 5) Configure ImageMagick if provided.
    if IMAGEMAGICK_BINARY:
        try:
            from moviepy.config import change_settings
            change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY})
            log.info("ImageMagick binary set to %s", IMAGEMAGICK_BINARY)
        except Exception as e:
            log.warning("Could not set ImageMagick binary: %s", e)

    # 6) Load Whisper once (reused across the whole batch).
    whisper_model = None
    if _WHISPER_OK:
        log.info("Loading Whisper model '%s' ...", args.whisper_model)
        try:
            whisper_model = whisper.load_model(args.whisper_model)
        except Exception as e:
            log.warning("Whisper load failed (%s) — subtitles disabled.", e)
    else:
        log.warning("Whisper not installed — subtitles disabled.")

    # 7) Batch render loop.
    rng = random.Random(args.seed)
    hints = load_hints(args.hints_file)
    if hints:
        log.info("Loaded %d replay-peak hint(s) from %s.",
                 len(hints), args.hints_file)
    to_render = plan if args.limit is None else plan[:args.limit]
    ok = 0
    for i, entry in enumerate(to_render, 1):
        log.info("-" * 70)
        log.info("[%d/%d] %s | %s | %s",
                 i, len(to_render), entry["publish_at"],
                 entry["creator"], entry["title"])
        success = render_short(entry, whisper_model, args.font, scratch_dir,
                               rng, hints)
        ok += int(success)
        # Persist progress after every video (crash-safe queue).
        write_queue(plan, queue_file)

    log.info("=" * 70)
    log.info("BATCH COMPLETE: %d/%d rendered OK.", ok, len(to_render))
    _print_summary(plan)
    return 0 if ok == len(to_render) else 2


def _print_summary(plan: list):
    log.info("PUBLICATION SCHEDULE")
    log.info("-" * 70)
    current_day = None
    for e in plan:
        if e["day"] != current_day:
            current_day = e["day"]
            log.info("%s", current_day.upper())
        t = dt.datetime.fromisoformat(e["publish_at"]).strftime("%H:%M")
        log.info("   %s [%-9s] %-22s %-8s %s",
                 t, e["shift"], e["creator"], e["status"], e["title"])


if __name__ == "__main__":
    sys.exit(main())
