#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 SOURCING  ·  Auto-discovery + download + REPLAY-AWARE hook detection
==============================================================================
 The "brain" that feeds the factory. For each target creator it:
   1. Finds recent videos via the YouTube Data API (last N days).
   2. Ranks them with a multi-signal score:
        - view velocity            (is it trending now?)
        - like + comment ratios    (is it engaging?)
        - BREAKOUT vs channel base  (did it beat the channel's own median?
                                     => the algorithm favored it => retention)
        - REPLAY INTENSITY          (the public "most replayed" heatmap peak,
                                     extracted with yt-dlp — the closest thing
                                     to real retention/replay data)
   3. Downloads the top picks with yt-dlp into ./raw, named with the creator
      key so shorts_factory.py classifies them automatically.
   4. Writes hooks_hints.json mapping each file to its MOST-REPLAYED timestamp,
      so shorts_factory.py cuts the hook exactly on the viral moment.

 WHY THIS IS THE "REINVENTION"
   True audience-retention is private to the owner. But YouTube's public
   "Most replayed" graph IS exposed via yt-dlp's `heatmap`. We use its peak
   both to SCORE videos (spiky = has a clippable moment) and to CHOOSE where
   to cut. That is as close to real replay data as third parties can get.

 CREDENTIALS
   Discovery is read-only => just an API KEY (no OAuth):
     export YOUTUBE_API_KEY="AIza..."

 DEPENDENCIES
   pip install google-api-python-client yt-dlp   (+ ffmpeg on PATH)

 USAGE
   export YOUTUBE_API_KEY="AIza..."
   python sourcing.py --out ./raw --per-creator 3 --days 30
   python sourcing.py --out ./raw --dry-run
==============================================================================
"""

import os
import re
import sys
import json
import argparse
import logging
import statistics
import subprocess
import datetime as dt

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

try:
    import yt_dlp
    _YTDLP_OK = True
except Exception:                                          # pragma: no cover
    yt_dlp = None
    _YTDLP_OK = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sourcing")

# Keys MUST match shorts_factory.py so classification works.
CREATORS = [
    {"key": "einerd",     "search": "EiNerd"},
    {"key": "felipeneto", "search": "Felipe Neto"},
    {"key": "cariani",    "search": "Renato Cariani"},
    {"key": "alanzoka",   "search": "alanzoka"},
    {"key": "paulinho",   "search": "Paulinho o Loko"},
]

# Optional: pre-fill to skip channel search (saves 100 quota units each).
CHANNEL_ID_OVERRIDES = {}

CACHE_FILE = "channels_cache.json"
HINTS_FILE = "hooks_hints.json"

MIN_SRC_SEC = 60
MAX_SRC_SEC = 30 * 60
HOOK_LEAD_SEC = 4.0          # start the hook this many seconds before the peak


# --------------------------------------------------------------------------- #
#  API HELPERS                                                                 #
# --------------------------------------------------------------------------- #
def get_service(api_key: str):
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_channel_id(service, name: str, cache: dict) -> str:
    if name in cache:
        return cache[name]
    resp = service.search().list(
        q=name, part="snippet", type="channel", maxResults=1).execute()
    items = resp.get("items", [])
    if not items:
        return None
    cid = items[0]["snippet"]["channelId"]
    cache[name] = cid
    return cid


def fetch_recent_video_ids(service, channel_id, days, max_results):
    published_after = (dt.datetime.utcnow() - dt.timedelta(days=days)
                       ).strftime("%Y-%m-%dT%H:%M:%SZ")
    resp = service.search().list(
        channelId=channel_id, part="id", type="video", order="date",
        publishedAfter=published_after, maxResults=min(max_results, 50),
    ).execute()
    return [it["id"]["videoId"] for it in resp.get("items", [])
            if it["id"].get("videoId")]


def get_video_stats(service, video_ids):
    out = []
    for i in range(0, len(video_ids), 50):
        resp = service.videos().list(
            id=",".join(video_ids[i:i + 50]),
            part="statistics,contentDetails,snippet").execute()
        out.extend(resp.get("items", []))
    return out


# --------------------------------------------------------------------------- #
#  SCORING                                                                     #
# --------------------------------------------------------------------------- #
_DUR_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def iso_duration_to_sec(s: str) -> int:
    m = _DUR_RE.fullmatch(s or "")
    if not m:
        return 0
    h, mi, se = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + se


def video_velocity(item) -> float:
    st = item.get("statistics", {})
    views = int(st.get("viewCount", 0) or 0)
    if views <= 0:
        return 0.0
    pub = dt.datetime.strptime(item["snippet"]["publishedAt"],
                               "%Y-%m-%dT%H:%M:%SZ")
    age_days = max((dt.datetime.utcnow() - pub).days, 1)
    return views / age_days


def engagement_multiplier(item) -> float:
    st = item.get("statistics", {})
    views = int(st.get("viewCount", 0) or 0)
    if views <= 0:
        return 1.0
    likes = int(st.get("likeCount", 0) or 0)
    cmts = int(st.get("commentCount", 0) or 0)
    return 1.0 + 5.0 * (likes / views) + 10.0 * (cmts / views)


def get_heatmap_peak(video_id: str):
    """
    Extract the public 'most replayed' heatmap via yt-dlp.
    Returns (intensity, peak_start_seconds). intensity = peak - mean of the
    normalized replay curve (a spiky curve => a strong clippable moment).
    Returns (0.0, None) when no heatmap is available.
    """
    if not _YTDLP_OK:
        return 0.0, None
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return 0.0, None
    hm = info.get("heatmap")
    if not hm:
        return 0.0, None
    values = [h["value"] for h in hm]
    mean = sum(values) / len(values)
    peak = max(hm, key=lambda h: h["value"])
    intensity = max(0.0, peak["value"] - mean)
    return intensity, float(peak["start_time"])


# --------------------------------------------------------------------------- #
#  DOWNLOAD                                                                    #
# --------------------------------------------------------------------------- #
def download_video(video_id, creator_key, out_dir) -> bool:
    os.makedirs(out_dir, exist_ok=True)
    template = os.path.join(out_dir, f"{creator_key}_{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]",
        "--merge-output-format", "mp4",
        "--no-playlist", "--no-warnings",
        "-o", template,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except FileNotFoundError:
        log.error("yt-dlp not installed. Run: pip install yt-dlp")
        return False
    except subprocess.CalledProcessError as e:
        log.warning("Download failed for %s: %s", video_id, e)
        return False


# --------------------------------------------------------------------------- #
#  MAIN                                                                        #
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Discover + download trending sources.")
    p.add_argument("--out", default="./raw")
    p.add_argument("--per-creator", type=int, default=3)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--pool", type=int, default=25,
                   help="Candidates fetched per creator before ranking.")
    p.add_argument("--api-key", default=os.environ.get("YOUTUBE_API_KEY"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def rank_creator(service, creator, args):
    """Return a list of picks [{item, dur, score, intensity, peak_start}]."""
    key, name = creator["key"], creator["search"]
    cache = rank_creator.cache
    cid = resolve_channel_id(service, name, cache)
    if not cid:
        log.warning("   channel not found — skipping.")
        return []
    ids = fetch_recent_video_ids(service, cid, args.days, args.pool)
    if not ids:
        log.warning("   no recent videos — skipping.")
        return []
    items = get_video_stats(service, ids)

    # First pass: velocity + engagement + breakout vs channel median.
    velocities = [video_velocity(it) for it in items]
    median_vel = statistics.median([v for v in velocities if v > 0] or [1.0])
    prelim = []
    for it, vel in zip(items, velocities):
        dur = iso_duration_to_sec(it["contentDetails"]["duration"])
        if not (MIN_SRC_SEC <= dur <= MAX_SRC_SEC):
            continue
        breakout = min(max(vel / median_vel, 0.5), 3.0) if median_vel else 1.0
        base = vel * engagement_multiplier(it) * breakout
        prelim.append({"item": it, "dur": dur, "base": base, "breakout": breakout})

    prelim.sort(key=lambda x: x["base"], reverse=True)
    shortlist = prelim[: args.per_creator * 2]

    # Second pass: replay heatmap on the shortlist only (cheap, metadata-only).
    for p in shortlist:
        vid = p["item"]["id"]
        intensity, peak_start = get_heatmap_peak(vid)
        p["intensity"] = intensity
        p["peak_start"] = peak_start
        p["score"] = p["base"] * (1.0 + 2.0 * intensity)

    shortlist.sort(key=lambda x: x["score"], reverse=True)
    return shortlist[: args.per_creator]


def main(argv=None):
    args = parse_args(argv)
    if not args.api_key:
        log.error("Set YOUTUBE_API_KEY (env var) or pass --api-key.")
        return 1
    if not _YTDLP_OK:
        log.warning("yt-dlp not importable — replay heatmap disabled "
                    "(hooks will fall back to random). Run: pip install yt-dlp")

    service = get_service(args.api_key)
    cache = load_json(CACHE_FILE, {})
    cache.update(CHANNEL_ID_OVERRIDES)
    rank_creator.cache = cache

    report, hints, picks = [], {}, []
    for creator in CREATORS:
        log.info("=" * 60)
        log.info("Creator: %s", creator["search"])
        try:
            chosen = rank_creator(service, creator, args)
        except HttpError as e:
            log.error("   API error (quota?): %s", e)
            break

        for p in chosen:
            it = p["item"]
            vid, title = it["id"], it["snippet"]["title"]
            stem = f"{creator['key']}_{vid}"
            peak = p.get("peak_start")
            log.info("   ★ score=%.0f replay=%.2f peak=%s | %s",
                     p["score"], p.get("intensity", 0.0),
                     f"{peak:.0f}s" if peak is not None else "n/a", title[:50])
            picks.append((vid, creator["key"]))
            if peak is not None:
                hints[stem] = {"peak_start": round(peak, 1),
                               "intensity": round(p["intensity"], 3),
                               "lead": HOOK_LEAD_SEC}
            report.append({
                "creator": creator["key"], "video_id": vid, "title": title,
                "duration_sec": p["dur"], "score": round(p["score"], 1),
                "replay_intensity": round(p.get("intensity", 0.0), 3),
                "peak_start_sec": round(peak, 1) if peak is not None else None,
                "breakout": round(p["breakout"], 2),
                "views": int(it["statistics"].get("viewCount", 0) or 0),
            })

    save_json(cache, CACHE_FILE)
    save_json(hints, HINTS_FILE)
    save_json(report, "sourcing_report.json")

    if args.dry_run:
        log.info("=" * 60)
        log.info("DRY RUN — %d picks ranked (not downloaded). "
                 "Replay hints -> %s", len(picks), HINTS_FILE)
        return 0

    ok = 0
    for vid, key in picks:
        log.info("Downloading %s (%s) ...", vid, key)
        ok += int(download_video(vid, key, args.out))
    log.info("=" * 60)
    log.info("Downloaded %d/%d into %s | hints -> %s | report -> %s",
             ok, len(picks), args.out, HINTS_FILE, "sourcing_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
