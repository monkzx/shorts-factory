#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 PUBLISHER  ·  Uploads rendered Shorts to YouTube using publication_queue.json
==============================================================================
 - Reads the queue produced by shorts_factory.py
 - Uploads each rendered video as PRIVATE with `publishAt` = its shift time
   => YouTube itself makes the video public at the exact scheduled moment,
      so YOUR machine does NOT need to be on at 09:00 / 13:00 / 19:00 etc.
 - Sends a Telegram (or Discord) notification confirming each publication.
 - Respects the YouTube quota: videos.insert costs ~1600 units and the
   default daily quota is 10,000 => MAX ~6 uploads/day. This script is meant
   to run ONCE PER DAY and upload only that day's 5 videos (8000 units).

 SETUP (one time)
 ----------------
 1. Google Cloud Console -> create project -> enable "YouTube Data API v3".
 2. OAuth consent screen -> add yourself as Test User (or set to Production
    to avoid the 7-day refresh-token expiry in Testing mode).
 3. Credentials -> OAuth client ID -> "Desktop app" -> download JSON as
    client_secrets.json (place next to this script).
 4. First run (on a machine WITH a browser) to mint token.json:
        python publisher.py --auth
    Copy client_secrets.json + token.json to the server afterwards.

 DEPENDENCIES
 ------------
    pip install google-api-python-client google-auth-oauthlib google-auth-httplib2

 DAILY USAGE (on the server, via cron)
 -------------------------------------
    python publisher.py --queue ./out/publication_queue.json \
                        --date today --tz America/Sao_Paulo
==============================================================================
"""

import os
import sys
import json
import time
import argparse
import logging
import datetime as dt
import urllib.parse
import urllib.request

try:
    from zoneinfo import ZoneInfo            # Python 3.9+
except Exception:                            # pragma: no cover
    ZoneInfo = None

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("publisher")

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
YOUTUBE_CATEGORY_ID = "24"          # 24 = Entertainment
MAX_UPLOADS_PER_DAY = 6             # quota safety cap (6 * 1600 < 10000)


# --------------------------------------------------------------------------- #
#  AUTH                                                                        #
# --------------------------------------------------------------------------- #
def get_youtube_service(client_secrets: str, token_path: str, interactive: bool):
    """Return an authenticated YouTube API client (refreshing tokens headlessly)."""
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log.info("Refreshing OAuth token ...")
            creds.refresh(Request())
        else:
            if not interactive:
                raise RuntimeError(
                    "No valid token.json. Run once with --auth on a machine "
                    "with a browser to create it, then copy it to the server."
                )
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        log.info("Saved credentials -> %s", token_path)

    return build("youtube", "v3", credentials=creds)


# --------------------------------------------------------------------------- #
#  TIME HANDLING                                                               #
# --------------------------------------------------------------------------- #
def to_rfc3339_utc(naive_local_iso: str, tz_name: str) -> str:
    """
    Convert a naive local ISO timestamp (as stored in the queue) into an
    RFC 3339 UTC string ('...Z') required by YouTube's publishAt field.
    """
    local_dt = dt.datetime.fromisoformat(naive_local_iso)
    if ZoneInfo is not None:
        local_dt = local_dt.replace(tzinfo=ZoneInfo(tz_name))
        utc_dt = local_dt.astimezone(dt.timezone.utc)
    else:                                    # pragma: no cover
        utc_dt = local_dt                    # assume already UTC (fallback)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
#  UPLOAD                                                                      #
# --------------------------------------------------------------------------- #
def upload_video(service, entry: dict, tz_name: str) -> str:
    """Upload one video as private with publishAt; return the YouTube video id."""
    path = entry["output_file"]
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"rendered file missing: {path}")

    publish_at = to_rfc3339_utc(entry["publish_at"], tz_name)
    # Tags cannot contain '#'; hashtags stay in the description.
    tags = [h.lstrip("#") for h in entry.get("hashtags", [])]

    body = {
        "snippet": {
            "title": entry["title"][:100],           # YT hard limit = 100 chars
            "description": entry["description"][:4900],
            "tags": tags[:15],
            "categoryId": YOUTUBE_CATEGORY_ID,
        },
        "status": {
            "privacyStatus": "private",              # required for publishAt
            "publishAt": publish_at,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(path, chunksize=8 * 1024 * 1024,
                            resumable=True, mimetype="video/*")
    request = service.videos().insert(
        part="snippet,status", body=body, media_body=media)

    response = None
    retries = 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                log.info("   ... %d%% uploaded", int(status.progress() * 100))
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and retries < 5:
                retries += 1
                wait = 2 ** retries
                log.warning("   transient error %s — retry in %ds", e.resp.status, wait)
                time.sleep(wait)
                continue
            raise
    return response["id"]


# --------------------------------------------------------------------------- #
#  NOTIFICATIONS                                                               #
# --------------------------------------------------------------------------- #
def notify(text: str):
    """
    Fire a notification via Telegram and/or Discord if the corresponding
    environment variables are set. Silent no-op otherwise.
    """
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat:
        try:
            url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            data = urllib.parse.urlencode(
                {"chat_id": tg_chat, "text": text,
                 "parse_mode": "HTML", "disable_web_page_preview": "true"}
            ).encode()
            urllib.request.urlopen(url, data=data, timeout=15)
        except Exception as e:                          # pragma: no cover
            log.warning("Telegram notify failed: %s", e)

    dc_hook = os.environ.get("DISCORD_WEBHOOK_URL")
    if dc_hook:
        try:
            data = json.dumps({"content": text}).encode()
            req = urllib.request.Request(
                dc_hook, data=data,
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=15)
        except Exception as e:                          # pragma: no cover
            log.warning("Discord notify failed: %s", e)


# --------------------------------------------------------------------------- #
#  QUEUE SELECTION                                                             #
# --------------------------------------------------------------------------- #
def load_queue(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_queue(payload: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def select_for_date(queue: list, target_date: dt.date) -> list:
    """Return rendered, not-yet-uploaded entries whose publish date == target."""
    out = []
    for e in queue:
        pub = dt.datetime.fromisoformat(e["publish_at"]).date()
        already = e.get("youtube_id") or e["status"] in ("uploaded", "published")
        if pub == target_date and e["status"] == "rendered" and not already:
            out.append(e)
    return out


# --------------------------------------------------------------------------- #
#  MAIN                                                                        #
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Publish rendered Shorts to YouTube.")
    p.add_argument("--queue", default="./out/publication_queue.json")
    p.add_argument("--client-secrets", default="./client_secrets.json")
    p.add_argument("--token", default="./token.json")
    p.add_argument("--tz", default="America/Sao_Paulo",
                   help="Timezone of the timestamps stored in the queue.")
    p.add_argument("--date", default="today",
                   help="'today', 'tomorrow', or YYYY-MM-DD.")
    p.add_argument("--auth", action="store_true",
                   help="Run the interactive OAuth flow to create token.json, "
                        "then exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be uploaded, but do not upload.")
    return p.parse_args(argv)


def resolve_date(spec: str) -> dt.date:
    if spec == "today":
        return dt.date.today()
    if spec == "tomorrow":
        return dt.date.today() + dt.timedelta(days=1)
    return dt.date.fromisoformat(spec)


def main(argv=None):
    args = parse_args(argv)

    # --auth: just mint the token and exit.
    if args.auth:
        get_youtube_service(args.client_secrets, args.token, interactive=True)
        log.info("Auth complete. token.json is ready — copy it to the server.")
        return 0

    payload = load_queue(args.queue)
    queue = payload["queue"]
    target = resolve_date(args.date)

    todo = select_for_date(queue, target)
    if not todo:
        log.info("Nothing to publish for %s.", target)
        return 0

    if len(todo) > MAX_UPLOADS_PER_DAY:
        log.warning("Found %d uploads for %s but quota caps at %d — "
                    "trimming to stay within daily quota.",
                    len(todo), target, MAX_UPLOADS_PER_DAY)
        todo = todo[:MAX_UPLOADS_PER_DAY]

    log.info("Publishing %d video(s) for %s (tz=%s).",
             len(todo), target, args.tz)

    if args.dry_run:
        for e in todo:
            log.info("   [DRY] %s -> publishAt %s | %s",
                     os.path.basename(e["output_file"]),
                     to_rfc3339_utc(e["publish_at"], args.tz), e["title"])
        return 0

    service = get_youtube_service(args.client_secrets, args.token,
                                  interactive=False)

    ok = 0
    for e in todo:
        try:
            log.info("Uploading: %s", os.path.basename(e["output_file"]))
            vid = upload_video(service, e, args.tz)
            e["youtube_id"] = vid
            e["status"] = "uploaded"
            e["youtube_url"] = f"https://youtu.be/{vid}"
            ok += 1
            when = dt.datetime.fromisoformat(e["publish_at"]).strftime("%d/%m %H:%M")
            notify(f"✅ <b>Agendado no YouTube</b>\n"
                   f"{e['title']}\n"
                   f"🕒 Vai ao ar: {when}\n"
                   f"🔗 https://youtu.be/{vid}")
            log.info("   ✔ id=%s (scheduled for %s)", vid, e["publish_at"])
        except Exception as ex:                          # pragma: no cover
            e["status"] = "publish_failed"
            e["publish_error"] = repr(ex)
            log.exception("   [X] upload failed: %s", ex)
            notify(f"❌ Falha ao publicar: {e['title']}\n{ex}")
        finally:
            save_queue(payload, args.queue)     # persist after each upload

    log.info("Done: %d/%d uploaded.", ok, len(todo))
    notify(f"📊 Publicação diária concluída: {ok}/{len(todo)} vídeos agendados "
           f"para {target}.")
    return 0 if ok == len(todo) else 2


if __name__ == "__main__":
    sys.exit(main())
