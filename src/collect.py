"""
collect.py — Step 0: YouTube Data API crawler
═══════════════════════════════════════════════════════════════════════════════

Collects videos and comments across 5 content categories without using any
depression-related keywords to filter or rank content. Depression relevance
is determined entirely post-hoc in Steps 1–2.

Category sampling strategy
──────────────────────────
  mental_health        search-based  ("mental health vlog", "therapy journey" …)
  fitness_wellness     search-based  ("workout routine", "wellness journey" …)
  music_entertainment  chart-based   (YouTube category 10 — Music, global chart)
  news_current_events  chart-based   (YouTube category 25 — News & Politics)
  personal_vlog        search-based  ("day in my life vlog", "life update" …)

Within each category, videos are ranked by view count; the top TARGET_PER_CATEGORY
are selected. Per-video comment collection: top COMMENTS_PER_VIDEO comments
sorted by relevance (approximates like-count ranking).

Outputs
───────
  data/raw/videos.json    — list of video objects with metadata
  data/raw/comments.json  — list of comment objects with video_id foreign key

Quota budget  (YouTube Data API v3 limit: 10,000 units / day)
─────────────
  search.list   = 100 units/call  ×  ~15 calls   =  ~1,500 units (Phase 1 search)
  videos.list   =   1 unit/call   ×  ~10 calls   =    ~10 units  (Phase 1 chart)
  videos.list   =   1 unit/call   ×  ~10 calls   =    ~10 units  (Phase 2 stats)
  commentThreads=   1 unit/call   × ~250 calls   =   ~250 units  (Phase 3)
  ─────────────────────────────────────────────────────────────────
  Total                                           ≈  1,770 units  (18 % of daily cap)

Rate limit: CALL_INTERVAL seconds between every API call.
Progress is check-pointed after every call; safe to interrupt and resume.
"""

import json
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytz
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ── Load API key from .env ────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
API_KEY = os.getenv("YOUTUBE_API_KEY", "")
if not API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY not set. Add it to .env in the project root.")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parents[1]
RAW_DIR         = ROOT / "data" / "raw"
VIDEOS_FILE     = RAW_DIR / "videos.json"
COMMENTS_FILE   = RAW_DIR / "comments.json"
CHECKPOINT_FILE = RAW_DIR / "checkpoint.json"
LOG_FILE        = RAW_DIR / "collect.log"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Collection parameters ─────────────────────────────────────────────────────
TARGET_PER_CATEGORY = 50    # videos per category
COMMENTS_PER_VIDEO  = 50    # top-N comments per video (one API call, no pagination)
CALL_INTERVAL       = 3     # seconds between API calls
DAILY_QUOTA         = 10_000
QUOTA_BUFFER        = 300   # stop this many units before the hard cap

# ── Category definitions ──────────────────────────────────────────────────────
# search-based: use broad topical queries; order=viewCount picks most-viewed first
# chart-based:  YouTube mostPopular chart filtered by official category ID
CATEGORIES = {
    "mental_health": {
        "method": "search",
        "queries": [
            "mental health vlog",
            "therapy journey",
            "anxiety story",
            "mental illness awareness",
            "wellness mental health",
        ],
        "label": "Mental Health",
    },
    "fitness_wellness": {
        "method": "search",
        "queries": [
            "workout routine",
            "fitness vlog",
            "wellness journey",
            "healthy lifestyle vlog",
            "gym motivation",
        ],
        "label": "Fitness / Wellness",
    },
    "music_entertainment": {
        "method": "chart",
        "category_id": "10",   # Music
        "label": "Music / Entertainment",
    },
    "news_current_events": {
        "method": "chart",
        "category_id": "25",   # News & Politics
        "label": "News / Current Events",
    },
    "personal_vlog": {
        "method": "search",
        "queries": [
            "day in my life vlog",
            "life update vlog",
            "weekly vlog",
            "college vlog",
            "moving vlog",
        ],
        "label": "Personal Vlog",
    },
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Quota / rate-limit helpers ────────────────────────────────────────────────
PACIFIC = pytz.timezone("America/Los_Angeles")

def today_pt() -> str:
    return datetime.now(PACIFIC).strftime("%Y-%m-%d")

def sleep_until_quota_reset():
    now_pt        = datetime.now(PACIFIC)
    next_midnight = (now_pt + timedelta(days=1)).replace(
        hour=0, minute=1, second=0, microsecond=0
    )
    wait = max(0, next_midnight.timestamp() - time.time()) + 10
    log.warning("Quota exhausted — sleeping %.0f min until midnight PT …", wait / 60)
    time.sleep(wait)

def reset_quota_if_new_day(cp: dict):
    today = today_pt()
    if cp.get("quota_date") != today:
        log.info("New quota day: %s — resetting counter.", today)
        cp["quota_used_today"] = 0
        cp["quota_date"]       = today

def quota_remaining(cp: dict) -> int:
    return DAILY_QUOTA - cp["quota_used_today"] - QUOTA_BUFFER

def charge(cp: dict, units: int):
    cp["quota_used_today"] += units

def rate_wait(cp: dict, cost: int, client_ref: list, label: str):
    reset_quota_if_new_day(cp)
    if quota_remaining(cp) < cost:
        save_checkpoint(cp)
        sleep_until_quota_reset()
        client_ref[0] = build_client()
        reset_quota_if_new_day(cp)
    log.info("%s | quota used: %d", label, cp["quota_used_today"])
    time.sleep(CALL_INTERVAL)

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def empty_checkpoint() -> dict:
    return {
        # Phase 1 state: {cat_key: {"ids": [...], "done": bool, "query_idx": int, "page_token": str|None}}
        "p1": {cat: {"ids": [], "done": False, "query_idx": 0, "page_token": None}
               for cat in CATEGORIES},
        # Phase 2 state: {video_id: video_dict}
        "p2_videos": {},
        "p2_done": False,
        # Phase 3 state
        "p3_done_ids": [],
        "quota_used_today": 0,
        "quota_date": "",
    }

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return empty_checkpoint()

def save_checkpoint(cp: dict):
    tmp = str(CHECKPOINT_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False)
    os.replace(tmp, str(CHECKPOINT_FILE))

# ── API client ────────────────────────────────────────────────────────────────
def build_client():
    return build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1 — Collect video IDs per category
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def phase1_collect_ids(cp: dict):
    client_ref = [build_client()]

    for cat_key, cat_cfg in CATEGORIES.items():
        state  = cp["p1"][cat_key]
        ids    = set(state["ids"])
        target = TARGET_PER_CATEGORY

        if state["done"]:
            log.info("Phase 1 | %-25s already done (%d IDs)", cat_cfg["label"], len(ids))
            continue

        log.info("Phase 1 | %-25s target=%d, have=%d", cat_cfg["label"], target, len(ids))

        if cat_cfg["method"] == "chart":
            # ── chart-based: one page per call, paginate until target ──
            page_token = state["page_token"]
            while len(ids) < target:
                rate_wait(cp, 1, client_ref,
                          f"Phase1 chart {cat_cfg['label']}")
                try:
                    resp = client_ref[0].videos().list(
                        part="id",
                        chart="mostPopular",
                        videoCategoryId=cat_cfg["category_id"],
                        maxResults=50,
                        **({"pageToken": page_token} if page_token else {}),
                    ).execute()
                except HttpError as e:
                    log.error("Phase 1 chart error %s: %s", cat_key, e)
                    if e.resp.status in (403, 429):
                        save_checkpoint(cp)
                        sleep_until_quota_reset()
                        client_ref[0] = build_client()
                    break

                charge(cp, 1)
                new = [item["id"] for item in resp.get("items", [])]
                ids.update(new)
                log.info("  got %d (total %d)", len(new), len(ids))

                page_token = resp.get("nextPageToken")
                state["page_token"] = page_token

                if not page_token:
                    break

        else:
            # ── search-based: iterate queries; order=viewCount ──
            queries = cat_cfg["queries"]
            qi      = state["query_idx"]
            page_token = state["page_token"]

            while len(ids) < target and qi < len(queries):
                query = queries[qi]
                rate_wait(cp, 100, client_ref,
                          f"Phase1 search '{query}' ({cat_cfg['label']})")
                try:
                    resp = client_ref[0].search().list(
                        part="id",
                        q=query,
                        type="video",
                        order="viewCount",
                        maxResults=50,
                        safeSearch="none",
                        **({"pageToken": page_token} if page_token else {}),
                    ).execute()
                except HttpError as e:
                    log.error("Phase 1 search error %s q='%s': %s", cat_key, query, e)
                    if e.resp.status in (403, 429):
                        save_checkpoint(cp)
                        sleep_until_quota_reset()
                        client_ref[0] = build_client()
                    break

                charge(cp, 100)
                new = [
                    item["id"]["videoId"]
                    for item in resp.get("items", [])
                    if item.get("id", {}).get("kind") == "youtube#video"
                ]
                ids.update(new)
                log.info("  query='%s' got %d (total %d)", query, len(new), len(ids))

                page_token = resp.get("nextPageToken")
                # move to next query if this one is exhausted or we have enough
                if not page_token or len(ids) >= target:
                    qi += 1
                    page_token = None

                state["query_idx"]  = qi
                state["page_token"] = page_token

        # cap at target
        state["ids"]  = list(ids)[:target]
        state["done"] = True
        save_checkpoint(cp)
        log.info("Phase 1 | %-25s done — %d IDs", cat_cfg["label"], len(state["ids"]))

    log.info("Phase 1 complete.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2 — Fetch video metadata + statistics; write videos.json
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def phase2_fetch_metadata(cp: dict):
    client_ref = [build_client()]

    # Build list of (video_id, category_key) pairs not yet fetched
    already = set(cp["p2_videos"].keys())
    todo: list[tuple[str, str]] = []
    for cat_key, state in cp["p1"].items():
        for vid in state["ids"]:
            if vid not in already:
                todo.append((vid, cat_key))

    log.info("Phase 2 | fetching metadata for %d videos (%d already done)",
             len(todo), len(already))

    for i in range(0, len(todo), 50):
        batch      = todo[i : i + 50]
        batch_ids  = [v for v, _ in batch]
        batch_cats = {v: c for v, c in batch}

        rate_wait(cp, 1, client_ref,
                  f"Phase2 batch {i//50+1}/{(len(todo)+49)//50}")
        try:
            resp = client_ref[0].videos().list(
                part="snippet,statistics",
                id=",".join(batch_ids),
                maxResults=50,
            ).execute()
        except HttpError as e:
            log.error("Phase 2 HttpError: %s", e)
            if e.resp.status in (403, 429):
                save_checkpoint(cp)
                sleep_until_quota_reset()
                client_ref[0] = build_client()
            continue

        charge(cp, 1)

        for item in resp.get("items", []):
            vid     = item["id"]
            snippet = item.get("snippet", {})
            stats   = item.get("statistics", {})
            cp["p2_videos"][vid] = {
                "video_id":      vid,
                "category":      batch_cats.get(vid, "unknown"),
                "category_label": CATEGORIES.get(batch_cats.get(vid, ""), {}).get("label", ""),
                "title":         snippet.get("title", ""),
                "channel":       snippet.get("channelTitle", ""),
                "published_at":  snippet.get("publishedAt", ""),
                "description":   snippet.get("description", ""),
                "tags":          snippet.get("tags", []),
                "view_count":    int(stats.get("viewCount",    0)),
                "like_count":    int(stats.get("likeCount",    0)),
                "comment_count": int(stats.get("commentCount", 0)),
            }

        save_checkpoint(cp)

    # Write videos.json sorted by category then view_count
    all_videos = sorted(
        cp["p2_videos"].values(),
        key=lambda v: (v["category"], -v["view_count"]),
    )
    with open(VIDEOS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_videos, f, ensure_ascii=False, indent=2)

    cp["p2_done"] = True
    save_checkpoint(cp)

    # Summary
    from collections import Counter
    dist = Counter(v["category_label"] for v in all_videos)
    log.info("Phase 2 complete. %d videos written to %s", len(all_videos), VIDEOS_FILE)
    for cat, cnt in dist.most_common():
        log.info("  %-30s %d videos", cat, cnt)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3 — Collect top-50 comments per video; write comments.json
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def phase3_collect_comments(cp: dict):
    client_ref = [build_client()]

    done_ids = set(cp["p3_done_ids"])
    videos   = [v for v in cp["p2_videos"].values() if v["video_id"] not in done_ids]

    log.info("Phase 3 | collecting top-%d comments for %d videos (%d already done)",
             COMMENTS_PER_VIDEO, len(videos), len(done_ids))

    # Load existing comments so we can append
    existing: list[dict] = []
    if COMMENTS_FILE.exists():
        with open(COMMENTS_FILE, encoding="utf-8") as f:
            existing = json.load(f)
    existing_ids = {c["comment_id"] for c in existing}

    buf: list[dict] = []

    for video in videos:
        vid_id = video["video_id"]

        rate_wait(cp, 1, client_ref,
                  f"Phase3 {video['category_label']} | {vid_id}")
        try:
            resp = client_ref[0].commentThreads().list(
                part="snippet",
                videoId=vid_id,
                maxResults=COMMENTS_PER_VIDEO,
                order="relevance",   # most-liked comments first
            ).execute()
        except HttpError as e:
            if e.resp.status == 403:
                body   = json.loads(e.content)
                reason = body.get("error", {}).get("errors", [{}])[0].get("reason", "")
                if reason == "commentsDisabled":
                    log.info("  Comments disabled — %s, skipping.", vid_id)
                    cp["p3_done_ids"].append(vid_id)
                    save_checkpoint(cp)
                    continue
                save_checkpoint(cp)
                sleep_until_quota_reset()
                client_ref[0] = build_client()
                reset_quota_if_new_day(cp)
                continue
            log.error("Phase 3 HttpError vid=%s: %s", vid_id, e)
            cp["p3_done_ids"].append(vid_id)
            save_checkpoint(cp)
            continue

        charge(cp, 1)
        items = resp.get("items", [])

        for item in items:
            cid = item["id"]
            if cid in existing_ids:
                continue
            top = item["snippet"]["topLevelComment"]["snippet"]
            buf.append({
                # Comment fields
                "comment_id":   cid,
                "text":         top.get("textDisplay", ""),
                "author":       top.get("authorDisplayName", ""),
                "like_count":   top.get("likeCount", 0),
                "published_at": top.get("publishedAt", ""),
                "collected_at": datetime.now(timezone.utc).isoformat(),
                # Video context (foreign key + denormalised for convenience)
                "video_id":          vid_id,
                "category":          video["category"],
                "category_label":    video["category_label"],
                "video_title":       video["title"],
                "video_channel":     video["channel"],
                "video_description": video["description"],
                "video_tags":        video["tags"],
                "view_count":        video["view_count"],
            })

        log.info("  +%d comments | video: %s", len(items), video["title"][:55])
        cp["p3_done_ids"].append(vid_id)
        save_checkpoint(cp)

    # Merge and write
    all_comments = existing + buf
    with open(COMMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_comments, f, ensure_ascii=False, indent=2)

    from collections import Counter
    dist = Counter(c["category_label"] for c in all_comments)
    log.info("Phase 3 complete. %d comments written to %s", len(all_comments), COMMENTS_FILE)
    for cat, cnt in dist.most_common():
        log.info("  %-30s %d comments", cat, cnt)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    log.info("=" * 65)
    log.info("YouTube Depression Signal — collect.py")
    log.info("Categories: %s", ", ".join(c["label"] for c in CATEGORIES.values()))
    log.info("Target: %d videos/category × %d comments/video = ~%d comments",
             TARGET_PER_CATEGORY, COMMENTS_PER_VIDEO,
             len(CATEGORIES) * TARGET_PER_CATEGORY * COMMENTS_PER_VIDEO)
    log.info("=" * 65)

    cp = load_checkpoint()
    reset_quota_if_new_day(cp)

    # Phase 1
    all_done = all(cp["p1"][cat]["done"] for cat in CATEGORIES)
    if not all_done:
        phase1_collect_ids(cp)
    else:
        total = sum(len(cp["p1"][c]["ids"]) for c in CATEGORIES)
        log.info("Phase 1 already complete (%d total IDs). Skipping.", total)

    # Phase 2
    if not cp.get("p2_done"):
        phase2_fetch_metadata(cp)
    else:
        log.info("Phase 2 already complete (%d videos). Skipping.", len(cp["p2_videos"]))

    # Phase 3
    total_videos = len(cp["p2_videos"])
    done_videos  = len(cp["p3_done_ids"])
    if done_videos < total_videos:
        phase3_collect_comments(cp)
    else:
        log.info("Phase 3 already complete. Nothing to do.")

    log.info("Done.  videos  → %s", VIDEOS_FILE)
    log.info("       comments → %s", COMMENTS_FILE)

if __name__ == "__main__":
    main()
