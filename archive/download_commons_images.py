#!/usr/bin/env python3
"""Download freely-licensed images from Wikimedia Commons per class.

Creates a classification-style folder structure:
  <out>/train/<class>/*.jpg
  <out>/val/<class>/*.jpg

Also writes a manifest CSV with:
- class
- filename
- source page title
- file URL
- license + attribution (from extmetadata when available)

This is useful for bootstrapping a dataset for later manual labeling (detection) or
for image classification experiments.

NOTE: Always keep attribution/license info when using the images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml

API = "https://commons.wikimedia.org/w/api.php"
ALLOWED_LICENSE_KEYWORDS = [
    "CC BY",
    "CC BY-SA",
    "CC0",
    "Public domain",
    "GNU",
]

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s.strip("_")


def _api_get(params: dict[str, Any], session: requests.Session) -> dict[str, Any]:
    # Wikimedia APIs may block generic clients without a proper User-Agent.
    last_err = None
    for attempt in range(1, 6):
        try:
            r = session.get(API, params=params, timeout=30)
            if r.status_code in (429, 403):
                # Backoff and retry (rate limit / UA issues / transient blocks)
                time.sleep(min(10, 1.5 * attempt))
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(min(10, 1.5 * attempt))
    raise last_err


def _search_files(query: str, session: requests.Session, limit: int = 25) -> list[dict[str, Any]]:
    data = _api_get(
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srnamespace": 6,  # File:
            "srlimit": limit,
        },
        session,
    )
    return data.get("query", {}).get("search", [])


def _imageinfo_for_titles(titles: list[str], session: requests.Session) -> list[dict[str, Any]]:
    if not titles:
        return []

    data = _api_get(
        {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "titles": "|".join(titles),
            "iiprop": "url|extmetadata",
        },
        session,
    )

    pages = data.get("query", {}).get("pages", {})
    out: list[dict[str, Any]] = []
    for _pid, page in pages.items():
        out.append(page)
    return out


def _pick_license(meta: dict[str, Any]) -> str:
    # extmetadata keys vary; LicenseShortName is usually present
    for key in ["LicenseShortName", "License", "UsageTerms"]:
        v = meta.get(key, {}).get("value")
        if v:
            return re.sub(r"\s+", " ", str(v)).strip()
    return ""


def _pick_attribution(meta: dict[str, Any]) -> str:
    for key in ["Attribution", "Artist", "Credit"]:
        v = meta.get(key, {}).get("value")
        if v:
            v = re.sub(r"\s+", " ", str(v)).strip()
            return v
    return ""


def _license_allowed(lic: str) -> bool:
    if not lic:
        return False
    return any(k.lower() in lic.lower() for k in ALLOWED_LICENSE_KEYWORDS)


def _download(url: str, dst: Path, session: requests.Session, max_mb: int = 15, sleep_s: float = 0.8) -> bool:
    # Stream download with size cap + retry/backoff on 429.
    last_err = None
    for attempt in range(1, 6):
        try:
            with session.get(url, stream=True, timeout=60) as r:
                if r.status_code == 429:
                    ra = r.headers.get('Retry-After')
                    wait = float(ra) if ra and ra.isdigit() else min(30.0, 2.0 * attempt)
                    time.sleep(wait)
                    last_err = RuntimeError(f"429 rate limited, waited {wait}s")
                    continue

                r.raise_for_status()
                total = 0
                dst.parent.mkdir(parents=True, exist_ok=True)
                with dst.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_mb * 1024 * 1024:
                            return False
                        f.write(chunk)

            # polite throttling
            time.sleep(max(0.0, sleep_s))
            return True
        except Exception as e:
            last_err = e
            time.sleep(min(30.0, 2.0 * attempt))

    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Download CC images from Wikimedia Commons per class")
    ap.add_argument("--config", type=str, default="commons_queries.yaml")
    ap.add_argument("--out", type=str, default=str(Path("datasets") / "herbs_disease_cls"))
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--search-limit", type=int, default=50, help="Files to consider per query")
    ap.add_argument("--sleep", type=float, default=0.8, help="Seconds to sleep between file downloads")

    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    classes: dict[str, list[str]] = cfg.get("classes", {})

    out = Path(args.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    manifest = out / "manifest.csv"

    with requests.Session() as sess, manifest.open("w", newline="", encoding="utf-8") as f:
        sess.headers.update({
            # Required by Wikimedia API etiquette; reduces chance of 403.
            "User-Agent": "sera-hidroponica/1.0 (https://github.com/Iustin02/sera-hidroponica)",
            "Accept": "application/json",
        })
        w = csv.writer(f)
        w.writerow([
            "class",
            "split",
            "filename",
            "commons_title",
            "file_url",
            "license",
            "attribution",
            "source_queries",
        ])

        for cls, queries in classes.items():
            cls_slug = _slug(cls)
            candidates: list[tuple[str, str]] = []  # (title, query)

            for q in queries:
                res = _search_files(q, sess, limit=args.search_limit)
                for it in res:
                    title = it.get("title")
                    if title:
                        candidates.append((title, q))
                time.sleep(0.2)

            # De-dup while keeping query info (pick first query occurrence)
            seen = set()
            dedup: list[tuple[str, str]] = []
            for title, q in candidates:
                if title in seen:
                    continue
                seen.add(title)
                dedup.append((title, q))

            random.shuffle(dedup)

            need = args.per_class
            downloaded: list[tuple[Path, str, str, str, str, str]] = []

            # Fetch imageinfo in batches
            titles = [t for t, _q in dedup]
            for i in range(0, len(titles), 25):
                batch = titles[i : i + 25]
                pages = _imageinfo_for_titles(batch, sess)

                for page in pages:
                    if len(downloaded) >= need:
                        break
                    title = page.get("title", "")
                    ii = (page.get("imageinfo") or [{}])[0]
                    url = ii.get("url")
                    meta = ii.get("extmetadata") or {}
                    lic = _pick_license(meta)
                    attr = _pick_attribution(meta)

                    if not url or not _license_allowed(lic):
                        continue

                    # Filter to actual raster images (avoid PDFs, etc.)
                    ext = os.path.splitext(url)[1].lower()
                    if ext not in ALLOWED_IMAGE_EXTS:
                        continue

                    # Stable filename based on URL hash to avoid collisions
                    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
                    fname = f"{cls_slug}_{h}{ext}"

                    # Download to a temp location first
                    tmp = out / "_tmp" / fname
                    ok = _download(url, tmp, sess, sleep_s=args.sleep)
                    if not ok:
                        if tmp.exists():
                            tmp.unlink(missing_ok=True)
                        continue

                    # Decide split
                    split = "val" if random.random() < args.val_ratio else "train"
                    dst = out / split / cls_slug / fname
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp.replace(dst)

                    # Find query used (best-effort)
                    q_used = ""
                    for t, q in dedup:
                        if t == title:
                            q_used = q
                            break

                    w.writerow([cls_slug, split, fname, title, url, lic, attr, q_used])
                    downloaded.append((dst, split, fname, title, url, lic))

                if len(downloaded) >= need:
                    break

            # Cleanup tmp folder if empty
            tmp_dir = out / "_tmp"
            if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                tmp_dir.rmdir()

            print(f"{cls_slug}: downloaded {len(downloaded)}/{need}")

    print("Done.")
    print("Output:", out)
    print("Manifest:", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
