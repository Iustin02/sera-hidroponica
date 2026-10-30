#!/usr/bin/env python3
"""download_inat_yolo.py

Descarcă imagini din iNaturalist și construiește un dataset YOLO (detect) cu bbox full-frame.

Use-case: busuioc vs mentă (2 clase) ca un task de "classification via detection".

Exemplu:
  python download_inat_yolo.py --out inat_datasets\\inat_basil_mint_20k --per-class 10000 --val-per-class 1000 --workers 16

Note:
- Pentru "license any" NU setăm `photo_license` (echivalent: fără filtru).
- Folosim `large` URL (nu `original`) ca să nu explodeze dimensiunea.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm
import yaml

INAT_OBS_API = "https://api.inaturalist.org/v1/observations"
IMG_EXT_FALLBACK = ".jpg"


@dataclass(frozen=True)
class PhotoItem:
    class_id: int
    class_name: str
    split: str  # 'train' | 'val'
    photo_id: int
    obs_id: int
    url: str


def _photo_download_url(photo: dict) -> Optional[str]:
    """Returnează un URL descărcabil, preferând 'large' pentru mărime rezonabilă."""
    url = photo.get("url")
    if not url:
        return None

    # iNat: .../square.jpg | small.jpg | medium.jpg | large.jpg | original.jpg
    for size in ("square.", "small.", "medium."):
        url = url.replace(size, "large.")

    return url


def _file_ext_from_url(url: str) -> str:
    p = urlparse(url).path
    ext = Path(p).suffix.lower()
    return ext if ext else IMG_EXT_FALLBACK


def _load_manifest(manifest_csv: Path) -> Tuple[Set[int], Dict[Tuple[int, str], int]]:
    """Returnează (photo_ids_already_downloaded, counts[(class_id, split)])."""
    done: Set[int] = set()
    counts: Dict[Tuple[int, str], int] = {}

    if not manifest_csv.exists():
        return done, counts

    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pid = int(row["photo_id"])
                cid = int(row["class_id"])
                split = row["split"]
            except Exception:
                continue
            done.add(pid)
            counts[(cid, split)] = counts.get((cid, split), 0) + 1

    return done, counts


def _append_manifest_rows(manifest_csv: Path, rows: List[dict]) -> None:
    is_new = not manifest_csv.exists()
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "class_id",
        "class_name",
        "split",
        "photo_id",
        "obs_id",
        "url",
        "image_path",
        "label_path",
        "bytes",
    ]

    with manifest_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _fetch_observations_page(
    taxon_id: int,
    page: int,
    per_page: int = 200,
    photo_license: Optional[str] = None,
    quality_grade: Optional[str] = None,  # 'research' | 'needs_id' | 'casual'
    verifiable: bool = False,
    timeout_s: int = 30,
) -> dict:
    params = {
        "taxon_id": taxon_id,
        "photos": "true",
        "order_by": "created_at",
        "order": "desc",
        "page": page,
        "per_page": per_page,
    }
    if verifiable:
        params["verifiable"] = "true"
    if quality_grade:
        params["quality_grade"] = quality_grade
    if photo_license:
        params["photo_license"] = photo_license

    r = requests.get(INAT_OBS_API, params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _choose_split(need_train: int, need_val: int, rng: random.Random) -> str:
    if need_train <= 0:
        return "val"
    if need_val <= 0:
        return "train"

    # Pondere proporțională cu ce mai lipsește
    p_val = need_val / (need_train + need_val)
    return "val" if rng.random() < p_val else "train"


def _download_one(item: PhotoItem, out_root: Path, timeout_s: int = 60, retries: int = 3) -> Optional[dict]:
    ext = _file_ext_from_url(item.url)
    fname = f"{item.class_name}_{item.photo_id}{ext}"

    img_dir = out_root / "images" / item.split
    lbl_dir = out_root / "labels" / item.split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    img_path = img_dir / fname
    lbl_path = lbl_dir / f"{Path(fname).stem}.txt"

    if img_path.exists() and lbl_path.exists():
        try:
            size = img_path.stat().st_size
        except OSError:
            size = 0
        return {
            "class_id": item.class_id,
            "class_name": item.class_name,
            "split": item.split,
            "photo_id": item.photo_id,
            "obs_id": item.obs_id,
            "url": item.url,
            "image_path": str(img_path),
            "label_path": str(lbl_path),
            "bytes": size,
        }

    headers = {"User-Agent": "sera-hidroponica-downloader/1.0"}
    last_err = None
    for _ in range(retries):
        try:
            r = requests.get(item.url, stream=True, timeout=timeout_s, headers=headers)
            r.raise_for_status()

            tmp_path = img_path.with_suffix(img_path.suffix + ".tmp")
            total = 0
            with tmp_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    total += len(chunk)

            tmp_path.replace(img_path)

            # YOLO label: class x_center y_center width height (normalized)
            lbl_path.write_text(f"{item.class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")

            return {
                "class_id": item.class_id,
                "class_name": item.class_name,
                "split": item.split,
                "photo_id": item.photo_id,
                "obs_id": item.obs_id,
                "url": item.url,
                "image_path": str(img_path),
                "label_path": str(lbl_path),
                "bytes": total,
            }
        except Exception as e:
            last_err = e
            time.sleep(0.5)

    # eșec după retries
    return None


def _write_dataset_yaml(out_root: Path, class_names: List[str]) -> Path:
    cfg = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    yml = out_root / "data.yaml"
    yml.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yml


def build_dataset(
    out_root: Path,
    classes: List[Tuple[int, str, int]],
    per_class: int,
    val_per_class: int,
    workers: int,
    seed: int,
    photo_license: Optional[str],
    quality_grade: Optional[str],
    verifiable: bool,
) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_csv = out_root / "manifest.csv"

    downloaded_photo_ids, counts = _load_manifest(manifest_csv)
    seen_photo_ids: Set[int] = set(downloaded_photo_ids)

    rng = random.Random(seed)

    for (taxon_id, class_name, class_id) in classes:
        want_val = val_per_class
        want_train = per_class - val_per_class

        have_train = counts.get((class_id, "train"), 0)
        have_val = counts.get((class_id, "val"), 0)

        print(f"\n== {class_name} (taxon_id={taxon_id}, class_id={class_id}) ==")
        print(f"Target: train={want_train} val={want_val}")
        print(f"Have:   train={have_train} val={have_val}")

        page = 1
        per_page = 200

        with tqdm(total=(want_train + want_val), initial=(have_train + have_val), desc=f"{class_name}") as pbar:
            while have_train < want_train or have_val < want_val:
                data = _fetch_observations_page(
                    taxon_id=taxon_id,
                    page=page,
                    per_page=per_page,
                    photo_license=photo_license,
                    quality_grade=quality_grade,
                    verifiable=verifiable,
                )
                results = data.get("results", [])
                total = int(data.get("total_results") or 0)
                if not results:
                    # Am ajuns la capătul paginării (sau o pagină goală). Dacă încă nu avem destule, raportăm clar.
                    if total and (page - 1) * per_page >= total:
                        raise RuntimeError(
                            f"Insuficiente observații iNaturalist pentru {class_name}. "
                            f"total_obs={total}, downloaded_photos=train={have_train}, val={have_val}."
                        )
                    page += 1
                    time.sleep(0.5)
                    continue

                batch: List[PhotoItem] = []
                sched_train = 0
                sched_val = 0

                for obs in results:
                    obs_id = obs.get("id")
                    for ph in obs.get("photos", []) or []:
                        pid = ph.get("id")
                        if pid is None:
                            continue
                        pid = int(pid)
                        if pid in seen_photo_ids:
                            continue

                        url = _photo_download_url(ph)
                        if not url:
                            continue

                        # Nu programăm peste țintă (în batch)
                        need_train = want_train - have_train - sched_train
                        need_val = want_val - have_val - sched_val
                        if need_train <= 0 and need_val <= 0:
                            break

                        split = _choose_split(need_train, need_val, rng)
                        if split == 'train':
                            sched_train += 1
                        else:
                            sched_val += 1

                        batch.append(
                            PhotoItem(
                                class_id=class_id,
                                class_name=class_name,
                                split=split,
                                photo_id=pid,
                                obs_id=int(obs_id) if obs_id is not None else -1,
                                url=url,
                            )
                        )

                        seen_photo_ids.add(pid)

                        # Limitează batch-ul ca să nu facem 1000 futures deodată
                        if len(batch) >= 200:
                            break
                    if len(batch) >= 200:
                        break
                    if (want_train - have_train - sched_train) <= 0 and (want_val - have_val - sched_val) <= 0:
                        break

                if not batch:
                    page += 1
                    time.sleep(0.2)
                    continue

                rows: List[dict] = []
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(_download_one, it, out_root) for it in batch]
                    for fut in as_completed(futs):
                        row = fut.result()
                        if not row:
                            continue
                        rows.append(row)

                        downloaded_photo_ids.add(int(row["photo_id"]))

                        if row["split"] == "train" and have_train < want_train:
                            have_train += 1
                            pbar.update(1)
                        elif row["split"] == "val" and have_val < want_val:
                            have_val += 1
                            pbar.update(1)

                if rows:
                    _append_manifest_rows(manifest_csv, rows)

                page += 1
                time.sleep(0.2)

        counts[(class_id, "train")] = have_train
        counts[(class_id, "val")] = have_val

    class_names = [name for _, name, _ in sorted(classes, key=lambda x: x[2])]
    data_yaml = _write_dataset_yaml(out_root, class_names)

    print(f"\n✅ Dataset YOLO creat: {data_yaml}")
    return data_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="Download iNaturalist -> YOLO dataset (bbox full-frame)")
    ap.add_argument("--out", type=str, default=r"inat_datasets\\inat_basil_mint_20k", help="Folder output dataset")
    ap.add_argument("--per-class", type=int, default=10000, help="Număr imagini per clasă (train+val)")
    ap.add_argument("--val-per-class", type=int, default=1000, help="Număr imagini per clasă în val")
    ap.add_argument("--workers", type=int, default=16, help="Thread-uri pentru download")
    ap.add_argument("--seed", type=int, default=42, help="Seed pentru split")

    ap.add_argument("--basil-taxon", type=int, default=61398, help="taxon_id iNat pentru Ocimum basilicum")
    ap.add_argument("--mint-taxon", type=int, default=62652, help="taxon_id iNat pentru Mentha spicata")

    ap.add_argument(
        "--photo-license",
        type=str,
        default=None,
        help="Filtru licență (ex: cc0,cc-by,cc-by-sa). Pentru ANY lăsați gol.",
    )
    ap.add_argument(
        "--quality-grade",
        type=str,
        default="any",
        choices=["any", "research", "needs_id", "casual"],
        help="Calitatea observației iNat (default: any).",
    )
    ap.add_argument(
        "--verifiable",
        action="store_true",
        help="Dacă e setat, folosește doar observații verifiable (poate reduce mult dataset-ul).",
    )

    args = ap.parse_args()

    if args.val_per_class >= args.per_class:
        raise SystemExit("--val-per-class trebuie să fie mai mic decât --per-class")

    out_root = Path(args.out)

    classes = [
        (args.basil_taxon, "basil", 0),
        (args.mint_taxon, "mint", 1),
    ]

    qg = None if args.quality_grade == 'any' else args.quality_grade

    build_dataset(
        out_root=out_root,
        classes=classes,
        per_class=args.per_class,
        val_per_class=args.val_per_class,
        workers=args.workers,
        seed=args.seed,
        photo_license=args.photo_license,
        quality_grade=qg,
        verifiable=args.verifiable,
    )


if __name__ == "__main__":
    main()
