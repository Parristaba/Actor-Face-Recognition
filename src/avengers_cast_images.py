"""
avengers_cast_images.py

Purpose
-------
A small, well-documented utility to extract actor image URLs (from a CSV such as IMDb-Face),
download the images concurrently, and save them per-actor along with their bounding-rect
metadata. This version is structured for maintainability and clarity: high-level sections,
typed functions, retry-capable requests, progress reporting, and simple CLI options.

Usage
-----
Run from the repository root (Windows):
    python avengers_cast_images.py --csv data/IMDb-Face.csv --out downloaded_faces --max-workers 8

Major Sections
--------------
1) Configuration & constants
2) Utilities (path/name helpers, requests session with retries)
3) CSV loading and filtering
4) Download & save logic (concurrent per-actor)
5) CLI entrypoint / orchestration

Notes
-----
- Uses pathlib.Path for robust path handling on Windows.
- Uses requests.Session with HTTPAdapter + Retry for better network resilience.
- Saves each actor image as <idx>.jpg and rectangle metadata as <idx>.txt in the actor folder.
- Minimal per-line comments: sections are separated and functions documented with docstrings.
"""

# -------------------------
# Standard / Third-party imports
# -------------------------
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import csv
import re
import time
import logging
import argparse
import concurrent.futures

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# -------------------------
# Configuration / Constants
# -------------------------
DEFAULT_CSV = Path("data/IMDb-Face.csv")
DEFAULT_OUT_DIR = Path("downloaded_faces")
TIMEOUT = 10  # seconds for HTTP requests
MAX_WORKERS_DEFAULT = 8
MILESTONE = 100  # actors per milestone log
REQUEST_RETRIES = 3
REQUEST_BACKOFF_FACTOR = 0.5
USER_AGENT = "avengers-cast-downloader/1.0"

# Example curated MCU actors list for demo filtering (can be extended or replaced).
MCU_ACTORS = [
    "Robert_Downey_Jr", "Gwyneth_Paltrow", "Jeff_Bridges", "Don_Cheadle",
    "Scarlett_Johansson", "Paul_Bettany", "Chris_Hemsworth", "Natalie_Portman",
    "Tom_Hiddleston", "Idris_Elba", "Anthony_Hopkins", "Chris_Evans",
    "Mark_Ruffalo", "Benedict_Cumberbatch", "Jeremy_Renner", "Samuel_L_Jackson",
    "Paul_Rudd", "Evangeline_Lilly", "Tom_Holland", "Zendaya", "Benedict_Wong",
    # (list truncated for brevity)
]

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Utilities: path & name helpers
# -------------------------
def safe_folder_name(name: str) -> str:
    """
    Convert an actor name into a filesystem-safe folder name.
    Keeps letters, digits and underscore.
    """
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


# -------------------------
# Utilities: HTTP session with retries
# -------------------------
def build_session(retries: int = REQUEST_RETRIES, backoff: float = REQUEST_BACKOFF_FACTOR) -> requests.Session:
    """
    Build a requests.Session configured with retry backoff to increase robustness on flaky networks.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry_strategy = Retry(
        total=retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        backoff_factor=backoff,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_image(session: requests.Session, url: str, timeout: int = TIMEOUT) -> Optional[np.ndarray]:
    """
    Download an image from `url` using `session`. Returns a BGR OpenCV image or None on failure.
    """
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.debug("cv2.imdecode returned None for URL: %s", url)
        return img
    except Exception as exc:
        logger.debug("Failed to download %s => %s", url, exc)
        return None


# -------------------------
# CSV Loading & Filtering
# -------------------------
def load_imdb_face_csv(csv_path: Path) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load the IMDb-Face formatted CSV into a dict mapping actor name -> list of (rect, url).
    Expects CSV with columns at least: 'name', 'rect', 'url'.
    """
    actor_images: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("name")
            rect = row.get("rect", "")
            url = row.get("url", "")
            if not name or not url:
                continue
            actor_images[name].append((rect, url))
    return actor_images


def filter_actors(actor_images: Dict[str, List[Tuple[str, str]]], allowed_actors: Optional[Iterable[str]] = None) -> Dict[str, List[Tuple[str, str]]]:
    """
    Filter actor_images to only include actors in allowed_actors (if provided).
    If allowed_actors is None, returns same dict.
    """
    if allowed_actors is None:
        return actor_images
    allowed_set = set(allowed_actors)
    return {actor: imgs for actor, imgs in actor_images.items() if actor in allowed_set}


# -------------------------
# Disk I/O: saving images & rect
# -------------------------
def save_image_and_rect(actor_folder: Path, idx: int, img: np.ndarray, rect: str) -> None:
    """
    Save image and rect metadata to disk.
    Writes <idx>.jpg and <idx>.txt inside actor_folder.
    """
    jpg_path = actor_folder / f"{idx}.jpg"
    txt_path = actor_folder / f"{idx}.txt"
    cv2.imwrite(str(jpg_path), img)
    txt_path.write_text(rect or "")


# -------------------------
# Worker: per-image processing
# -------------------------
def process_image(session: requests.Session, actor_folder: Path, idx: int, rect: str, url: str) -> bool:
    """
    Download a single image and save it; returns True on success, False otherwise.
    Kept simple so it can be used with concurrent.futures.Executor.map.
    """
    img = download_image(session, url)
    if img is None:
        return False
    save_image_and_rect(actor_folder, idx, img, rect)
    return True


# -------------------------
# Orchestration: per-actor concurrent download
# -------------------------
def download_for_actor(session: requests.Session, actor: str, images: List[Tuple[str, str]], out_root: Path, max_workers: int) -> Tuple[str, int, int]:
    """
    Download all images for a single actor using a ThreadPoolExecutor.
    Returns (actor, downloaded_count, skipped_count)
    """
    folder_name = safe_folder_name(actor)
    actor_folder = out_root / folder_name
    ensure_dir(actor_folder)

    args = [(actor_folder, idx, rect, url) for idx, (rect, url) in enumerate(images)]

    # We wrap process_image with partial-like behavior by passing session separately in lambda
    downloaded = 0
    skipped = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        # submit tasks and iterate as completed for simple counting
        futures = [ex.submit(process_image, session, *a) for a in args]
        for fut in concurrent.futures.as_completed(futures):
            try:
                success = fut.result()
            except Exception:
                success = False
            if success:
                downloaded += 1
            else:
                skipped += 1
    return actor, downloaded, skipped


# -------------------------
# Main orchestration / CLI
# -------------------------
def main(csv_path: Path, out_dir: Path, allowed_actors: Optional[Iterable[str]], max_workers: int) -> None:
    """
    Top-level orchestration:
      - loads CSV
      - filters actors
      - iterates actors and downloads images concurrently per-actor
      - logs progress and milestones
    """
    start = time.time()
    ensure_dir(out_dir)
    session = build_session()

    # Load CSV -> actor -> list[(rect, url)]
    actor_images = load_imdb_face_csv(csv_path)
    logger.info("Total unique actors in CSV: %d", len(actor_images))

    # Optionally filter to allowed set (e.g., MCU actors)
    if allowed_actors is not None:
        actor_images = filter_actors(actor_images, allowed_actors)
        logger.info("Actors after filtering: %d", len(actor_images))

    processed_actors = 0
    total_downloaded = 0
    total_skipped = 0

    # Iterate actors sequentially but use per-actor thread pools for their images.
    # This keeps resource usage stable while enabling parallel downloads for each actor.
    actors_iter = list(actor_images.items())
    for actor, images in tqdm(actors_iter, desc="Actors"):
        processed_actors += 1
        actor_name, dcount, scount = download_for_actor(session, actor, images, out_dir, max_workers=max_workers)
        total_downloaded += dcount
        total_skipped += scount

        if processed_actors % MILESTONE == 0:
            elapsed = time.time() - start
            logger.info("[MILESTONE] %d actors processed (%d downloaded, %d skipped) in %.1fs", processed_actors, total_downloaded, total_skipped, elapsed)

    elapsed = time.time() - start
    logger.info("Done. Actors processed: %d | Images downloaded: %d | Images skipped: %d | Elapsed: %.1fs",
                processed_actors, total_downloaded, total_skipped, elapsed)


# -------------------------
# CLI argument parsing
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download actor images from IMDb-Face CSV (filterable).")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to IMDb-Face CSV file.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output base directory for downloaded actor folders.")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS_DEFAULT, help="Max worker threads per actor.")
    parser.add_argument("--filter-mcu", action="store_true", help="Only download images for curated MCU actors list.")
    return parser.parse_args()


# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    args = parse_args()
    allowed = MCU_ACTORS if args.filter_mcu else None
    try:
        main(csv_path=args.csv, out_dir=args.out, allowed_actors=allowed, max_workers=args.max_workers)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Exiting.")