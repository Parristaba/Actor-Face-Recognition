"""
create_db.py

Purpose
-------
Scan a directory of per-actor folders containing images (+ optional rectangle metadata),
compute face embeddings using facenet-pytorch's InceptionResnetV1 (vggface2),
and persist a FAISS index plus a pickle of names mapping each embedding to an actor.

Behavior (unchanged)
--------------------
- Looks for actor subfolders in DOWNLOAD_DIR.
- If an actor folder name ends with '+' the entire image is treated as a pre-cropped face.
- Otherwise, expects accompanying .txt files per image containing newline-separated rects:
    x1 y1 x2 y2
  Each rect yields a cropped face embedding (skips rects smaller than MIN_FACE_SIZE).
- Normalizes embeddings and writes a FAISS IndexFlatIP index and a pickle list of names.
- Uses the same constants and file names as before.

This refactor reorganizes the code into clear sections and functions with docstrings and
logging while preserving the original algorithm and outputs.
"""

# -------------------------
# Standard / third-party imports
# -------------------------
import os
import time
import pickle
import logging
from typing import List, Sequence, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

import torch
from facenet_pytorch import InceptionResnetV1
import faiss

# -------------------------
# Configuration / Constants (identical defaults to original)
# -------------------------
DOWNLOAD_DIR = 'downloaded_faces'  # folders per actor
SAVE_INDEX = 'celebrity_index.faiss'
SAVE_NAMES = 'celebrity_names.pkl'
MODEL_SIZE = 160
MIN_FACE_SIZE = 50

# -------------------------
# Logging / device / model setup
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)  # kept for parity with original script

# Load model (same pretrained checkpoint as original)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# -------------------------
# Helper functions
# -------------------------
def preprocess_face(face_bgr: np.ndarray) -> Optional[torch.Tensor]:
    """
    Resize, convert BGR->RGB, normalize and return a torch tensor suitable for the model.
    Returns None if input is invalid or resizing fails.
    """
    if face_bgr is None or face_bgr.size == 0:
        return None
    try:
        face_resized = cv2.resize(face_bgr, (MODEL_SIZE, MODEL_SIZE))
    except Exception:
        return None
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
    t = (t - 0.5) / 0.5
    return t.unsqueeze(0).to(device)


def read_rect(rect_file: str) -> List[Tuple[int, int, int, int]]:
    """
    Read a rect file and return a list of rect tuples (x1, y1, x2, y2).
    If reading/parsing fails, returns an empty list.
    """
    try:
        with open(rect_file, "r") as f:
            rect_str = f.read().strip()
        rects = [tuple(map(int, line.strip().split()))
                 for line in rect_str.split("\n") if line.strip()]
        # keep only rects of length 4
        rects = [r for r in rects if len(r) == 4]
        return rects
    except Exception:
        return []


def embed_face(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Preprocess the BGR face image and compute a normalized 1-D embedding (float32).
    Returns None on failure.
    """
    inp = preprocess_face(face_bgr)
    if inp is None:
        return None
    with torch.no_grad():
        emb = model(inp)[0].cpu().numpy()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype("float32")


# -------------------------
# Main processing
# -------------------------
def process_dataset(download_dir: str = DOWNLOAD_DIR,
                    min_face_size: int = MIN_FACE_SIZE) -> Tuple[List[np.ndarray], List[str]]:
    """
    Walk through actor folders, embed faces, and return parallel lists:
      - embeddings (float32 numpy arrays)
      - names (actor name per embedding)

    Behavior matches the original script:
      - actor folder with trailing '+' => treat images as precropped faces
      - otherwise, read associated .txt rect files per image and crop each rect
      - skip rects smaller than MIN_FACE_SIZE
    """
    actor_folders = [f for f in os.listdir(download_dir) if os.path.isdir(os.path.join(download_dir, f))]
    all_embeddings: List[np.ndarray] = []
    all_names: List[str] = []

    start_time = time.time()
    for actor in tqdm(actor_folders, desc="Processing actors"):
        folder_path = os.path.join(download_dir, actor)
        is_precropped = actor.endswith("+")

        for file_name in os.listdir(folder_path):
            # only process image files
            if not file_name.lower().endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            if is_precropped:
                # Directly embed the whole image (actor folder name ends with '+')
                emb = embed_face(img)
                if emb is not None:
                    all_embeddings.append(emb)
                    all_names.append(actor.rstrip("+"))  # remove '+' to keep names consistent
            else:
                # Use rect files (same basename, .txt extension)
                rect_path = os.path.splitext(img_path)[0] + ".txt"
                if not os.path.exists(rect_path):
                    continue

                rects = read_rect(rect_path)
                h, w = img.shape[:2]
                for rect in rects:
                    if len(rect) != 4:
                        continue
                    x1, y1, x2, y2 = rect
                    # clamp to image bounds (same behavior as original)
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)

                    if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
                        continue

                    face_crop = img[y1:y2, x1:x2]
                    emb = embed_face(face_crop)
                    if emb is not None:
                        all_embeddings.append(emb)
                        all_names.append(actor)

    elapsed = time.time() - start_time
    logger.info("Processed dataset in %.1fs, embeddings collected: %d", elapsed, len(all_embeddings))
    return all_embeddings, all_names


def build_and_save_faiss(embeddings: Sequence[np.ndarray],
                         names: Sequence[str],
                         save_index: str = SAVE_INDEX,
                         save_names: str = SAVE_NAMES) -> None:
    """
    Build a FAISS IndexFlatIP index from the provided embeddings and save the index and names.
    Preserves original file names and behavior.
    """
    if len(embeddings) == 0:
        raise RuntimeError("No embeddings generated! Check your dataset.")

    embeddings_array = np.array(embeddings)
    dim = embeddings_array.shape[1]
    print("Embedding dimension:", dim)
    print("Total embeddings:", len(embeddings))

    index = faiss.IndexFlatIP(dim)  # cosine similarity on normalized vectors (inner product)
    index.add(embeddings_array)

    faiss.write_index(index, save_index)
    with open(save_names, "wb") as f:
        pickle.dump(list(names), f)

    print("✅ Saved FAISS index and actor names.")


def main() -> None:
    all_embeddings, all_names = process_dataset(DOWNLOAD_DIR, MIN_FACE_SIZE)
    build_and_save_faiss(all_embeddings, all_names, SAVE_INDEX, SAVE_NAMES)
    print("✅ Done.")


# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    main()