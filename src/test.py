"""
vis_improved_async_recognition.py

Purpose
-------
A self-contained script to run asynchronous face recognition on a video using:
 - YOLOv8 (ultralytics) for face detection
 - facenet-pytorch (InceptionResnetV1) for face embeddings
 - FAISS for nearest-neighbor lookups
 - OpenCV for tracking, visualization and output writing

Notes for publishing
--------------------
- This file is formatted and documented for GitHub publication but its runtime
  behavior is unchanged from the original. No functional changes were made.
- Ensure the following assets exist relative to the repository root:
    * VIDEO_PATH -> input video file
    * YOLO_MODEL_PATH -> yolov8 face weights (e.g. yolov8n-face.pt)
    * FAISS_INDEX_PATH -> saved FAISS index (.faiss)
    * FAISS_NAMES_PATH -> pickle file with actor names (.pkl)
- Requirements (approx):
    ultralytics, facenet-pytorch, faiss, torch, opencv-python, numpy

Usage
-----
python test.py
"""

import os
import cv2
import torch
import faiss
import pickle
import time
import numpy as np
import threading
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from collections import OrderedDict, defaultdict

# -------------------------
# CONFIG
# -------------------------

# To use with your own files, download necessary assets and set paths here. I removed mine for privacy.
VIDEO_PATH = '' 
YOLO_MODEL_PATH = ''
FAISS_INDEX_PATH = ''
FAISS_NAMES_PATH = ''   

# These constants control detection and recognition behavior. For me, these are the best settings, however note that each movie may require different tuning.
DETECT_EVERY_N = 5
YOLO_INPUT_SIZE = 320
MIN_FACE_SIZE = 50
K_NEIGHBORS = 3
SIM_THRESHOLD = 0.65
LOW_CONF_THRESHOLD = 0.8
MAX_CONCURRENT_RECOG_THREADS = 3


# Cuda enabled device is advised, since CPU inference will be very slow.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# -------------------------
# FACE DETECTION (YOLOv8)
# -------------------------
class FaceDetector:
    """
    Wrapper around a YOLO model instance tuned for face detection.
    detect_faces(frame) -> list of (x1, y1, x2, y2) boxes
    """
    def __init__(self, model_path=YOLO_MODEL_PATH, input_size=YOLO_INPUT_SIZE):
        self.model = YOLO(model_path)
        self.input_size = input_size

    def detect_faces(self, frame):
        results = self.model.predict(frame, imgsz=self.input_size, verbose=False)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                    continue
                boxes.append((x1, y1, x2, y2))
        return boxes

# -------------------------
# KNN IDENTIFIER
# -------------------------
class KNNFaceIdentifier:
    """
    Simple KNN-style identifier using a FAISS index and names list.
    Votes are weighted by similarity; a similarity threshold filters low-confidence matches.
    """
    def __init__(self, index, names, k=K_NEIGHBORS, sim_threshold=SIM_THRESHOLD):
        self.index = index
        self.names = names
        self.k = k
        self.sim_threshold = sim_threshold

    def identify(self, emb):
        emb = emb.astype('float32')
        D, I = self.index.search(np.array([emb]), k=self.k)
        votes = defaultdict(float)
        counts = defaultdict(int)
        for sim, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            actor = self.names[idx]
            votes[actor] += float(sim)
            counts[actor] += 1

        if not votes:
            return "Unknown", 0.0

        chosen_actor = max(votes.items(), key=lambda x: x[1])[0]
        sims_for_actor = [float(sim) for sim, idx in zip(D[0], I[0]) if idx >= 0 and self.names[idx] == chosen_actor]
        confidence = sum(sims_for_actor) / len(sims_for_actor) if sims_for_actor else 0.0
        if confidence < self.sim_threshold:
            return "Unknown", confidence
        return chosen_actor, confidence

# -------------------------
# FACE RECOGNITION
# -------------------------
class FaceRecognizer:
    """
    Loads FAISS index, name mapping, and the embedding model.
    Provides methods to preprocess crops and obtain normalized embeddings.
    """
    def __init__(self, faiss_index_path=FAISS_INDEX_PATH, faiss_names_path=FAISS_NAMES_PATH):
        self.index = faiss.read_index(faiss_index_path)
        with open(faiss_names_path, 'rb') as f:
            self.names = pickle.load(f)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
        self.knn = KNNFaceIdentifier(self.index, self.names)

    def preprocess_face(self, face_bgr):
        try:
            face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (160, 160)), cv2.COLOR_BGR2RGB)
        except Exception:
            return None
        t = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        return t.unsqueeze(0).to(DEVICE)

    def get_embedding(self, face_bgr):
        inp = self.preprocess_face(face_bgr)
        if inp is None:
            return None
        with torch.no_grad():
            emb = self.model(inp).cpu().numpy()[0].astype('float32')
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        emb /= norm
        return emb

    def recognize(self, emb):
        if emb is None:
            return "Unknown", 0.0
        return self.knn.identify(emb)

# -------------------------
# Async recognizer thread
# -------------------------
class AsyncRecognizerThread(threading.Thread):
    """
    Background thread to compute embeddings and perform identification.
    Uses a semaphore to limit concurrent recognitions and a lock to update shared caches.
    """
    def __init__(self, recognizer, face_id, face_crop, name_cache, conf_cache, cache_lock, sem):
        super().__init__(daemon=True)
        self.recognizer = recognizer
        self.face_id = face_id
        self.face_crop = face_crop
        self.name_cache = name_cache
        self.conf_cache = conf_cache
        self.cache_lock = cache_lock
        self.sem = sem

    def run(self):
        acquired = self.sem.acquire(timeout=10)
        if not acquired:
            with self.cache_lock:
                self.name_cache[self.face_id] = "Unknown"
                self.conf_cache[self.face_id] = 0.0
            return
        try:
            emb = self.recognizer.get_embedding(self.face_crop)
            name, conf = self.recognizer.recognize(emb)
            with self.cache_lock:
                self.name_cache[self.face_id] = name
                self.conf_cache[self.face_id] = conf
        finally:
            self.sem.release()

# -------------------------
# Visualization helpers
# -------------------------
def alpha_rectangle(img, xy1, xy2, color, alpha=0.35):
    """
    Draw a semi-transparent filled rectangle on img between xy1 and xy2.
    """
    overlay = img.copy()
    cv2.rectangle(overlay, xy1, xy2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_label_with_bg(img, text, org, bg_color, text_color=(255,255,255), alpha=0.9, pad=6):
    """
    Draw a text label with a filled background and subtle shadow for readability.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    x1, y1 = x - pad, y - h - pad
    x2, y2 = x + w + pad, y + baseline + pad
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    shadow_org = (x + 1, y + 1)
    cv2.putText(img, text, shadow_org, font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, text_color, thickness, cv2.LINE_AA)

def lerp_box(b0, b1, t):
    """
    Linear interpolation between two boxes (b0, b1) with factor t in [0,1].
    """
    return tuple(int(b0[i] + (b1[i] - b0[i]) * t) for i in range(4))

# -------------------------
# MAIN LOOP
# -------------------------
def main():
    """
    Main processing loop:
     - Detect faces every N frames
     - Create trackers for detected faces
     - Spawn background recognition threads per detected face (bounded by semaphore)
     - Visualize tracked faces, labels, popup cards, and write output video
    """
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: cannot open video.")
        return

    # Create output folder and writer
    os.makedirs("output", exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    output_path = os.path.join("output", f"{clip_name}_processed.mp4")

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps_video, (w_frame, h_frame))
    print(f"[INFO] Saving output to: {output_path}")

    frame_count = 0
    trackers = OrderedDict()
    face_id_counter = 0
    name_cache, conf_cache, meta = {}, {}, {}
    cache_lock = threading.Lock()
    sem = threading.BoundedSemaphore(MAX_CONCURRENT_RECOG_THREADS)
    popup_cards = []
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0.0
    interp_smoothing = 0.25

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h_frame, w_frame = frame.shape[:2]
            fps_counter += 1
            if (time.time() - fps_start_time) >= 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            # Detection step: clear existing trackers and spawn new ones for detected faces
            if frame_count % DETECT_EVERY_N == 0:
                trackers.clear()
                boxes = detector.detect_faces(frame)
                for (x1, y1, x2, y2) in boxes:
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                    face_crop = frame[y1:y2, x1:x2].copy()
                    face_id = face_id_counter
                    face_id_counter += 1
                    with cache_lock:
                        name_cache[face_id] = "Processing..."
                        conf_cache[face_id] = 0.0
                        meta[face_id] = {
                            'bbox': (x1, y1, x2, y2),
                            'prev_bbox': (x1, y1, x2, y2),
                            'alpha': 0.0,
                            'last_seen': frame_count,
                            'thumb': cv2.resize(face_crop, (80, 80)) if face_crop.size else None,
                            'label_shown': False
                        }
                    t = AsyncRecognizerThread(recognizer, face_id, face_crop, name_cache, conf_cache, cache_lock, sem)
                    t.start()
                    trackers[face_id] = (tracker, (x1, y1, x2, y2))

            # Tracking / visualization step for intermediate frames
            else:
                dead = []
                for fid, (tracker, last_box) in list(trackers.items()):
                    ok, bbox = tracker.update(frame)
                    if not ok:
                        dead.append(fid)
                        continue
                    x, y, w, h = map(int, bbox)
                    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                        continue
                    x2, y2 = x + w, y + h
                    with cache_lock:
                        name = name_cache.get(fid, "Processing...")
                        conf = conf_cache.get(fid, 0.0)
                        m = meta.get(fid, None)
                    if m is None:
                        with cache_lock:
                            meta[fid] = {'bbox': (x, y, x2, y2), 'prev_bbox': (x, y, x2, y2),
                                         'alpha': 0.0, 'last_seen': frame_count, 'thumb': None, 'label_shown': False}
                        m = meta[fid]
                    prev = m.get('bbox', (x, y, x2, y2))
                    cur = (x, y, x2, y2)
                    interp_box = lerp_box(prev, cur, interp_smoothing)
                    m['prev_bbox'] = prev
                    m['bbox'] = interp_box
                    m['last_seen'] = frame_count
                    m['alpha'] = min(1.0, m.get('alpha', 0.0) + 0.15)

                    # Choose a base color depending on recognition state / confidence
                    base_color = (0, 200, 0) if (name not in ["Unknown", "Processing..."] and conf >= LOW_CONF_THRESHOLD) else \
                                 (0, 200, 200) if conf < LOW_CONF_THRESHOLD else \
                                 (0, 0, 220) if name == "Unknown" else (160, 160, 160)

                    alpha_rectangle(frame, (interp_box[0], interp_box[1]), (interp_box[2], interp_box[3]), base_color, alpha=0.25 * m['alpha'])
                    label_text = f"{name}" if name in ["Processing...", "Unknown"] else f"{name} ({conf:.2f})"
                    draw_label_with_bg(frame, label_text, (interp_box[0] + 6, max(20, interp_box[1] - 6)),
                                       bg_color=base_color, text_color=(255,255,255), alpha=0.9)
                    cv2.rectangle(frame, (interp_box[0], interp_box[1]), (interp_box[2], interp_box[3]), base_color, 2)

                    # Append a popup card the first time a label is shown for a tracked face
                    if name not in ["Unknown", "Processing..."] and not m.get('label_shown', False):
                        popup_cards.append({'name': name, 'conf': conf, 'thumb': m.get('thumb', None), 'start_time': time.time()})
                        m['label_shown'] = True
                for fid in dead:
                    trackers.pop(fid, None)
                    with cache_lock:
                        name_cache.pop(fid, None)
                        conf_cache.pop(fid, None)
                        meta.pop(fid, None)

            # Render popup cards on the bottom-left
            now = time.time()
            popup_y = h_frame - 10
            popup_height = 84
            popup_margin = 8
            active_popups = []
            for card in reversed(popup_cards):
                age = now - card['start_time']
                if age > 3.0:
                    continue
                alpha = max(0.0, 1.0 - (age / 3.0))
                card_w = 260
                card_h = 72
                popup_x = 10
                popup_y = popup_y - card_h - popup_margin
                overlay = frame.copy()
                cv2.rectangle(overlay, (popup_x, popup_y), (popup_x + card_w, popup_y + card_h), (20,20,20), -1)
                cv2.addWeighted(overlay, 0.75 * alpha, frame, 1 - (0.75 * alpha), 0, frame)
                if card['thumb'] is not None:
                    th = card['thumb']
                    th_h, th_w = th.shape[:2]
                    tx, ty = popup_x + 6, popup_y + (card_h - th_h) // 2
                    if 0 <= tx < w_frame and 0 <= ty < h_frame and tx + th_w <= w_frame and ty + th_h <= h_frame:
                        frame[ty:ty+th_h, tx:tx+th_w] = th
                draw_label_with_bg(frame, f"{card['name']} ({card['conf']:.2f})",
                                   (popup_x + 100, popup_y + 30), bg_color=(50,50,50), text_color=(255,255,255), alpha=0.0)
                active_popups.append(card)
            popup_cards = list(reversed(active_popups))

            # Heads-up display (FPS, counts)
            total_faces = len(trackers)
            with cache_lock:
                known_faces = sum(1 for v in name_cache.values() if v not in ["Unknown", "Processing..."])
                unknown_faces = sum(1 for v in name_cache.values() if v == "Unknown")
            hud = f"FPS: {fps:.1f}  |  Faces: {total_faces}  |  Known: {known_faces}  |  Unknown: {unknown_faces}"
            cv2.rectangle(frame, (8, 8), (420, 38), (0,0,0), -1)
            cv2.putText(frame, hud, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)

            # Write processed frame to output video
            out_writer.write(frame)

            # Display current frame
            cv2.imshow("Face Recognition (vis)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    finally:
        # Clean up resources
        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()
        time.sleep(0.5)

if __name__ == "__main__":
    main()
