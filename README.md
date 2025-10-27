# Face Recognition Database Builder 🧑‍💻

This project is a two-stage pipeline designed to build a fast, searchable face recognition database (**FAISS index**) using images and bounding boxes extracted from the **IMDb-Face dataset** (source: [IMDb-Face](https://github.com/fwang91/IMDb-Face)).

The pipeline consists of two highly-structured Python scripts:  
**`avengers_cast_images.py`** (Data Extraction) and **`create_db.py`** (FAISS Index Creation).

---

## 🚀 Quick Start

### 1. Prerequisites

You'll need **Python 3.8+** and the following data file:

- **IMDb-Face CSV:** Download the primary data file and place it in a `data/` folder in the project root.

### 2. Installation

Clone the repository and install the required dependencies:

```bash
# Clone the repository (if not already done)
git clone <your-repo-link>
cd <your-repo-name>

# Install dependencies
pip install -r requirements.txt
# Note: Ensure you install torch and faiss according to your system's configuration.
```

---

## 🧠 Key Technologies Used

- **Dataset:** IMDb-Face  
- **Face Detection:** Bounding boxes sourced from the CSV (pre-processed using a detector like YOLOv8-Face — see [YapaLab/yolo-face](https://github.com/YapaLab/yolo-face)).  
- **Face Embedding:** `facenet-pytorch` (InceptionResnetV1 / VGG-Face2).  
- **Search Index:** FAISS for high-performance similarity search.  
- **Utilities:** `requests`, `opencv-python` (`cv2`).

---

## ⚙️ Usage

### Stage 1: Data Extraction and Download (`avengers_cast_images.py`)

This script reads the CSV, filters actors, and downloads corresponding images and bounding boxes.

| Argument | Description | Usage Note |
| :--- | :--- | :--- |
| `--csv` | Path to the input IMDb-Face CSV file. | Default: `data/IMDb-Face.csv` |
| `--out` | Output base directory for actor folders. | Default: `downloaded_faces` |
| `--max-workers` | Max worker threads used for concurrent downloads per actor. | Default: 8 |
| `--filter-mcu` | Enables filtering to a curated list of MCU actors defined in the script. | Omit this flag to process **all actors** found in the CSV. |

#### Example: Filtered Download (Marvel/Avengers Cast)

The script contains an example list (`MCU_ACTORS`) to demonstrate filtering.  
This limits the dataset scope due to its massive size.

```bash
python avengers_cast_images.py --filter-mcu
```

#### Example: Full Dataset Processing (or Custom List)

To process **all actors**, omit the `--filter-mcu` flag.  
You can also edit the `MCU_ACTORS` list in the script to include your own custom list.

```bash
python avengers_cast_images.py
```

Output structure example:

```
downloaded_faces/
 ├── Robert_Downey_Jr/
 │    ├── 0001.jpg
 │    ├── 0001.txt
 │    ├── 0002.jpg
 │    └── 0002.txt
 ├── Chris_Evans/
 │    ├── ...
```

---

### Stage 2: Database Creation (`create_db.py`)

This script scans the downloaded images, computes 512-dimensional face embeddings, and constructs the final searchable FAISS index.

```bash
# Build the FAISS index and save actor names
python create_db.py
```

Output files:
- `celebrity_index.faiss` → Searchable FAISS index (`IndexFlatIP`)
- `celebrity_names.pkl` → Pickled list of actor names (aligned with embeddings)

---

## ⚠️ Notes on the Dataset and Scope

- **Filtering:** The default Marvel/Avengers cast filter is optional.  
  Customize the `MCU_ACTORS` list or omit `--filter-mcu` to process any celebrity group.
- **Data Limitations:** Some actors (e.g., Don Cheadle, Tom Holland) may be underrepresented or labeled *unknown* due to limited face crops.
- **Metadata Reliance:** Relies on pre-calculated bounding box info (`rect` data) derived from YOLOv8-Face during IMDb-Face preprocessing.

---

## 🌟 Future Upgrades

This was a fun hobby project to demonstrate a robust face recognition backend.  
Planned upgrades include:

- Implementing live face detection (e.g., YOLOv8-Face) before querying FAISS.  
- Integrating real-time camera/video input.  
- Expanding the dataset for better recognition accuracy.

---

# README-SRC.md

## Source Code Details

This folder contains the core Python scripts for the **Face Recognition Database Builder** project.  
The code is structured for **maintainability**, **clarity**, and **robustness**, featuring clear function separation, type hints, and detailed docstrings.

---

### `avengers_cast_images.py`

Handles concurrent extraction of image URLs and bounding boxes from the dataset and manages a network-resilient download process.

| Section | Description | Key Features |
| :--- | :--- | :--- |
| **Utilities: HTTP** | Provides the network layer. | Uses `requests.Session` with `HTTPAdapter` and **`Retry`** strategy to handle transient errors (e.g., 429, 503). |
| **CSV Loading** | Parses and filters input data. | Uses `csv.DictReader`; can filter via the `--filter-mcu` CLI flag. |
| **Orchestration** | Manages download flow. | Iterates through actors sequentially but uses **`ThreadPoolExecutor`** per actor for parallel downloads. |
| **I/O** | Saves outputs. | Saves each actor image as `<idx>.jpg` and bounding box as `<idx>.txt`. |

---

### `create_db.py`

Processes local image data (`downloaded_faces/`), computes embeddings, and constructs the searchable FAISS database.

| Component | Description | Details |
| :--- | :--- | :--- |
| **Model** | Face Embedding Model | Uses `InceptionResnetV1` from `facenet-pytorch`, pre-trained on `vggface2`. Auto-selects CUDA if available. |
| **Face Processing** | Image → Embedding | Crops using `.txt` bounding box, resizes to 160×160, normalizes, and computes $\ell_2$-normalized 512D embedding. |
| **Data Logic** | Handling Crops | If folder name ends with `+`, treats image as full face crop. Skips faces smaller than 50×50 px. |
| **Database** | Searchable Index | Builds a FAISS `IndexFlatIP` (inner product ≈ cosine similarity on normalized vectors). |

---
