## Source Code Details

This folder contains the core Python scripts for the Face Recognition Database Builder project. The code is structured for **maintainability**, **clarity**, and **robustness**, featuring clear function separation, type hints, and detailed docstrings.

---

### `avengers_cast_images.py`

This script handles the concurrent extraction of image URLs and face bounding boxes from a large CSV dataset and manages the network-resilient download process.

| Section | Description | Key Features |
| :--- | :--- | :--- |
| **Utilities: HTTP** | Provides the network layer. | Uses `requests.Session` with a `HTTPAdapter` and `Retry` strategy to handle transient network errors (e.g., 429, 503). |
| **CSV Loading** | Parses and filters the input data. | Uses `csv.DictReader` and can optionally filter the entire dataset to a small, curated list (e.g., `MCU_ACTORS`) via the `--filter-mcu` CLI flag. |
| **Orchestration** | Manages the download flow. | Iterates through actors sequentially but uses a **`concurrent.futures.ThreadPoolExecutor`** *per actor* to download all their images in parallel, which stabilizes resource use. |
| **I/O** | Saves the outputs. | Saves each actor image as `<idx>.jpg` and its face bounding box metadata as a separate, corresponding `<idx>.txt` file. |

---

### `create_db.py`

This script processes the local image dataset (`downloaded_faces/`), computes the high-dimensional face embeddings, and constructs the final, searchable database.

| Component | Description | Details |
| :--- | :--- | :--- |
| **Model** | Face Embedding Model. | Uses the **`InceptionResnetV1`** model pre-trained on the `vggface2` dataset via `facenet-pytorch`. Automatically selects `cuda` if available. |
| **Face Processing** | Image to Embedding Pipeline. | **Crops** the face from the image using the bounding box (`.txt` file). **Resizes** the crop to $160 \times 160$. **Normalizes** the pixel values. Finally, computes and $\ell_2$-**normalizes** the 512-dimensional embedding. |
| **Data Logic** | Handling Face Crops. | If an actor's folder name ends with `+`, the entire image is assumed to be a pre-cropped face. Otherwise, it uses the `.txt` file for bounding box coordinates. **Small faces** (less than $50 \times 50$) are skipped. |
| **Database** | Searchable Index. | Uses **FAISS** to build an **`IndexFlatIP`**, which allows for extremely fast nearest-neighbor lookups using the Inner Product (which is equivalent to cosine similarity on $\ell_2$-normalized vectors). |

---
