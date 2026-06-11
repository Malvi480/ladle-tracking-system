# 🏭 Automated Hot Metal & Steel Ladle Tracking System

> **Smart India Hackathon (SIH) 2025 — Winner**  
> Real-time ladle identification and movement tracking for high-temperature steel plant environments — no RFID, no IoT hardware.

---

## What This Does

Steel plants operate hundreds of ladles (large crucibles that carry molten metal at ~1600 °C). Knowing *which* ladle is *where* at any given moment directly affects safety, scheduling, and process efficiency. Traditional solutions rely on RFID tags or manual logbooks — both fail in high-temperature, electromagnetically noisy environments.

This system identifies ladles by reading **colour-coded alumina rings** mounted on each ladle's exterior. A camera reads the colour sequence top-to-bottom, maps it to a digit string, and produces the ladle's unique ID — then logs it to a central MySQL database in real time.

```
3 Cameras → YOLOv8-OBB Detection → HSV Strip Decoding → MySQL live_status table
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Industrial Environment                    │
│                                                                  │
│   [Camera 1]    [Camera 2]    [Camera 3]                        │
│       │              │              │                            │
└───────┼──────────────┼──────────────┼────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      detection.py                                │
│                                                                  │
│  ┌─────────────┐    ┌─────────────────────┐    ┌─────────────┐  │
│  │  YOLOv8-OBB │───▶│  HSV Strip Decoder  │───▶│  DB Writer  │  │
│  │  (best.pt)  │    │ (color_identific...) │    │ (database.py)│  │
│  │             │    │                     │    │             │  │
│  │ Detects     │    │ Green→1, Blue→2,    │    │ Upserts to  │  │
│  │ ladle bbox  │    │ Brown→3             │    │ live_status │  │
│  └─────────────┘    │ Top-to-bottom read  │    └─────────────┘  │
│                     │ e.g. [Green,Blue]   │                     │
│                     │       → Ladle #12   │                     │
│                     └─────────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  MySQL: live_status   │
                    │  (Aiven Cloud)        │
                    │                       │
                    │  ladle_id   INT PK    │
                    │  board_no   INT       │
                    │  camera_no  INT       │
                    │  timestamp  DATETIME  │
                    └───────────────────────┘
```

**Why Oriented Bounding Boxes (OBB)?**  
Ladles are frequently tilted during pouring and transport. Standard axis-aligned boxes bleed into adjacent ladles or refractory material, polluting the HSV colour histogram. OBB tightly wraps the rotated object, isolating exactly the ladle surface where the alumina rings are mounted.

---

## Key Design Decisions

| Decision | Reasoning |
|---|---|
| Colour rings instead of RFID | RFID tags fail above ~85 °C; ladle exteriors reach 200–400 °C |
| YOLOv8-OBB over standard YOLO | Handles rotated ladles; tighter crop reduces HSV noise |
| All ladles detected per frame | Frames with multiple ladles in view log all IDs simultaneously — no silent drops |
| `ON DUPLICATE KEY UPDATE` in DB | Table always holds the *current* position of each ladle; no stale accumulation |
| 5-second DB write cadence | Balances freshness vs. connection overhead on a cloud MySQL instance |
| HSV over RGB for colour detection | HSV separates hue from illumination — critical in the variable-brightness conditions of a furnace bay |
| Elliptical morphological kernel | Matches the circular cross-section of alumina rings better than a square kernel for open/close ops |
| Contour overlay on live preview | Magenta outlines show exactly which pixels triggered each strip read — essential for HSV tuning in the field |
| Optional CSV export (`SAVE_CSV`) | Enables post-session analysis of detection patterns without relying on DB connectivity |

---

## Dataset

| Split | Images |
|---|---|
| Train | 363 |
| Validation | 34 |
| Test | 18 |
| **Total** | **415** |

- **Format:** YOLOv8 Oriented Object Detection (8-point polygon coordinates)
- **Class:** `Ladle` (single class)
- **Pre-processing:** Auto-orientation, resize to 512×512
- **Augmentation:** ±15° rotation, 50% horizontal flip, Gaussian blur (σ ≤ 1.4 px)
- **Source:** [Roboflow Universe — ladle_detection_test](https://universe.roboflow.com/sih-testladle-detection/ladle_detection_test) (CC BY 4.0)

---

## Colour Encoding Scheme

Each ladle carries 1–3 alumina rings in a vertical column. Reading top-to-bottom:

| Ring Colour | Digit |
|---|---|
| 🟢 Green | 1 |
| 🔵 Blue | 2 |
| 🟤 Brown | 3 |

Example: rings `[Green, Blue]` → ladle **#12**. This scheme supports IDs from 1–333 with up to 3 rings.

HSV ranges were tuned empirically under furnace bay lighting conditions and can be adjusted in `configs/config.yaml` without touching source code.

---

## Project Structure

```
## Project Structure

```text
ladle-tracking-system/
├── detection.py             # Main detection and tracking pipeline
├── color_identification.py  # HSV colour decoding and ladle ID generation
├── database.py              # MySQL connection and live_status updates
├── config.yaml              # Camera settings, HSV thresholds and runtime parameters
├── model_evaluation.ipynb   # Model evaluation and visualization notebook
├── best.pt                  # Trained YOLOv8-OBB weights
├── requirements.txt         # Python dependencies
├── .env.example             # Database credential template
├── .gitignore
└── README.md
```

```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/ladle-tracking-system.git
cd ladle-tracking-system
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env with your MySQL host, user, password, and database name
```

### 3. Configure cameras and thresholds

Edit `configs/config.yaml`. The most common changes:

```yaml
cameras:
  - index: 1   # ← change to your actual OpenCV camera index
    name: "Camera 1"
    number: 1
```

### 4. Run

```bash
python src/detection.py
```

Press `q` to quit the preview window. The live feed shows:
- Green bounding boxes around every detected ladle
- Ladle ID label with confidence score
- Magenta contour outlines showing which pixels triggered each colour-strip read
- Live FPS counter (top-left of each feed)

To enable CSV logging of every detection, open `src/detection.py` and set `SAVE_CSV = True` at the top of the file. Logs are saved to `detections_log.csv` in the project root.

### 5. (Optional) Run the evaluation notebook

```bash
# First, unzip the dataset
unzip Ladle_detection_test_v1i_yolov8-obb.zip -d dataset/

# Then open the notebook
jupyter notebook notebooks/model_evaluation.ipynb
```

---

## Database Schema

```sql
CREATE TABLE live_status (
    ladle_id            INT      NOT NULL,
    board_number        INT      NOT NULL,
    camera_number       INT      NOT NULL,
    detection_timestamp DATETIME NOT NULL,
    PRIMARY KEY (ladle_id)
);
```

`board_number` identifies which machine (physical board/PC) made the detection — enabling multi-node deployments across different plant zones.

---

## Environment Variables

| Variable | Description |
|---|---|
| `DB_HOST` | MySQL host |
| `DB_PORT` | MySQL port (default 3306) |
| `DB_USER` | Database username |
| `DB_PASSWORD` | Database password |
| `DB_NAME` | Database name |
| `DB_SSL_CA` | *(Optional)* Path to SSL CA cert |

---

## Tech Stack

- **Detection:** [Ultralytics YOLOv8-OBB](https://docs.ultralytics.com/tasks/obb/)
- **Computer Vision:** OpenCV 4.x, NumPy
- **Database:** MySQL (Aiven Cloud) via PyMySQL
- **Dataset Annotation:** [Roboflow](https://roboflow.com)
- **Config:** PyYAML
- **Credentials:** python-dotenv

---

## Acknowledgements

Built for **Smart India Hackathon (SIH) 2025** — a national-level 36-hour hackathon.  
Team members: Mushie (Gati Shakti Vishwavidyalaya), and teammates.

---

## License

MIT
