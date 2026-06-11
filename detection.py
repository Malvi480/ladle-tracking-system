"""
detection.py
------------
Real-time ladle detection and tracking across 3 camera feeds.

Pipeline (per frame, per camera):
  1. YOLOv8-OBB detects ALL ladle bounding boxes in the frame.
  2. Each ladle crop is passed to color_identification.detect_strips()
     which finds colour-coded alumina rings via HSV masking.
  3. Strips are read top-to-bottom; colour→digit mapping produces the
     ladle ID number.
  4. Detected colour contours are overlaid on the frame for visual
     confirmation of which pixels triggered each strip read.
  5. Every 5 s (configurable) detected IDs are upserted into MySQL via
     database.upsert_live_status().
  6. Optionally logs every detection to a CSV file (set SAVE_CSV=True).

Usage:
    python src/detection.py

Requirements:
    - .env file with DB credentials (see .env.example)
    - best.pt in the project root (or update model.path in config.yaml)
    - Three USB cameras accessible at the indices set in config.yaml
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from color_identification import detect_strips, strips_to_ladle_id
from database import test_connection, upsert_live_status

# ── Load config ───────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"

with open(_CONFIG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

BOARD_ID        = _CFG["system"]["board_id"]
CAMERAS         = _CFG["cameras"]           # list of {index, name, number}
MODEL_PATH      = Path(__file__).parent.parent / _CFG["model"]["path"]
TARGET_CLASS    = _CFG["model"]["target_class"]
CONF_THRESHOLD  = _CFG["model"]["conf_threshold"]
LOG_INTERVAL    = _CFG["logging"]["db_write_interval_sec"]
DISPLAY_H       = _CFG["logging"]["display_height_px"]

# ── Optional CSV logging ──────────────────────────────────────────────────────
SAVE_CSV  = False                   # set True to enable CSV export
CSV_PATH  = Path(__file__).parent.parent / "detections_log.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp_box(x1, y1, x2, y2, w_img, h_img):
    """Clamp bounding box coordinates to image dimensions."""
    x1 = max(0, min(x1, w_img - 1))
    x2 = max(0, min(x2, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(0, min(y2, h_img - 1))
    return x1, y1, x2, y2


def _process_frame(
    model:      YOLO,
    frame:      np.ndarray,
    cam_label:  str,
    csv_writer: csv.writer | None = None,
    frame_no:   int = 0,
) -> list[int]:
    """
    Run YOLO + colour detection on one frame.

    Processes ALL detected ladles in the frame (not just the first).
    Draws bounding boxes, ladle ID labels, and colour-contour overlays
    directly onto `frame` in-place.

    Args:
        model:      Loaded YOLO model.
        frame:      BGR frame to process and annotate.
        cam_label:  Short label for this camera (e.g. "C1") used in annotations.
        csv_writer: Optional csv.writer for detection logging.
        frame_no:   Current frame counter (used in CSV rows).

    Returns:
        List of decoded ladle IDs detected this frame (may be empty).
    """
    detected: list[int] = []

    try:
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    except Exception as e:
        print(f"YOLO predict error ({cam_label}): {e}")
        return detected

    h_img, w_img = frame.shape[:2]

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            cls_idx    = int(box.cls[0])
            class_name = r.names.get(cls_idx, "")
            if class_name != TARGET_CLASS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, w_img, h_img)
            if x2 <= x1 or y2 <= y1:
                continue

            conf_score = float(box.conf[0])
            ladle_roi  = frame[y1:y2, x1:x2]
            strips     = detect_strips(ladle_roi, x1, y1)
            ladle_id, digits = strips_to_ladle_id(strips)

            # ── Draw bounding box ────────────────────────────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if ladle_id is not None:
                detected.append(ladle_id)
                label = f"{cam_label} #{ladle_id} ({conf_score:.2f})"
            else:
                label = f"{cam_label} ? ({conf_score:.2f})"

            cv2.putText(
                frame, label,
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            # ── Colour-contour overlay ───────────────────────────────────────
            # Draw the exact HSV-masked pixels that produced each strip read.
            # Each strip tuple: (y, color, rect, pct, area, contours_global)
            for strip in strips:
                _, color_name, _, _, _, contours_global = strip
                for cnt in contours_global:
                    cv2.drawContours(frame, [cnt], -1, (255, 0, 255), 2)

            # ── Strip labels to the right of the box ────────────────────────
            sx, sy = x2 + 8, y1 + 16
            for strip in strips:
                _, color_name, _, color_pct, _, _ = strip
                cv2.putText(
                    frame,
                    f"{color_name} {color_pct:.0f}%",
                    (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
                sy += 20

            # ── Optional CSV row ─────────────────────────────────────────────
            if csv_writer and ladle_id is not None:
                csv_writer.writerow([
                    datetime.now().isoformat(),
                    frame_no,
                    cam_label,
                    ladle_id,
                    f"{conf_score:.3f}",
                    "|".join(digits),
                ])

    return detected


def _resize_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * target_h / h), target_h))


def _build_display(frames: list[np.ndarray], target_h: int) -> np.ndarray:
    """
    Compose 3 frames into a 2-row display:
        Row 1: Camera 1 | Camera 2
        Row 2: Camera 3 (centred)
    """
    resized = [_resize_to_height(f, target_h) for f in frames]
    row1    = np.hstack(resized[:2])
    row1_w  = row1.shape[1]
    f3      = resized[2]
    f3_w    = f3.shape[1]

    if f3_w < row1_w:
        pad   = row1_w - f3_w
        f3 = cv2.copyMakeBorder(
            f3, 0, 0, pad // 2, pad - pad // 2,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
    else:
        f3 = cv2.resize(f3, (row1_w, target_h))

    return np.vstack([row1, f3])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Database check
    if not test_connection():
        print("\n❌  Cannot proceed without database connection. Exiting.")
        return

    # 2. Load model
    print(f"\nLoading YOLO model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # 3. Open cameras
    caps = []
    for cam in CAMERAS:
        cap = cv2.VideoCapture(cam["index"])
        if not cap.isOpened():
            print(f"❌  Cannot open {cam['name']} (index {cam['index']})")
            for c in caps:
                c.release()
            return
        caps.append(cap)
        print(f"✅  {cam['name']} opened (index {cam['index']})")

    print(f"\nBoard ID  : {BOARD_ID}")
    print(f"Cameras   : {[c['name'] for c in CAMERAS]}")
    print(f"DB write  : every {LOG_INTERVAL}s")
    print(f"CSV log   : {'enabled → ' + str(CSV_PATH) if SAVE_CSV else 'disabled'}")
    print("Press 'q' to quit.\n")

    last_log_time = 0.0
    frame_count   = 0
    t_start       = time.time()

    # ── Optional CSV setup ────────────────────────────────────────────────────
    csv_file   = None
    csv_writer = None
    if SAVE_CSV:
        csv_file   = open(CSV_PATH, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "frame", "camera", "ladle_id", "conf", "digits"])

    try:
        while True:
            # 4. Capture frames
            frames = []
            skip   = False
            for cap, cam in zip(caps, CAMERAS):
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"Warning: failed to read {cam['name']}")
                    skip = True
                    break
                frames.append(frame)

            if skip:
                time.sleep(0.1)
                continue

            frame_count += 1

            # 5. Detect ladles per camera (all ladles in frame, not just first)
            cam_results: list[list[int]] = []
            for frame, cam in zip(frames, CAMERAS):
                ids = _process_frame(
                    model, frame, f"C{cam['number']}",
                    csv_writer=csv_writer, frame_no=frame_count,
                )
                cam_results.append(ids)

            # 6. FPS overlay on each frame
            elapsed = max(time.time() - t_start, 1e-6)
            fps_val = frame_count / elapsed
            for frame in frames:
                cv2.putText(
                    frame, f"FPS: {fps_val:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )

            # 7. DB log every LOG_INTERVAL seconds
            now = time.time()
            if now - last_log_time >= LOG_INTERVAL:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{ts}] Detected Ladles (Board = {BOARD_ID}):")

                for cam, ids in zip(CAMERAS, cam_results):
                    unique_ids = sorted(set(ids))
                    if unique_ids:
                        for lid in unique_ids:
                            print(f"  Board {BOARD_ID} | Camera {cam['number']} | Ladle {lid}")
                            upsert_live_status(BOARD_ID, cam["number"], lid)
                    else:
                        print(f"  Board {BOARD_ID} | Camera {cam['number']} | Ladle: None")

                print("-" * 60)
                last_log_time = now

            # 8. Display
            try:
                combined = _build_display(frames, DISPLAY_H)
                cv2.imshow("Ladle Tracking — Triple Camera", combined)
            except Exception as e:
                print(f"Display warning: {e}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # 9. Cleanup — always runs even on exception
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()
            print(f"CSV saved → {CSV_PATH}")
        print("Stopped.")


if __name__ == "__main__":
    main()
