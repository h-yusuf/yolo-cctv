"""ALPR MVP: YOLO vehicle tracking + fast-alpr plate OCR on a CCTV stream or video file.

Pipeline: frame -> YOLO11 + ByteTrack (vehicles) -> vehicle inside zone? -> fast-alpr on the
vehicle crop -> Indonesian plate regex -> vote per track id -> SQLite row when the track ends.

    uv run alpr.py --show                      # source + zone from .env (CCTV_URL, ZONE)
    uv run alpr.py --source video.mp4 --save out.mp4
"""

import argparse
import logging
import os
import re
import sqlite3
import threading
import time
from collections import Counter, deque
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from fast_alpr import ALPR
from ultralytics import YOLO

# CoreML EP logs an error on every crop with zero plates; the result is still an empty list.
ort.set_default_logger_severity(4)
logging.getLogger("open_image_models.detection.core.yolo_v9.inference").setLevel(logging.ERROR)

VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO ids
PLATE_RE = re.compile(r"^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$")  # e.g. K1989RC, B1234ABC, H12


def load_env(path: Path = Path(__file__).with_name(".env")):
    """Minimal .env loader (KEY=VALUE lines, # comments). Variables already set in the shell win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


AUTO_ZONE = (0.25, 0.85)  # auto zone: full-width band from 25% to 85% of the frame height


def parse_zone(spec: str | None, width: int, height: int) -> np.ndarray:
    """ZONE "auto" (or empty) -> centre band of the frame; "x,y x,y ..." -> that polygon (manual).
    ponytail: auto is a fixed band, not learned from where plates are actually readable; set a
    manual zone when the band misses the lane or covers an on-screen text overlay."""
    if not spec or spec.strip().lower() == "auto":
        top, bottom = int(height * AUTO_ZONE[0]), int(height * AUTO_ZONE[1])
        return np.array([[0, top], [width, top], [width, bottom], [0, bottom]], np.int32)
    return np.array([[int(v) for v in p.split(",")] for p in spec.split()], np.int32)


def normalize(text: str | None) -> str | None:
    """Uppercase, drop non-alphanumerics, return None unless it looks like an Indonesian plate."""
    plate = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    return plate if PLATE_RE.match(plate) else None


class Track:
    """OCR votes for one tracked vehicle."""

    def __init__(self, vehicle: str, now: float):
        self.vehicle = vehicle
        self.first_seen = self.last_seen = now
        self.last_ocr = 0.0
        self.votes: Counter[str] = Counter()
        self.score: Counter[str] = Counter()
        self.crops: dict[str, tuple[float, np.ndarray | None]] = {}  # plate -> (conf, best crop of that plate)
        self.plate_box: tuple[int, int, int, int] | None = None  # last plate box, relative to the vehicle

    def add(self, plate: str, conf: float, crop: np.ndarray | None = None):
        self.votes[plate] += 1
        self.score[plate] += conf
        if conf > self.crops.get(plate, (-1.0, None))[0]:
            self.crops[plate] = (conf, crop)

    def result(self, min_votes: int) -> tuple[str, int, float] | None:
        """Plate with the highest summed confidence, if it got at least min_votes reads."""
        if not self.score:
            return None
        plate, score = self.score.most_common(1)[0]
        n = self.votes[plate]
        return (plate, n, score / n) if n >= min_votes else None


class StreamReader:
    """Reads a live stream in a thread, keeps every `stride`-th frame in a bounded buffer and
    reconnects when the stream drops. HLS delivers a whole segment at once, so a "latest frame
    only" reader would keep ~1 frame per segment; the buffer keeps the burst, and when inference
    is slower than the stream the oldest frames fall off instead of lagging behind live."""

    def __init__(self, source: str, stride: int, maxlen: int = 50):
        self.source, self.stride, self.running = source, stride, True
        self.buf: deque[np.ndarray] = deque(maxlen=maxlen)
        self.cap = cv2.VideoCapture(source)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        n = 0
        while self.running:
            if not self.cap.grab():
                print(f"{datetime.now():%H:%M:%S} stream dropped, reconnecting", flush=True)
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.source)
                continue
            n += 1
            if n % self.stride == 0:
                ok, frame = self.cap.retrieve()
                if ok:
                    self.buf.append(frame)

    def read(self) -> np.ndarray | None:
        while self.running:
            if self.buf:
                return self.buf.popleft()
            time.sleep(0.005)
        return None

    def release(self):
        # Releasing the capture while the thread is inside grab() crashes ffmpeg; let it finish first.
        self.running = False
        self.thread.join(timeout=10)
        self.cap.release()


def is_duplicate(plate: str, now: float, last_saved: dict[str, float], window: float) -> bool:
    """Same or near-same plate saved within the window. A car waiting at the light often gets a
    new track id and a slightly different read (AD1753QR / AD1751QV / D1753QF).
    ponytail: similarity >= 0.75 can also merge two real cars like H1224QA / H1234QA inside the
    window; tighten the threshold or key on location if that matters."""
    return any(now - ts < window and SequenceMatcher(None, p, plate).ratio() >= 0.75
               for p, ts in last_saved.items())


def ocr_vehicle(alpr: ALPR, crop: np.ndarray):
    """Best Indonesian-looking plate read in a vehicle crop:
    (plate, mean char confidence, plate crop, plate box relative to the vehicle crop) or None."""
    best = None
    for r in alpr.predict(crop):
        if r.ocr is None:
            continue
        plate = normalize(r.ocr.text)
        conf = float(np.mean(r.ocr.confidence))
        if plate and (best is None or conf > best[1]):
            b = r.detection.bounding_box
            box = (max(b.x1, 0), max(b.y1, 0), b.x2, b.y2)
            best = (plate, conf, crop[box[1]:box[3], box[0]:box[2]].copy(), box)
    return best


def mask_plate(plate: str) -> str:
    """Privacy mode: hide the digits, keep region and suffix letters (H1234AB -> H****AB)."""
    return re.sub(r"\d", "*", plate)


def pixelate(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, block: int = 6):
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, img.shape[1]), min(y2, img.shape[0])
    roi = img[y1:y2, x1:x2]
    if roi.size:
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (max(1, w // block), max(1, h // block)))
        img[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def draw_corners(img, x1, y1, x2, y2, color, t, k=0.25):
    lx, ly = int((x2 - x1) * k), int((y2 - y1) * k)
    for x, y, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (x, y), (x + dx * lx, y), color, t)
        cv2.line(img, (x, y), (x, y + dy * ly), color, t)


def draw_panel(img, recent: list[tuple[str, np.ndarray | None]], s: float, privacy: bool = False):
    """Left-side gallery of the latest finalized plates (crop + text)."""
    w, row = int(300 * s), int(110 * s)
    h = min(img.shape[0], row * len(recent) + int(10 * s))
    img[:h, :w] = (img[:h, :w] * 0.3).astype(np.uint8)
    for i, (plate, crop) in enumerate(recent):
        y = i * row + int(10 * s)
        if crop is not None and crop.size:
            ch = int(60 * s)
            cw = min(w - int(20 * s), int(crop.shape[1] * ch / crop.shape[0]))
            x = int(10 * s)
            img[y:y + ch, x:x + cw] = cv2.resize(crop, (cw, ch))
            if privacy:
                pixelate(img, x, y, x + cw, y + ch, block=max(4, cw // 12))
        cv2.putText(img, mask_plate(plate) if privacy else plate, (int(10 * s), y + int(95 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (255, 255, 255), max(1, int(2 * s)))


def main():
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=os.environ.get("CCTV_URL"), help="HLS/RTSP URL or video file (env CCTV_URL)")
    ap.add_argument("--zone", default=os.environ.get("ZONE") or "auto",
                    help='"auto" = centre band, or manual polygon "x,y x,y ..." in source pixels; OCR only runs inside (env ZONE)')
    ap.add_argument("--db", default="out/plates.db")
    ap.add_argument("--snapshots", default="out/snapshots")
    ap.add_argument("--save", help="write annotated video to this .mp4")
    ap.add_argument("--show", action="store_true", help="open a preview window")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = forever)")
    ap.add_argument("--vehicle-model", default="yolo11n.pt")
    ap.add_argument("--plate-model", default="yolo-v9-s-608-license-plate-end2end")
    ap.add_argument("--ocr-model", default="cct-s-v2-global-model")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-width", type=int, default=100, help="skip OCR on vehicles narrower than this (px)")
    ap.add_argument("--ocr-interval", type=float, default=0.2, help="seconds between OCR runs per track")
    ap.add_argument("--min-votes", type=int, default=2, help="reads of the same plate needed to save it")
    ap.add_argument("--min-conf", type=float, default=0.7, help="mean OCR confidence needed to save a plate")
    ap.add_argument("--dedupe-sec", type=float, default=60, help="skip a plate already saved within this window (track id switches)")
    ap.add_argument("--lost-sec", type=float, default=2.0, help="track unseen this long = finished")
    ap.add_argument("--stride", type=int, default=3, help="live only: process every Nth frame (25fps / 3 = ~8fps)")
    ap.add_argument("--privacy", action="store_true", help="overlay only: blur plates, mask digits (for public demos; DB unchanged)")
    args = ap.parse_args()
    if not args.source:
        ap.error("no source: set CCTV_URL in .env or pass --source")

    zone = None  # built from the first frame, auto mode needs the frame size
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(args.vehicle_model)
    alpr = ALPR(detector_model=args.plate_model, ocr_model=args.ocr_model)

    Path(args.snapshots).mkdir(parents=True, exist_ok=True)
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.db)
    db.execute("""CREATE TABLE IF NOT EXISTS plates (
        id INTEGER PRIMARY KEY, plate TEXT, votes INT, conf REAL, vehicle TEXT, track_id INT,
        source TEXT, first_seen TEXT, last_seen TEXT, snapshot TEXT)""")

    live = "://" in args.source
    reader = StreamReader(args.source, args.stride) if live else None
    cap = reader.cap if live else cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {args.source}")
    fps = min(cap.get(cv2.CAP_PROP_FPS) or 25, 30)  # HLS can report a bogus value like 90000
    writer = None

    tracks: dict[int, Track] = {}
    recent: list[tuple[str, np.ndarray | None]] = []
    last_saved: dict[str, float] = {}
    start, frames, fps_smooth = time.time(), 0, 0.0

    def finish(tid: int, t: Track):
        res = t.result(args.min_votes)
        if not res or res[2] < args.min_conf:
            return
        plate, votes, conf = res
        if is_duplicate(plate, t.last_seen, last_saved, args.dedupe_sec):
            return
        last_saved[plate] = t.last_seen
        crop = t.crops[plate][1]
        snap = ""
        if crop is not None and crop.size:
            snap = f"{args.snapshots}/{datetime.now():%Y%m%d_%H%M%S}_{plate}_{tid}.jpg"
            cv2.imwrite(snap, crop)
        db.execute(
            "INSERT INTO plates (plate, votes, conf, vehicle, track_id, source, first_seen, last_seen, snapshot) VALUES (?,?,?,?,?,?,?,?,?)",
            (plate, votes, round(conf, 3), t.vehicle, tid, args.source,
             datetime.fromtimestamp(t.first_seen).isoformat(timespec="seconds"),
             datetime.fromtimestamp(t.last_seen).isoformat(timespec="seconds"), snap))
        db.commit()
        recent.insert(0, (plate, crop))
        del recent[6:]
        print(f"{datetime.now():%H:%M:%S} {plate:<10} votes={votes} conf={conf:.2f} {t.vehicle} track={tid}", flush=True)

    try:
        while not args.duration or time.time() - start < args.duration:
            t0 = time.time()
            if live:
                frame = reader.read()
            else:
                ok, frame = cap.read()
                frame = frame if ok else None
            if frame is None:
                break
            now = time.time() if live else start + frames / fps
            frames += 1
            if zone is None:
                zone = parse_zone(args.zone, frame.shape[1], frame.shape[0])
                mode = "auto" if args.zone.strip().lower() == "auto" else "manual"
                print(f"zone ({mode}, {frame.shape[1]}x{frame.shape[0]}): {' '.join(f'{x},{y}' for x, y in zone)}", flush=True)

            r = yolo.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=args.imgsz,
                           classes=list(VEHICLES), conf=0.3, device=device, verbose=False)[0]
            s = max(frame.shape[0], 1) / 1080.0
            vis = frame.copy() if (args.show or args.save) else None

            if r.boxes.id is not None:
                for (x1, y1, x2, y2), tid, cls in zip(r.boxes.xyxy.int().tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
                    t = tracks.setdefault(tid, Track(VEHICLES[cls], now))
                    t.last_seen = now
                    in_zone = cv2.pointPolygonTest(zone, ((x1 + x2) / 2, y2), False) >= 0
                    if in_zone and x2 - x1 >= args.min_width and now - t.last_ocr >= args.ocr_interval:
                        t.last_ocr = now
                        hit = ocr_vehicle(alpr, frame[y1:y2, x1:x2])
                        if hit:
                            t.add(*hit[:3])
                            t.plate_box = hit[3]
                    if vis is not None and args.privacy and t.plate_box:
                        # ponytail: reuses the last OCR plate box relative to the vehicle; fine for slow
                        # traffic at the stop line, plates of vehicles never OCR'd stay unblurred.
                        px1, py1, px2, py2 = t.plate_box
                        pad = int(0.3 * (py2 - py1))
                        pixelate(vis, x1 + px1 - pad, y1 + py1 - pad, x1 + px2 + pad, y1 + py2 + pad)
                    if vis is not None and in_zone:
                        draw_corners(vis, x1, y1, x2, y2, (255, 160, 0), max(2, int(3 * s)))
                        if t.score:
                            label = t.score.most_common(1)[0][0]
                            cv2.putText(vis, mask_plate(label) if args.privacy else label, (x1, y1 - int(10 * s)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s, (0, 230, 255), max(2, int(3 * s)))

            for tid in [tid for tid, t in tracks.items() if now - t.last_seen > args.lost_sec]:
                finish(tid, tracks.pop(tid))

            if vis is not None:
                overlay = vis.copy()
                cv2.fillPoly(overlay, [zone], (0, 200, 255))
                vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)
                cv2.polylines(vis, [zone], True, (0, 200, 255), max(2, int(2 * s)))
                draw_panel(vis, recent, s, args.privacy)
                dt = time.time() - t0
                fps_smooth = 0.9 * fps_smooth + 0.1 / dt if fps_smooth else 1 / dt
                cv2.putText(vis, f"{fps_smooth:.1f} FPS", (vis.shape[1] - int(250 * s), int(50 * s)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2 * s, (0, 255, 0), max(2, int(3 * s)))
                if args.save:
                    out = cv2.resize(vis, (1920, int(1920 * vis.shape[0] / vis.shape[1])))
                    if writer is None:
                        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps / args.stride if live else fps, (out.shape[1], out.shape[0]))
                    writer.write(out)
                if args.show:
                    cv2.imshow("ALPR", cv2.resize(vis, (1280, int(1280 * vis.shape[0] / vis.shape[1]))))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.release()
        for tid, t in tracks.items():
            finish(tid, t)
        if reader:
            reader.release()
        else:
            cap.release()
        if args.show:
            cv2.destroyAllWindows()
        db.close()
        print(f"done: {frames} frames in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
