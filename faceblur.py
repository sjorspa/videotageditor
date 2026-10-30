"""Face detection (OpenCV YuNet), tracking over time and blurring of faces."""

import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
DETECT_MAX_WIDTH = 1280  # larger frames are downscaled for detection
SMOOTH = 2  # frames on each side averaged to steady the boxes
MIN_HITS = 2  # detections needed around a frame; a one-off hit is likely not a face


@dataclass
class FaceBlurSettings:
    mode: str = "blur"  # "blur" or "pixelate"
    strength: int = 5  # 1..10
    padding: int = 30  # % extra around each face
    sensitivity: int = 50  # 1..100, higher finds more (and more false) faces
    hold: float = 0.5  # seconds a box stays when the face is briefly lost


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(MODEL_PATH)
    return MODEL_PATH


class FaceDetector:
    def __init__(self):
        self.net = cv2.FaceDetectorYN.create(str(ensure_model()), "", (320, 320), 0.6, 0.3, 5000)
        self.size = None

    def detect(self, bgr: np.ndarray, sensitivity: int) -> list[tuple[float, float, float, float]]:
        """Faces as (x, y, w, h) in frame pixels."""
        h, w = bgr.shape[:2]
        scale = min(1.0, DETECT_MAX_WIDTH / w)
        img = cv2.resize(bgr, (round(w * scale), round(h * scale))) if scale < 1 else bgr
        size = (img.shape[1], img.shape[0])
        if size != self.size:
            self.net.setInputSize(size)
            self.size = size
        self.net.setScoreThreshold(0.95 - sensitivity * 0.004)  # 1 -> 0.95, 50 -> 0.75, 100 -> 0.55
        _, faces = self.net.detect(img)
        if faces is None:
            return []
        return [tuple(float(v) / scale for v in f[:4]) for f in faces]


def blur_faces(bgr: np.ndarray, boxes, s: FaceBlurSettings) -> np.ndarray:
    """Blur or pixelate an elliptical, soft-edged area over each box (in place)."""
    fh, fw = bgr.shape[:2]
    for x, y, w, h in boxes:
        pad = s.padding / 100
        cx, cy = x + w / 2, y + h / 2
        w, h = w * (1 + 2 * pad), h * (1 + 2 * pad)
        x0, y0 = max(int(cx - w / 2), 0), max(int(cy - h / 2), 0)
        x1, y1 = min(int(cx + w / 2), fw), min(int(cy + h / 2), fh)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        roi = bgr[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]
        if s.mode == "pixelate":
            blocks = max(2, 22 - 2 * s.strength)  # blocks across the face
            small = cv2.resize(roi, (blocks, max(1, round(blocks * rh / rw))), interpolation=cv2.INTER_AREA)
            out = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        else:
            # a wide blur looks the same when done at lower resolution, and is far cheaper
            sigma = max(rw, rh) * s.strength / 40
            f = max(1.0, sigma / 4)
            sw, sh = max(1, round(rw / f)), max(1, round(rh / f))
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
            out = cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / f), (rw, rh), interpolation=cv2.INTER_LINEAR)
        # soft-edged ellipse, drawn small and scaled up
        ms = min(1.0, 64 / max(rw, rh))
        mw, mh = max(2, round(rw * ms)), max(2, round(rh * ms))
        mask = np.zeros((mh, mw), np.float32)
        cv2.ellipse(mask, (mw // 2, mh // 2), (mw // 2, mh // 2), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(mw, mh) / 30)
        mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_LINEAR)[..., None]
        roi[:] = (out * mask + roi * (1 - mask)).astype(np.uint8)
    return bgr


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


class FaceTracker:
    """Links detections over time and fills short gaps on both sides.

    Frames go in with push() and come out `hold` frames later, so a face that is
    missed for a moment (or found a bit late) is still covered, at a position
    interpolated between the surrounding detections.
    """

    def __init__(self, detector: FaceDetector, settings: FaceBlurSettings, fps: float):
        self.det = detector
        self.s = settings
        self.hold = max(1, round(settings.hold * fps))
        self.tracks: list[dict[int, tuple]] = []  # per track: frame index -> box
        self.buffer: deque = deque()  # (index, frame) waiting for lookahead
        self.index = 0

    def push(self, bgr: np.ndarray) -> list[np.ndarray]:
        """Add a frame; returns the frames that are ready (blurred), in order."""
        i = self.index
        self.index += 1
        for box in self.det.detect(bgr, self.s.sensitivity):
            self._assign(i, box)
        self.buffer.append((i, bgr))
        out = []
        while self.buffer and self.buffer[0][0] <= i - self.hold:
            out.append(self._render(*self.buffer.popleft()))
        return out

    def flush(self) -> list[np.ndarray]:
        out = [self._render(j, f) for j, f in self.buffer]
        self.buffer.clear()
        return out

    def _assign(self, i: int, box):
        best, best_score = None, 0.0
        for t in self.tracks:
            last = max(t)
            if last == i or i - last > 2 * self.hold:
                continue
            prev = t[last]
            score = _iou(prev, box)
            # fast movement: also accept a nearby centre
            dist = np.hypot(prev[0] + prev[2] / 2 - box[0] - box[2] / 2, prev[1] + prev[3] / 2 - box[1] - box[3] / 2)
            if score < 0.1 and dist < max(prev[2], prev[3]):
                score = 0.1
            if score > best_score:
                best, best_score = t, score
        if best is None:
            self.tracks.append({i: box})
        else:
            best[i] = box

    def boxes_at(self, j: int) -> list[tuple]:
        boxes = []
        for t in self.tracks:
            if sum(1 for k in t if abs(k - j) <= self.hold) < MIN_HITS:
                continue
            near = [t[k] for k in range(j - SMOOTH, j + SMOOTH + 1) if k in t]
            if near:
                boxes.append(tuple(np.mean(near, axis=0)))
                continue
            before = [k for k in t if j - self.hold <= k < j]
            after = [k for k in t if j < k <= j + self.hold]
            if before and after:
                p, n = max(before), min(after)
                a = (j - p) / (n - p)
                boxes.append(tuple(np.add(np.multiply(t[p], 1 - a), np.multiply(t[n], a))))
            elif before or after:
                boxes.append(t[max(before) if before else min(after)])
        return boxes

    def _render(self, j: int, bgr: np.ndarray) -> np.ndarray:
        boxes = self.boxes_at(j)
        # drop what can no longer influence upcoming frames
        for t in self.tracks:
            for k in [k for k in t if k < j - self.hold]:
                del t[k]
        self.tracks = [t for t in self.tracks if t]
        return blur_faces(bgr, boxes, self.s)
