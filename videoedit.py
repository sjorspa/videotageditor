#!/usr/bin/env python3
"""Simple video editor: pick a segment, crop, and export with ffmpeg."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from PySide6.QtCore import QEvent, QObject, QPointF, QProcess, QRectF, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from batch import BatchTab
from combine import CombineTab, probe
from faceblur import FaceBlurSettings, FaceDetector, FaceTracker, blur_faces
from tagger import TaggerTab

HANDLE = 10  # px, size of the crop rectangle handles
MIN_CROP = 16  # minimum crop size in video pixels


def blur_image(img: QImage, kind: str, strength: float, passes: int) -> QImage:
    """Blur a frame the same way the export does (gaussian: gblur, box: boxblur)."""
    img = img.convertToFormat(QImage.Format_RGBA8888)
    w, h = img.width(), img.height()
    pil = Image.frombuffer("RGBA", (w, h), bytes(img.constBits()), "raw", "RGBA", img.bytesPerLine(), 1)
    flt = ImageFilter.GaussianBlur(strength) if kind == "gaussian" else ImageFilter.BoxBlur(round(strength))
    for _ in range(passes):
        pil = pil.filter(flt)
    return QImage(pil.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888).copy()


def blur_filter(kind: str, strength: float, passes: int) -> str:
    """ffmpeg filter matching blur_image()."""
    if kind == "gaussian":
        # n gaussian passes with sigma s equal one pass with sigma s * sqrt(n)
        # full-res chroma first, otherwise colours get blurred twice as much as in the preview
        return f"format=yuv444p,gblur=sigma={strength * passes ** 0.5:.3f}"
    r = round(strength)
    return f"boxblur=luma_radius={r}:luma_power={passes}:chroma_radius={r // 2}:chroma_power={passes}"


def qimage_to_bgr(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, :w * 3]
    return cv2.cvtColor(arr.reshape(h, w, 3), cv2.COLOR_RGB2BGR)


def bgr_to_qimage(arr: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class SelectAllOnFocus(QObject):
    """Select a spin box's whole value when it gets focus, so typing replaces it.

    Otherwise a click puts the cursor inside the number and digits get appended
    ("120" + "80" = "12080"), which the range validator silently rejects.
    """

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.FocusIn and ev.reason() in (Qt.MouseFocusReason, Qt.TabFocusReason,
                                                            Qt.BacktabFocusReason, Qt.ShortcutFocusReason):
            QTimer.singleShot(0, obj.selectAll)  # after the click has placed the cursor
        return False


def save_widget(st: QSettings, key: str, w):
    if isinstance(w, QCheckBox):
        st.setValue(key, w.isChecked())
    elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
        st.setValue(key, w.value())
    elif isinstance(w, QComboBox):
        st.setValue(key, w.currentText())
    elif isinstance(w, QLineEdit):
        st.setValue(key, w.text())
    elif isinstance(w, QPlainTextEdit):
        st.setValue(key, w.toPlainText())


def restore_widget(st: QSettings, key: str, w):
    """Restore a widget from settings; values come back as strings from the ini file."""
    v = st.value(key)
    if v is None:
        return
    try:
        if isinstance(w, QCheckBox):
            w.setChecked(v in (True, "true", "1", 1))
        elif isinstance(w, QSpinBox):
            w.setValue(int(float(v)))
        elif isinstance(w, QDoubleSpinBox):
            w.setValue(float(v))
        elif isinstance(w, QComboBox):
            if (i := w.findText(str(v))) >= 0:
                w.setCurrentIndex(i)
        elif isinstance(w, QLineEdit):
            w.setText(str(v))
        elif isinstance(w, QPlainTextEdit):
            w.setPlainText(str(v))
    except (TypeError, ValueError):
        pass


def fmt_time(ms: int) -> str:
    s, ms = divmod(max(ms, 0), 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}.{ms // 10:02d}" if h else f"{m:02d}:{s:02d}.{ms // 10:02d}"


class VideoView(QWidget):
    """Shows the video frame with a draggable/resizable crop rectangle."""

    cropChanged = Signal(QRectF)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 270)
        self.setMouseTracking(True)
        self.image = QImage()  # displayed frame (blurred if enabled)
        self.raw = QImage()  # frame as decoded
        self.blur = None  # (kind, strength, passes) or None
        self.face_blur = None  # FaceBlurSettings or None
        self.detector = None
        self.crop = QRectF()  # in video pixels
        self._aspect = None  # locked crop aspect ratio (w / h), "orig" or None
        self._drag_mode = None
        self._drag_start = QPointF()
        self._crop_start = QRectF()

    # --- coordinates ---------------------------------------------------
    def video_rect(self) -> QRectF:
        """Area in widget coordinates where the video is drawn."""
        if self.image.isNull():
            return QRectF()
        iw, ih = self.image.width(), self.image.height()
        scale = min(self.width() / iw, self.height() / ih)
        w, h = iw * scale, ih * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def scale(self) -> float:
        vr = self.video_rect()
        return vr.width() / self.image.width() if not vr.isNull() else 1.0

    def to_widget(self, r: QRectF) -> QRectF:
        vr, s = self.video_rect(), self.scale()
        return QRectF(vr.x() + r.x() * s, vr.y() + r.y() * s, r.width() * s, r.height() * s)

    # --- data ------------------------------------------------------------
    def set_frame(self, frame):
        img = frame.toImage()
        if img.isNull():
            return
        first = self.image.isNull() or img.size() != self.image.size()
        self.raw = img
        self.image = self.process(img)
        if first:
            self.reset_crop()
        self.update()

    def process(self, img: QImage) -> QImage:
        """Apply the effects to a frame, in the same order as the export."""
        if self.face_blur:
            if self.detector is None:
                self.detector = FaceDetector()
            bgr = qimage_to_bgr(img)
            # single frame: no tracking in the preview, the export fills gaps
            img = bgr_to_qimage(blur_faces(bgr, self.detector.detect(bgr, self.face_blur.sensitivity),
                                           self.face_blur))
        return blur_image(img, *self.blur) if self.blur else img

    def refresh(self):
        if not self.raw.isNull():
            self.image = self.process(self.raw)
            self.update()

    def set_blur(self, blur):
        self.blur = blur
        self.refresh()

    def set_face_blur(self, settings):
        self.face_blur = settings
        self.refresh()

    def reset_crop(self):
        if not self.image.isNull():
            self.set_crop(self.largest_crop(self.image.rect().center()))

    def largest_crop(self, center) -> QRectF:
        """Largest crop with the locked aspect ratio, centred on `center`."""
        iw, ih = self.image.width(), self.image.height()
        if not self.aspect:
            return QRectF(0, 0, iw, ih)
        w = min(iw, ih * self.aspect)
        h = w / self.aspect
        return QRectF(center.x() - w / 2, center.y() - h / 2, w, h)

    @property
    def aspect(self):
        if self._aspect == "orig":
            return self.image.width() / self.image.height() if not self.image.isNull() else None
        return self._aspect

    def set_aspect(self, aspect):
        self._aspect = aspect
        if aspect and not self.image.isNull():
            self.set_crop(self.largest_crop(self.crop.center()))

    def set_crop(self, r: QRectF):
        if self.image.isNull():
            return
        iw, ih = self.image.width(), self.image.height()
        if self.aspect:
            # shrink uniformly so the ratio survives clamping to the frame
            f = min(1.0, iw / max(r.width(), 1), ih / max(r.height(), 1))
            r = QRectF(r.center().x() - r.width() * f / 2, r.center().y() - r.height() * f / 2,
                       r.width() * f, r.height() * f)
        w = min(max(r.width(), MIN_CROP), iw)
        h = min(max(r.height(), MIN_CROP), ih)
        x = min(max(r.x(), 0), iw - w)
        y = min(max(r.y(), 0), ih - h)
        self.crop = QRectF(round(x), round(y), round(w), round(h))
        self.cropChanged.emit(self.crop)
        self.update()

    # --- painting ---------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        if self.image.isNull():
            p.setPen(QColor(160, 160, 160))
            p.drawText(self.rect(), Qt.AlignCenter, "Open a video (Ctrl+O)")
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        vr = self.video_rect()
        p.drawImage(vr, self.image)

        cr = self.to_widget(self.crop)
        # darken the area outside the crop
        shade = QColor(0, 0, 0, 140)
        p.fillRect(QRectF(vr.left(), vr.top(), vr.width(), cr.top() - vr.top()), shade)
        p.fillRect(QRectF(vr.left(), cr.bottom(), vr.width(), vr.bottom() - cr.bottom()), shade)
        p.fillRect(QRectF(vr.left(), cr.top(), cr.left() - vr.left(), cr.height()), shade)
        p.fillRect(QRectF(cr.right(), cr.top(), vr.right() - cr.right(), cr.height()), shade)

        p.setPen(QPen(QColor(255, 200, 0), 2))
        p.drawRect(cr)
        p.setBrush(QColor(255, 200, 0))
        for pt in self._handle_points(cr).values():
            p.drawRect(QRectF(pt.x() - HANDLE / 2, pt.y() - HANDLE / 2, HANDLE, HANDLE))

    @staticmethod
    def _handle_points(r: QRectF) -> dict:
        cx, cy = r.center().x(), r.center().y()
        return {
            "tl": r.topLeft(), "tr": r.topRight(), "bl": r.bottomLeft(), "br": r.bottomRight(),
            "t": QPointF(cx, r.top()), "b": QPointF(cx, r.bottom()),
            "l": QPointF(r.left(), cy), "r": QPointF(r.right(), cy),
        }

    def _hit(self, pos: QPointF):
        if self.image.isNull():
            return None
        cr = self.to_widget(self.crop)
        for name, pt in self._handle_points(cr).items():
            if abs(pos.x() - pt.x()) <= HANDLE and abs(pos.y() - pt.y()) <= HANDLE:
                return name
        return "move" if cr.contains(pos) else None

    # --- mouse ------------------------------------------------------------
    CURSORS = {
        "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
        "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
        "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
        "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
        "move": Qt.SizeAllCursor,
    }

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_mode = self._hit(e.position())
            self._drag_start = e.position()
            self._crop_start = QRectF(self.crop)

    def mouseMoveEvent(self, e):
        if not self._drag_mode:
            self.setCursor(self.CURSORS.get(self._hit(e.position()), Qt.ArrowCursor))
            return
        s = self.scale()
        dx = (e.position().x() - self._drag_start.x()) / s
        dy = (e.position().y() - self._drag_start.y()) / s
        r = QRectF(self._crop_start)
        mode = self._drag_mode
        if mode == "move":
            r.translate(dx, dy)
            self.set_crop(r)
            return
        iw, ih = self.image.width(), self.image.height()
        if self.aspect:
            self.set_crop(self._resize_with_aspect(r, mode, dx, dy, iw, ih))
            return
        left, top, right, bottom = r.left(), r.top(), r.right(), r.bottom()
        if "l" in mode:
            left = min(max(left + dx, 0), right - MIN_CROP)
        if "r" in mode:
            right = max(min(right + dx, iw), left + MIN_CROP)
        if "t" in mode:
            top = min(max(top + dy, 0), bottom - MIN_CROP)
        if "b" in mode:
            bottom = max(min(bottom + dy, ih), top + MIN_CROP)
        self.set_crop(QRectF(QPointF(left, top), QPointF(right, bottom)))

    def _resize_with_aspect(self, r: QRectF, mode: str, dx: float, dy: float, iw: int, ih: int) -> QRectF:
        a = self.aspect
        min_w = max(MIN_CROP, MIN_CROP * a)
        if len(mode) == 2:
            # corner: the opposite corner stays put
            ax = r.right() if "l" in mode else r.left()
            ay = r.bottom() if "t" in mode else r.top()
            w = r.width() + (-dx if "l" in mode else dx)
            h = r.height() + (-dy if "t" in mode else dy)
            max_w = ax if "l" in mode else iw - ax
            max_h = ay if "t" in mode else ih - ay
            w = max(min(max(w, h * a), max_w, max_h * a), min_w)
            h = w / a
            return QRectF(ax - w if "l" in mode else ax, ay - h if "t" in mode else ay, w, h)
        if mode in ("l", "r"):
            # side: the opposite side stays put, height follows around the centre
            w = r.width() + (-dx if mode == "l" else dx)
            w = max(min(w, r.right() if mode == "l" else iw - r.left(), ih * a), min_w)
            h = w / a
            return QRectF(r.right() - w if mode == "l" else r.left(), r.center().y() - h / 2, w, h)
        h = r.height() + (-dy if mode == "t" else dy)
        h = max(min(h, r.bottom() if mode == "t" else ih - r.top(), iw / a), min_w / a)
        w = h * a
        return QRectF(r.center().x() - w / 2, r.bottom() - h if mode == "t" else r.top(), w, h)

    def mouseReleaseEvent(self, _):
        self._drag_mode = None


class FaceExportWorker(QThread):
    """Export with face blur: ffmpeg decodes, Python blurs the faces, ffmpeg encodes."""

    progress = Signal(int)  # 0..1000
    done = Signal(bool, bool, str)  # ok, cancelled, error text

    def __init__(self, parent, ffmpeg, src, ss, t, fps, w, h, vf, encode, out, settings):
        super().__init__(parent)
        self.ffmpeg, self.src, self.ss, self.t, self.fps = ffmpeg, src, ss, t, fps
        self.w, self.h, self.vf, self.encode, self.out = w, h, vf, encode, out
        self.settings = settings
        self.cancelled = False

    def kill(self):
        self.cancelled = True

    def run(self):
        ff, w, h, rate = self.ffmpeg, self.w, self.h, f"{self.fps:g}"
        dec_err, enc_err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
        # scale to the preview size so the crop coordinates match exactly
        dec = subprocess.Popen(
            [ff, "-v", "error", "-ss", self.ss, "-i", self.src, "-t", self.t,
             "-vf", f"fps={rate},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, stderr=dec_err)
        enc = subprocess.Popen(
            [ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", rate,
             "-i", "-", "-ss", self.ss, "-t", self.t, "-i", self.src, "-map", "0:v:0", "-map", "1:a:0?",
             "-vf", self.vf, *self.encode, self.out],
            stdin=subprocess.PIPE, stderr=enc_err)
        total = max(1, round(float(self.t) * self.fps))
        written = 0
        err = ""
        try:
            tracker = FaceTracker(FaceDetector(), self.settings, self.fps)
            size = w * h * 3
            while not self.cancelled:
                buf = dec.stdout.read(size)
                if len(buf) < size:
                    break
                frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
                for f in tracker.push(frame):
                    enc.stdin.write(f.tobytes())
                    written += 1
                    self.progress.emit(min(1000, written * 1000 // total))
            if not self.cancelled:
                for f in tracker.flush():
                    enc.stdin.write(f.tobytes())
        except Exception as e:  # broken pipe when ffmpeg fails, model errors, ...
            err = str(e)
        finally:
            dec.kill()
            if self.cancelled:
                enc.kill()
            else:
                try:
                    enc.stdin.close()
                except OSError:
                    pass
            dec.wait()
            code = enc.wait()
        if self.cancelled:
            Path(self.out).unlink(missing_ok=True)
            self.done.emit(False, True, "")
            return
        for f in (dec_err, enc_err):
            f.seek(0)
            err += "\n" + f.read().decode(errors="ignore")
        self.done.emit(code == 0 and written > 0, False, err.strip() or "No frames were decoded.")


class ImportDialog(QDialog):
    """Asks for a URL, a maximum quality and a download folder."""

    def __init__(self, parent, folder: Path):
        super().__init__(parent)
        self.setWindowTitle("Import from URL")
        self.resize(560, 0)
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://…")
        clip = QGuiApplication.clipboard().text().strip()
        if clip.startswith(("http://", "https://")):
            self.url_edit.setText(clip)
        self.quality = QComboBox()
        for label, height in (("Best", 0), ("2160p", 2160), ("1440p", 1440), ("1080p", 1080),
                              ("720p", 720), ("480p", 480), ("360p", 360)):
            self.quality.addItem(label, height)
        self.quality.setCurrentIndex(3)
        self.folder_edit = QLineEdit(str(folder))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(browse)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Download")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form = QFormLayout(self)
        form.addRow("URL", self.url_edit)
        form.addRow("Max quality", self.quality)
        form.addRow("Save to", folder_row)
        form.addRow(buttons)
        self.url_edit.setFocus()
        self.url_edit.selectAll()

    def browse(self):
        d = QFileDialog.getExistingDirectory(self, "Download folder", self.folder_edit.text())
        if d:
            self.folder_edit.setText(d)

    def accept(self):
        if self.url_edit.text().strip():
            super().accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Editor")
        self.resize(1100, 800)
        self.path: Path | None = None
        self.duration = 0
        self.export_proc: QProcess | None = None
        self.undo_stack: list[tuple[str, Path, str]] = []  # (path, export dir, export name) before each cut
        self.temp_dir: Path | None = None  # cut results, removed on exit
        self.cut_proc: QProcess | None = None
        self._seek_after_load: int | None = None
        self.export_total_ms = 0

        # player
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.sink = QVideoSink(self)
        self.player.setVideoSink(self.sink)
        self.view = VideoView()
        self.sink.videoFrameChanged.connect(self.view.set_frame)
        self.player.durationChanged.connect(self.on_duration)
        self.player.positionChanged.connect(self.on_position)
        self.player.playbackStateChanged.connect(self.on_state)
        self.player.mediaStatusChanged.connect(self.on_media_status)

        # buttons
        open_btn = QPushButton("Open video…")
        open_btn.setShortcut("Ctrl+O")
        open_btn.clicked.connect(self.open_file)
        import_btn = QPushButton("Import from URL…")
        import_btn.setShortcut("Ctrl+U")
        import_btn.clicked.connect(self.import_url)
        self.tag_btn = QPushButton("Tag current video")
        self.tag_btn.setShortcut("Ctrl+T")
        self.tag_btn.setToolTip("Describe this video with the Tagger settings and save it as a .txt file")
        self.tag_btn.clicked.connect(self.tag_current)
        videos = Path.home() / "Videos"
        self.download_dir = videos if videos.is_dir() else Path.home()
        self.dl_proc: QProcess | None = None
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setShortcut("Space")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_seg_btn = QPushButton("▶ Play segment")
        self.play_seg_btn.clicked.connect(self.play_segment)
        self.loop_chk = QCheckBox("Loop segment")
        self.loop_chk.setChecked(True)

        # position
        self.pos_slider = QSlider(Qt.Horizontal)
        self.pos_slider.sliderMoved.connect(self.on_pos_moved)
        self.pos_label = QLabel("00:00.00 / 00:00.00")

        # segment
        self.start_slider = QSlider(Qt.Horizontal)
        self.end_slider = QSlider(Qt.Horizontal)
        self.start_slider.valueChanged.connect(self.on_start_changed)
        self.end_slider.valueChanged.connect(self.on_end_changed)
        self.start_label = QLabel("00:00.00")
        self.end_label = QLabel("00:00.00")
        self.seg_label = QLabel("Duration: 0.00 s")
        set_start_btn = QPushButton("Start = current position")
        set_start_btn.clicked.connect(lambda: self.start_slider.setValue(self.player.position()))
        set_end_btn = QPushButton("End = current position")
        set_end_btn.clicked.connect(lambda: self.end_slider.setValue(self.player.position()))

        self.lock_chk = QCheckBox("Fixed duration (s):")
        self.lock_spin = QDoubleSpinBox()
        self.lock_spin.setRange(0.1, 36000)
        self.lock_spin.setDecimals(2)
        self.lock_spin.setValue(15.0)
        self.lock_chk.toggled.connect(self.apply_lock)
        self.lock_spin.valueChanged.connect(self.apply_lock)

        seg_box = QGroupBox("Segment")
        g = QGridLayout(seg_box)
        g.addWidget(QLabel("Start"), 0, 0)
        g.addWidget(self.start_slider, 0, 1)
        g.addWidget(self.start_label, 0, 2)
        g.addWidget(set_start_btn, 0, 3)
        g.addWidget(QLabel("End"), 1, 0)
        g.addWidget(self.end_slider, 1, 1)
        g.addWidget(self.end_label, 1, 2)
        g.addWidget(set_end_btn, 1, 3)
        lock_row = QHBoxLayout()
        lock_row.addWidget(self.lock_chk)
        lock_row.addWidget(self.lock_spin)
        lock_row.addSpacing(20)
        lock_row.addWidget(self.seg_label)
        lock_row.addStretch()
        self.cut_btn = QPushButton("✂ Cut out selection")
        self.cut_btn.setToolTip("Remove the selected part and join what comes before and after it")
        self.cut_btn.clicked.connect(self.cut_selection)
        self.undo_btn = QPushButton("Undo cut")
        self.undo_btn.setShortcut("Ctrl+Z")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self.undo_cut)
        lock_row.addWidget(self.cut_btn)
        lock_row.addWidget(self.undo_btn)
        g.addLayout(lock_row, 2, 0, 1, 4)

        # crop
        self.crop_spins = {}
        crop_box = QGroupBox("Crop (drag the rectangle in the video)")
        ch = QHBoxLayout(crop_box)
        for key in ("x", "y", "w", "h"):
            sp = QSpinBox()
            sp.setRange(0, 100000)
            sp.editingFinished.connect(self.on_crop_spin)
            self.crop_spins[key] = sp
            ch.addWidget(QLabel(key.upper()))
            ch.addWidget(sp)
        self.aspect_combo = QComboBox()
        for label, ratio in (("Free", None), ("Original", "orig"), ("16:9", 16 / 9), ("9:16", 9 / 16),
                             ("1:1", 1.0), ("4:5", 4 / 5), ("4:3", 4 / 3), ("3:4", 3 / 4), ("21:9", 21 / 9)):
            self.aspect_combo.addItem(label, ratio)
        self.aspect_combo.currentIndexChanged.connect(self.on_aspect_changed)
        ch.addWidget(QLabel("Aspect"))
        ch.addWidget(self.aspect_combo)
        reset_btn = QPushButton("Full frame")
        reset_btn.clicked.connect(self.view.reset_crop)
        ch.addWidget(reset_btn)
        ch.addStretch()
        self.view.cropChanged.connect(self.on_crop_changed)

        # face blur
        self.face_chk = QCheckBox("Enable")
        self.face_mode = QComboBox()
        self.face_mode.addItem("Blur", "blur")
        self.face_mode.addItem("Pixelate", "pixelate")
        self.face_strength = QSpinBox()
        self.face_strength.setRange(1, 10)
        self.face_strength.setValue(5)
        self.face_padding = QSpinBox()
        self.face_padding.setRange(0, 150)
        self.face_padding.setValue(30)
        self.face_padding.setSuffix(" %")
        self.face_padding.setToolTip("Extra area around each face")
        self.face_sens = QSpinBox()
        self.face_sens.setRange(1, 100)
        self.face_sens.setValue(50)
        self.face_sens.setToolTip("Higher finds more faces, but also more false ones")
        self.face_hold = QDoubleSpinBox()
        self.face_hold.setRange(0.05, 5)
        self.face_hold.setSingleStep(0.1)
        self.face_hold.setValue(0.5)
        self.face_hold.setSuffix(" s")
        self.face_hold.setToolTip("How long a face stays covered when detection briefly loses it "
                                  "(export only; the preview looks at single frames)")
        for wdg in (self.face_chk, self.face_mode, self.face_strength, self.face_padding,
                    self.face_sens, self.face_hold):
            sig = wdg.toggled if wdg is self.face_chk else (
                wdg.currentIndexChanged if isinstance(wdg, QComboBox) else wdg.valueChanged)
            sig.connect(self.update_face_blur)
        face_box = QGroupBox("Face blur")
        fh = QHBoxLayout(face_box)
        for label, wdg in ((None, self.face_chk), ("Mode", self.face_mode), ("Strength", self.face_strength),
                           ("Padding", self.face_padding), ("Sensitivity", self.face_sens),
                           ("Hold", self.face_hold)):
            if label:
                fh.addWidget(QLabel(label))
            fh.addWidget(wdg)
        fh.addStretch()

        # blur
        self.blur_chk = QCheckBox("Enable")
        self.blur_type = QComboBox()
        self.blur_type.addItem("Gaussian", "gaussian")
        self.blur_type.addItem("Box", "box")
        self.blur_strength = QDoubleSpinBox()
        self.blur_strength.setRange(0.5, 100)
        self.blur_strength.setDecimals(1)
        self.blur_strength.setValue(5)
        self.blur_strength.setSuffix(" px")
        self.blur_passes = QSpinBox()
        self.blur_passes.setRange(1, 10)
        self.blur_chk.toggled.connect(self.update_blur)
        self.blur_type.currentIndexChanged.connect(self.update_blur)
        self.blur_strength.valueChanged.connect(self.update_blur)
        self.blur_passes.valueChanged.connect(self.update_blur)
        blur_box = QGroupBox("Blur")
        bh = QHBoxLayout(blur_box)
        bh.addWidget(self.blur_chk)
        bh.addWidget(QLabel("Type"))
        bh.addWidget(self.blur_type)
        bh.addWidget(QLabel("Strength"))
        bh.addWidget(self.blur_strength)
        bh.addWidget(QLabel("Passes"))
        bh.addWidget(self.blur_passes)
        bh.addStretch()

        # export
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setValue(24)
        self.audio_chk = QCheckBox("Include audio")
        self.audio_chk.setChecked(True)
        self.export_btn = QPushButton("Export…")
        self.export_btn.clicked.connect(self.export)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        exp_box = QGroupBox("Export")
        eh = QHBoxLayout(exp_box)
        eh.addWidget(QLabel("FPS"))
        eh.addWidget(self.fps_spin)
        eh.addWidget(self.audio_chk)
        eh.addStretch()
        eh.addWidget(self.progress, 1)
        eh.addWidget(self.export_btn)

        # layout
        top = QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(import_btn)
        top.addWidget(self.tag_btn)
        top.addWidget(self.play_btn)
        top.addWidget(self.play_seg_btn)
        top.addWidget(self.loop_chk)
        top.addStretch()
        pos_row = QHBoxLayout()
        pos_row.addWidget(self.pos_slider, 1)
        pos_row.addWidget(self.pos_label)

        central = QWidget()
        v = QVBoxLayout(central)
        v.addLayout(top)
        v.addWidget(self.view, 1)
        v.addLayout(pos_row)
        v.addWidget(seg_box)
        v.addWidget(crop_box)
        v.addWidget(face_box)
        v.addWidget(blur_box)
        v.addWidget(exp_box)
        self.combine = CombineTab()
        self.combine.combined.connect(self.on_combined)
        self.tabs = QTabWidget()
        self.tabs.addTab(central, "Editor")
        self.tabs.addTab(self.combine, "Combine")
        self.tagger = TaggerTab("video")
        self.tabs.addTab(self.tagger, "Video Tagger")
        self.img_tagger = TaggerTab("image")
        self.tabs.addTab(self.img_tagger, "Image Tagger")
        self.batch = BatchTab()
        self.tabs.addTab(self.batch, "Batch")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.setCentralWidget(self.tabs)
        self.last_open_dir = Path.home()
        self.restore_settings()
        self._select_all = SelectAllOnFocus(self)
        for sb in self.findChildren(QAbstractSpinBox):
            sb.installEventFilter(self._select_all)
        self.set_enabled(False)

    # --- general ----------------------------------------------------------
    def set_enabled(self, on: bool):
        for w in (self.play_btn, self.play_seg_btn, self.pos_slider, self.start_slider,
                  self.end_slider, self.export_btn, self.cut_btn, self.tag_btn, *self.crop_spins.values()):
            w.setEnabled(on)

    def open_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open video", str(self.last_open_dir),
            "Video (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.mpg *.mpeg *.wmv);;All files (*)")
        if fn:
            self.last_open_dir = Path(fn).parent
            self.load(fn)

    # --- settings (remembered between sessions) ----------------------------------
    def persist_widgets(self) -> dict:
        return {
            "editor/loop": self.loop_chk, "editor/fixed": self.lock_chk, "editor/fixed_s": self.lock_spin,
            "editor/aspect": self.aspect_combo, "export/fps": self.fps_spin, "export/audio": self.audio_chk,
            "face/enabled": self.face_chk, "face/mode": self.face_mode, "face/strength": self.face_strength,
            "face/padding": self.face_padding, "face/sensitivity": self.face_sens, "face/hold": self.face_hold,
            "blur/enabled": self.blur_chk, "blur/type": self.blur_type, "blur/strength": self.blur_strength,
            "blur/passes": self.blur_passes,
            **self.tagger.persist_widgets(),
            **self.img_tagger.persist_widgets(),
            **self.batch.persist_widgets(),
        }

    def restore_settings(self):
        st = QSettings("videoedit", "videoedit")
        for key, w in self.persist_widgets().items():
            restore_widget(st, key, w)
        for attr, key in (("last_open_dir", "dirs/open"), ("download_dir", "dirs/download")):
            if (d := st.value(key)) and Path(d).is_dir():
                setattr(self, attr, Path(d))
        if (d := st.value("dirs/combine")) and Path(d).is_dir():
            self.combine.last_dir = Path(d)
        for t in (self.tagger, self.img_tagger):
            t.preferred_model = st.value(f"{t.cfg['key']}/model", "") or ""
            t.after_restore()
        if geo := st.value("window/geometry"):
            self.restoreGeometry(geo)
        self.tabs.setCurrentIndex(int(st.value("window/tab", 0) or 0))

    def save_settings(self):
        st = QSettings("videoedit", "videoedit")
        for key, w in self.persist_widgets().items():
            save_widget(st, key, w)
        st.setValue("dirs/open", str(self.last_open_dir))
        st.setValue("dirs/download", str(self.download_dir))
        st.setValue("dirs/combine", str(self.combine.last_dir))
        for t in (self.tagger, self.img_tagger):
            st.setValue(f"{t.cfg['key']}/model", t.current_model())
        st.setValue("window/geometry", self.saveGeometry())
        st.setValue("window/tab", self.tabs.currentIndex())

    def on_tab_changed(self, idx: int):
        tab = self.tabs.widget(idx)
        if isinstance(tab, TaggerTab) and tab.model_combo.count() == 0:
            tab.refresh_models()

    # --- import from URL (yt-dlp) ----------------------------------------------
    def import_url(self):
        dlg = ImportDialog(self, self.download_dir)
        if dlg.exec() != QDialog.Accepted:
            return
        self.download_dir = Path(dlg.folder_edit.text()).expanduser()
        self.download_dir.mkdir(parents=True, exist_ok=True)
        height = dlg.quality.currentData()
        # prefer h264/aac in mp4: plays everywhere and loads fast
        sort = f"res:{height},vcodec:h264,acodec:m4a" if height else "vcodec:h264,acodec:m4a"
        args = ["-m", "yt_dlp", "--no-playlist", "--newline", "--progress", "--no-simulate",
                "-f", "bv*+ba/b", "-S", sort, "--merge-output-format", "mp4",
                "-o", str(self.download_dir / "%(title).150B [%(id)s].%(ext)s"),
                "--progress-template", "download:PROGRESS %(progress._percent_str)s",
                "--print", "after_move:FILE %(filepath)s",
                dlg.url_edit.text().strip()]

        self.dl_file = None
        self.dl_err = b""
        self.dl_dialog = QProgressDialog("Downloading…", "Cancel", 0, 1000, self)
        self.dl_dialog.setWindowTitle("Import from URL")
        self.dl_dialog.setMinimumDuration(0)
        self.dl_dialog.setValue(0)
        proc = QProcess(self)
        proc.readyReadStandardOutput.connect(self.on_dl_output)
        proc.readyReadStandardError.connect(
            lambda: setattr(self, "dl_err", self.dl_err + proc.readAllStandardError().data()))
        proc.finished.connect(self.on_dl_finished)
        self.dl_dialog.canceled.connect(proc.kill)
        self.dl_proc = proc
        proc.start(sys.executable, args)

    def on_dl_output(self):
        data = self.dl_proc.readAllStandardOutput().data().decode(errors="ignore")
        for line in data.splitlines():
            if line.startswith("PROGRESS "):
                try:
                    pct = float(line.split()[1].rstrip("%"))
                except (IndexError, ValueError):
                    continue
                self.dl_dialog.setValue(int(pct * 10))
            elif line.startswith("FILE "):
                self.dl_file = line[5:].strip()
                self.dl_dialog.setLabelText("Finishing…")

    def on_dl_finished(self, code, status):
        self.dl_proc = None
        self.dl_dialog.close()
        if status == QProcess.NormalExit and code == 0 and self.dl_file:
            self.load(self.dl_file)
        elif status != QProcess.CrashExit:
            err = [l for l in self.dl_err.decode(errors="ignore").splitlines() if l.strip()][-10:]
            QMessageBox.critical(self, "Import failed", "\n".join(err) or f"yt-dlp exited with code {code}")

    def on_combined(self, path: str, export_dir: str):
        self.load(path, export_dir)
        self.tabs.setCurrentIndex(0)

    def closeEvent(self, e):
        self.save_settings()
        self.tagger.shutdown()
        self.img_tagger.shutdown()
        self.batch.shutdown()
        self.combine.cleanup()
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().closeEvent(e)

    def load(self, fn: str, export_dir: str | None = None, stem: str | None = None, edit: bool = False):
        """Open a video. Exports default to `export_dir`/`stem`_edit.mp4 (for temp files).

        `edit` marks the result of a cut (or its undo): crop and undo history are kept.
        """
        self.path = Path(fn)
        self.export_dir = Path(export_dir) if export_dir else self.path.parent
        self.export_stem = stem or self.path.stem
        if not edit:
            self.view.image = self.view.raw = QImage()
            self.undo_stack.clear()
        self.undo_btn.setEnabled(bool(self.undo_stack))
        self.player.setSource(QUrl.fromLocalFile(str(self.path)))
        title = self.export_stem + (f" ({len(self.undo_stack)} cut{'s' if len(self.undo_stack) > 1 else ''})"
                                    if self.undo_stack else "")
        self.setWindowTitle(f"Video Editor – {title}")
        # show the first frame
        self.player.play()
        self.player.pause()

    def on_duration(self, d: int):
        self.duration = d
        for s in (self.pos_slider, self.start_slider, self.end_slider):
            s.setRange(0, d)
        self._seg_len = d
        self.start_slider.setValue(0)
        self.end_slider.setValue(d)
        self.apply_lock()
        self.set_enabled(d > 0)

    def on_media_status(self, status):
        # seeking only sticks once the media is loaded
        if self._seek_after_load is not None and status in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia):
            self.player.setPosition(min(self._seek_after_load, self.duration))
            self._seek_after_load = None

    def on_position(self, pos: int):
        if not self.pos_slider.isSliderDown():
            self.pos_slider.setValue(pos)
        self.pos_label.setText(f"{fmt_time(pos)} / {fmt_time(self.duration)}")
        if self.player.playbackState() == QMediaPlayer.PlayingState and self._segment_mode:
            if pos >= self.end_slider.value():
                if self.loop_chk.isChecked():
                    self.player.setPosition(self.start_slider.value())
                else:
                    self.player.pause()

    _segment_mode = False

    def on_state(self, st):
        playing = st == QMediaPlayer.PlayingState
        self.play_btn.setText("⏸ Pause" if playing else "▶ Play")
        if not playing:
            self._segment_mode = False

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self._segment_mode = False
            self.player.play()

    def on_pos_moved(self, v: int):
        """Dragging the playhead moves the segment start along, keeping its length."""
        self.player.setPosition(v)
        length = self.locked_ms()
        if length is None:
            start, length = v, self._seg_len
        else:
            start = min(v, self.duration - length)
        self._set_quiet(self.start_slider, start)
        self._set_quiet(self.end_slider, min(start + length, self.duration))
        self.update_seg_labels()

    def play_segment(self):
        self.player.setPosition(self.start_slider.value())
        self.player.play()
        self._segment_mode = True

    # --- segment sliders ---------------------------------------------------
    _seg_len = 0  # segment length the start keeps while dragging the end

    def locked_ms(self) -> int | None:
        if not self.lock_chk.isChecked():
            return None
        return min(int(round(self.lock_spin.value() * 1000)), self.duration)

    def on_start_changed(self, v: int):
        lock = self.locked_ms()
        if lock is not None:
            if v + lock > self.duration:
                v = self.duration - lock
                self.start_slider.blockSignals(True)
                self.start_slider.setValue(v)
                self.start_slider.blockSignals(False)
            self._set_quiet(self.end_slider, v + lock)
        else:
            if v > self.end_slider.value():
                self._set_quiet(self.end_slider, v)
            self._seg_len = self.end_slider.value() - v
        if self.start_slider.isSliderDown() or self.player.playbackState() != QMediaPlayer.PlayingState:
            self.player.setPosition(v)
        self.update_seg_labels()

    def on_end_changed(self, v: int):
        lock = self.locked_ms()
        if lock is not None:
            if v - lock < 0:
                v = lock
                self.end_slider.blockSignals(True)
                self.end_slider.setValue(v)
                self.end_slider.blockSignals(False)
            self._set_quiet(self.start_slider, v - lock)
        else:
            # start follows the end, keeping the segment length
            self._set_quiet(self.start_slider, max(v - self._seg_len, 0))
        if self.end_slider.isSliderDown():
            self.player.setPosition(v)
        self.update_seg_labels()

    @staticmethod
    def _set_quiet(slider: QSlider, v: int):
        slider.blockSignals(True)
        slider.setValue(v)
        slider.blockSignals(False)

    def apply_lock(self, *_):
        lock = self.locked_ms()
        if lock is not None and self.duration:
            self.on_start_changed(self.start_slider.value())
        else:
            self._seg_len = self.end_slider.value() - self.start_slider.value()
        self.update_seg_labels()

    def update_seg_labels(self):
        s, e = self.start_slider.value(), self.end_slider.value()
        self.start_label.setText(fmt_time(s))
        self.end_label.setText(fmt_time(e))
        self.seg_label.setText(f"Duration: {(e - s) / 1000:.2f} s")

    # --- tag ------------------------------------------------------------------
    def tag_current(self):
        if not self.path:
            return
        if self.export_dir == self.path.parent and self.export_stem == self.path.stem:
            txt = self.path.with_suffix(".txt")
        else:
            # cut/combined temp file: put the text where the export goes by default
            txt = self.export_dir / f"{self.export_stem}_edit.txt"
        if txt.exists() and QMessageBox.question(
                self, "Tag current video", f"{txt.name} already exists. Overwrite it?") != QMessageBox.Yes:
            return
        self.player.pause()
        worker = self.tagger.tag_file(self.path, txt)
        if not worker:
            return
        self.tag_errors, self.tag_result = [], None
        dlg = QProgressDialog(f"Describing {self.export_stem}…", None, 0, 0, self)
        dlg.setWindowTitle("Tag current video")
        dlg.setMinimumDuration(0)
        dlg.show()
        worker.log.connect(lambda line: line.lstrip().startswith("✗") and self.tag_errors.append(line.strip("✗ ")))
        worker.described.connect(lambda path, text: setattr(self, "tag_result", (path, text)))
        worker.finished.connect(lambda: self.on_tag_finished(dlg))

    def on_tag_finished(self, dlg):
        dlg.close()
        if self.tag_result:
            path, text = self.tag_result
            QMessageBox.information(self, "Tag current video", f"Saved to {path}:\n\n{text}")
        else:
            QMessageBox.critical(self, "Tag current video",
                                 "\n".join(self.tag_errors) or "No description was written.")

    # --- cut ------------------------------------------------------------------
    def cut_selection(self):
        if not self.path or self.dl_proc or self.cut_proc:
            return
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            QMessageBox.critical(self, "Error", "ffmpeg was not found in PATH.")
            return
        s, e = self.start_slider.value(), self.end_slider.value()
        if e <= s:
            return
        if s <= 0 and e >= self.duration:
            QMessageBox.warning(self, "Cut", "The selection covers the whole video; nothing would be left.")
            return
        self.player.pause()
        try:
            has_audio = probe(str(self.path))["audio"]
        except Exception:
            has_audio = False
        parts = ([(0, s)] if s > 0 else []) + ([(e, None)] if e < self.duration else [])
        graph, concat_in = [], ""
        for n, (a, b) in enumerate(parts):
            rng = f"start={a / 1000:.3f}" + (f":end={b / 1000:.3f}" if b is not None else "")
            graph.append(f"[0:v:0]trim={rng},setpts=PTS-STARTPTS[v{n}]")
            concat_in += f"[v{n}]"
            if has_audio:
                graph.append(f"[0:a:0]atrim={rng},asetpts=PTS-STARTPTS[a{n}]")
                concat_in += f"[a{n}]"
        if len(parts) == 2:
            graph.append(f"{concat_in}concat=n=2:v=1:a={int(has_audio)}[v]" + ("[a]" if has_audio else ""))
            vmap, amap = "[v]", "[a]"
        else:
            vmap, amap = "[v0]", "[a0]"

        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="videoedit_cut_"))
        out = str(self.temp_dir / f"cut_{len(list(self.temp_dir.iterdir())):03d}.mp4")
        args = ["-hide_banner", "-y", "-i", str(self.path), "-filter_complex", ";".join(graph), "-map", vmap]
        args += ["-map", amap, "-c:a", "aac", "-b:a", "192k"] if has_audio else ["-an"]
        # intermediate file: quality high enough to edit further without visible loss
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                 "-progress", "pipe:1", "-nostats", out]

        self.cut_out, self.cut_at, self.cut_err = out, s, b""
        self.cut_total = max(1, self.duration - (e - s))
        self.cut_dialog = QProgressDialog("Cutting…", "Cancel", 0, 1000, self)
        self.cut_dialog.setWindowTitle("Cut")
        self.cut_dialog.setMinimumDuration(0)
        self.cut_dialog.setValue(0)
        proc = QProcess(self)
        proc.readyReadStandardOutput.connect(self.on_cut_output)
        proc.readyReadStandardError.connect(
            lambda: setattr(self, "cut_err", self.cut_err + proc.readAllStandardError().data()))
        proc.finished.connect(self.on_cut_finished)
        self.cut_dialog.canceled.connect(proc.kill)
        self.cut_proc = proc
        proc.start(ffmpeg, args)

    def on_cut_output(self):
        for line in self.cut_proc.readAllStandardOutput().data().decode(errors="ignore").splitlines():
            if line.startswith("out_time_us="):
                try:
                    self.cut_dialog.setValue(min(1000, int(int(line[12:]) / 1000 / self.cut_total * 1000)))
                except ValueError:
                    pass

    def on_cut_finished(self, code, status):
        self.cut_proc = None
        self.cut_dialog.close()
        if status == QProcess.NormalExit and code == 0:
            self.undo_stack.append((str(self.path), self.export_dir, self.export_stem))
            # show the frame right after the cut, where the two parts now join
            self._seek_after_load = self.cut_at
            self.load(self.cut_out, str(self.export_dir), self.export_stem, edit=True)
            return
        Path(self.cut_out).unlink(missing_ok=True)
        if status != QProcess.CrashExit:
            err = self.cut_err.decode(errors="ignore").strip().splitlines()[-15:]
            QMessageBox.critical(self, "Cut failed", "\n".join(err))

    def undo_cut(self):
        if not self.undo_stack or self.cut_proc:
            return
        path, export_dir, stem = self.undo_stack.pop()
        self.load(path, str(export_dir), stem, edit=True)

    # --- crop ----------------------------------------------------------------
    def on_crop_changed(self, r: QRectF):
        for key, val in zip("xywh", (r.x(), r.y(), r.width(), r.height())):
            self.crop_spins[key].blockSignals(True)
            self.crop_spins[key].setValue(int(val))
            self.crop_spins[key].blockSignals(False)

    def on_crop_spin(self):
        sp = self.crop_spins
        w, h = sp["w"].value(), sp["h"].value()
        a = self.view.aspect
        if a:
            # whichever of W/H was edited drives the other
            if w != int(self.view.crop.width()):
                h = w / a
            else:
                w = h * a
        self.view.set_crop(QRectF(sp["x"].value(), sp["y"].value(), w, h))

    def on_aspect_changed(self, *_):
        self.view.set_aspect(self.aspect_combo.currentData())

    # --- blur ----------------------------------------------------------------
    def blur_params(self):
        if not self.blur_chk.isChecked():
            return None
        return self.blur_type.currentData(), self.blur_strength.value(), self.blur_passes.value()

    def face_blur_settings(self):
        if not self.face_chk.isChecked():
            return None
        return FaceBlurSettings(self.face_mode.currentData(), self.face_strength.value(),
                                self.face_padding.value(), self.face_sens.value(), self.face_hold.value())

    def update_face_blur(self, *_):
        try:
            self.view.set_face_blur(self.face_blur_settings())
        except Exception as e:  # e.g. the model could not be downloaded
            self.face_chk.setChecked(False)
            QMessageBox.critical(self, "Face blur", f"Face detection is not available:\n{e}")

    def update_blur(self, *_):
        self.view.set_blur(self.blur_params())

    # --- export --------------------------------------------------------------
    def export(self):
        if not self.path:
            return
        if self.export_proc:
            self.export_proc.kill()
            return
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            QMessageBox.critical(self, "Error", "ffmpeg was not found in PATH.")
            return
        start, end = self.start_slider.value(), self.end_slider.value()
        if end <= start:
            QMessageBox.warning(self, "Export", "The segment has zero length.")
            return
        default = self.export_dir / f"{self.export_stem}_edit.mp4"
        out, _ = QFileDialog.getSaveFileName(self, "Export as", str(default),
                                             "MP4 (*.mp4);;MKV (*.mkv);;MOV (*.mov)")
        if not out:
            return
        self.player.pause()

        c = self.view.crop
        # libx264 with yuv420p requires even dimensions
        w, h = int(c.width()) // 2 * 2, int(c.height()) // 2 * 2
        x, y = int(c.x()), int(c.y())
        fps = self.fps_spin.value()
        filters = [f"crop={w}:{h}:{x}:{y}"]
        blur = self.blur_params()
        if blur:
            # blur before cropping, like the preview, so the crop edges stay natural
            filters.insert(0, blur_filter(*blur))
        encode = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
        encode += ["-c:a", "aac", "-b:a", "192k"] if self.audio_chk.isChecked() else ["-an"]
        ss, t = f"{start / 1000:.3f}", f"{(end - start) / 1000:.3f}"

        self.export_total_ms = end - start
        self.export_out = out
        self.export_err = b""
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.export_btn.setText("Cancel")

        face = self.face_blur_settings()
        if face:
            raw = self.view.raw
            worker = FaceExportWorker(self, ffmpeg, str(self.path), ss, t, fps, raw.width(), raw.height(),
                                      ",".join(filters), encode, out, face)
            worker.progress.connect(self.progress.setValue)
            worker.done.connect(self.on_face_export_done)
            worker.finished.connect(worker.deleteLater)
            self.export_proc = worker
            worker.start()
            return

        args = ["-hide_banner", "-y", "-ss", ss, "-i", str(self.path), "-t", t,
                "-vf", ",".join(filters + [f"fps={fps:g}"]), *encode,
                "-progress", "pipe:1", "-nostats", out]
        proc = QProcess(self)
        proc.readyReadStandardOutput.connect(self.on_export_output)
        proc.readyReadStandardError.connect(
            lambda: setattr(self, "export_err", self.export_err + proc.readAllStandardError().data()))
        proc.finished.connect(self.on_export_finished)
        self.export_proc = proc
        proc.start(ffmpeg, args)

    def on_export_output(self):
        data = self.export_proc.readAllStandardOutput().data().decode(errors="ignore")
        for line in data.splitlines():
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    us = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                self.progress.setValue(min(1000, int(us / 1000 / self.export_total_ms * 1000)))

    def on_export_finished(self, code, status):
        err = self.export_err.decode(errors="ignore")
        self.on_face_export_done(status == QProcess.NormalExit and code == 0, status == QProcess.CrashExit, err)

    def on_face_export_done(self, ok: bool, cancelled: bool, err: str):
        self.export_proc = None
        self.progress.setVisible(False)
        self.export_btn.setText("Export…")
        if ok:
            QMessageBox.information(self, "Export", f"Done:\n{self.export_out}")
        elif cancelled:
            QMessageBox.information(self, "Export", "Export cancelled.")
        else:
            QMessageBox.critical(self, "Export failed", "\n".join(err.strip().splitlines()[-15:]))


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Video Editor")
    app.setDesktopFileName("videoedit")  # links the window to videoedit.desktop in the dock
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win.load(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
