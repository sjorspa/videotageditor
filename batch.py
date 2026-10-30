"""Batch tab: resize every video in a folder, optionally without audio."""

import shutil
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from combine import VIDEO_EXTS, probe

# mode -> which size fields it uses
MODES = {
    "Fit within W×H": ("w", "h"),
    "Width": ("w",),
    "Height": ("h",),
    "Scale %": ("pct",),
    "Exact W×H (crop to fill)": ("w", "h"),
    "Exact W×H (pad)": ("w", "h"),
}


def scale_filter(mode: str, w: int, h: int, pct: int, upscale: bool) -> str:
    w, h = w // 2 * 2, h // 2 * 2  # H.264 in yuv420p needs even sizes
    if mode == "Fit within W×H":
        if upscale:
            return f"scale={w}:{h}:force_original_aspect_ratio=decrease:force_divisible_by=2"
        return f"scale='min({w},iw)':'min({h},ih)':force_original_aspect_ratio=decrease:force_divisible_by=2"
    if mode == "Width":
        return f"scale={w}:-2" if upscale else f"scale='min({w},iw)':-2"
    if mode == "Height":
        return f"scale=-2:{h}" if upscale else f"scale=-2:'min({h},ih)'"
    if mode == "Scale %":
        f = pct / 100 if upscale else min(pct, 100) / 100
        return f"scale=trunc(iw*{f}/2)*2:trunc(ih*{f}/2)*2"
    if mode == "Exact W×H (crop to fill)":
        return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"


class BatchWorker(QThread):
    log = Signal(str)
    progress = Signal(int)  # 0..1000 over all files

    def __init__(self, parent, jobs, vf, remove_audio, copy_txt):
        super().__init__(parent)
        self.jobs, self.vf, self.remove_audio, self.copy_txt = jobs, vf, remove_audio, copy_txt
        self.stopped = False
        self.proc = None

    def stop(self):
        self.stopped = True
        if self.proc:
            self.proc.kill()

    def run(self):
        ok = failed = 0
        for n, (src, dst) in enumerate(self.jobs):
            if self.stopped:
                break
            self.log.emit(f"▶ {src.name}")
            try:
                duration = probe(str(src))["duration"] or 1.0
            except Exception:
                duration = 1.0
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
            args = ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(src),
                    "-vf", f"{self.vf},setsar=1", "-map", "0:v:0",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
            args += ["-an"] if self.remove_audio else ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k"]
            args += ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(tmp)]
            self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in self.proc.stdout:
                if line.startswith("out_time_us="):
                    try:
                        frac = min(1.0, int(line[12:]) / 1e6 / duration)
                    except ValueError:
                        continue
                    self.progress.emit(int((n + frac) / len(self.jobs) * 1000))
            err = self.proc.stderr.read()
            code = self.proc.wait()
            self.proc = None
            if self.stopped:
                tmp.unlink(missing_ok=True)
                break
            if code != 0:
                tmp.unlink(missing_ok=True)
                self.log.emit("   ✗ " + (err.strip().splitlines() or [f"ffmpeg exited with code {code}"])[-1])
                failed += 1
                continue
            tmp.replace(dst)
            if self.copy_txt and (txt := src.with_suffix(".txt")).exists():
                shutil.copy2(txt, dst.with_suffix(".txt"))
            try:
                info = probe(str(dst))
                self.log.emit(f"   → {dst.name}  {info['width']}×{info['height']}"
                              + ("" if info["audio"] else " · no audio"))
            except Exception:
                self.log.emit(f"   → {dst.name}")
            ok += 1
        self.progress.emit(1000)
        self.log.emit(("Stopped. " if self.stopped else "Done: ") + f"{ok} written, {failed} failed.")


class BatchTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker: BatchWorker | None = None

        self.in_edit = QLineEdit()
        in_btn = QPushButton("Browse…")
        in_btn.clicked.connect(self.browse_in)
        in_row = QHBoxLayout()
        in_row.addWidget(self.in_edit, 1)
        in_row.addWidget(in_btn)
        self.recursive_chk = QCheckBox("Include subfolders (the folder structure is kept)")

        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("default: a 'resized' folder inside the video folder")
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self.browse_out)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(out_btn)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODES)
        self.mode_combo.currentIndexChanged.connect(self.update_fields)
        self.w_spin = QSpinBox()
        self.w_spin.setRange(16, 8192)
        self.w_spin.setValue(1280)
        self.w_spin.setSuffix(" px")
        self.h_spin = QSpinBox()
        self.h_spin.setRange(16, 8192)
        self.h_spin.setValue(720)
        self.h_spin.setSuffix(" px")
        self.pct_spin = QSpinBox()
        self.pct_spin.setRange(1, 400)
        self.pct_spin.setValue(50)
        self.pct_spin.setSuffix(" %")
        self.w_label, self.h_label = QLabel("W"), QLabel("H")
        size_row = QHBoxLayout()
        for wdg in (self.mode_combo, self.w_label, self.w_spin, self.h_label, self.h_spin, self.pct_spin):
            size_row.addWidget(wdg)
        size_row.addStretch()

        self.upscale_chk = QCheckBox("Allow upscaling (otherwise smaller videos keep their size)")
        self.audio_chk = QCheckBox("Remove audio")
        self.txt_chk = QCheckBox("Copy matching .txt files (captions) along")
        self.txt_chk.setChecked(True)
        self.overwrite_chk = QCheckBox("Overwrite existing output files")

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        run_row = QHBoxLayout()
        run_row.addWidget(self.progress, 1)
        run_row.addWidget(self.start_btn)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)

        form = QFormLayout()
        form.addRow("Video folder", in_row)
        form.addRow("", self.recursive_chk)
        form.addRow("Output folder", out_row)
        form.addRow("Size", size_row)
        form.addRow("", self.upscale_chk)
        form.addRow("", self.audio_chk)
        form.addRow("", self.txt_chk)
        form.addRow("", self.overwrite_chk)
        v = QVBoxLayout(self)
        v.addLayout(form)
        v.addLayout(run_row)
        v.addWidget(self.log_view, 1)
        self.update_fields()

    def persist_widgets(self) -> dict:
        return {"batch/in": self.in_edit, "batch/out": self.out_edit, "batch/recursive": self.recursive_chk,
                "batch/mode": self.mode_combo, "batch/w": self.w_spin, "batch/h": self.h_spin,
                "batch/pct": self.pct_spin, "batch/upscale": self.upscale_chk, "batch/no_audio": self.audio_chk,
                "batch/copy_txt": self.txt_chk, "batch/overwrite": self.overwrite_chk}

    def update_fields(self, *_):
        used = MODES[self.mode_combo.currentText()]
        for wdg, key in ((self.w_label, "w"), (self.w_spin, "w"), (self.h_label, "h"), (self.h_spin, "h"),
                         (self.pct_spin, "pct")):
            wdg.setVisible(key in used)
        # exact sizes always scale to fit, up or down
        exact = self.mode_combo.currentText().startswith("Exact")
        self.upscale_chk.setEnabled(not exact)
        self.upscale_chk.setToolTip("Exact sizes always scale up when needed" if exact else "")

    def browse_in(self):
        d = QFileDialog.getExistingDirectory(self, "Video folder", self.in_edit.text() or str(Path.home()))
        if d:
            self.in_edit.setText(d)

    def browse_out(self):
        start = self.out_edit.text() or self.in_edit.text() or str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "Output folder", start)
        if d:
            self.out_edit.setText(d)

    def start(self):
        if self.worker:
            self.worker.stop()
            self.start_btn.setEnabled(False)
            return
        src_dir = Path(self.in_edit.text()).expanduser()
        if not src_dir.is_dir():
            QMessageBox.warning(self, "Batch", "Choose an existing video folder.")
            return
        out_dir = Path(self.out_edit.text()).expanduser() if self.out_edit.text().strip() else src_dir / "resized"
        if out_dir.resolve() == src_dir.resolve():
            QMessageBox.warning(self, "Batch", "The output folder must differ from the video folder, "
                                "so the originals are never overwritten.")
            return
        files = src_dir.rglob("*") if self.recursive_chk.isChecked() else src_dir.iterdir()
        videos = sorted(p for p in files if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                        and out_dir.resolve() not in p.resolve().parents)  # skip earlier output
        jobs = [(p, out_dir / p.relative_to(src_dir).with_suffix(".mp4")) for p in videos]
        skipped = 0
        if not self.overwrite_chk.isChecked():
            skipped = sum(1 for _, d in jobs if d.exists())
            jobs = [(s, d) for s, d in jobs if not d.exists()]
        if not jobs:
            QMessageBox.information(self, "Batch", "No videos to process"
                                    + (f" ({skipped} already done)." if skipped else "."))
            return
        vf = scale_filter(self.mode_combo.currentText(), self.w_spin.value(), self.h_spin.value(),
                          self.pct_spin.value(), self.upscale_chk.isChecked())
        self.log_view.clear()
        self.log_view.appendPlainText(f"{len(jobs)} videos → {out_dir}"
                                      + (f" ({skipped} already done, skipped)" if skipped else ""))
        w = BatchWorker(self, jobs, vf, self.audio_chk.isChecked(), self.txt_chk.isChecked())
        w.log.connect(self.log_view.appendPlainText)
        w.progress.connect(self.progress.setValue)
        w.finished.connect(self.on_finished)
        self.worker = w
        self.progress.setValue(0)
        self.start_btn.setText("Stop")
        w.start()

    def on_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start")

    def shutdown(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(5000)
