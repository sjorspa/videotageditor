"""Combine tab: pick several videos, order them by drag and drop, join them with ffmpeg."""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}


def probe(path: str) -> dict:
    """Duration, display size, frame rate and whether there is audio."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height,r_frame_rate:stream_side_data=rotation",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("no video stream")
    w, h = video["width"], video["height"]
    rotation = next((sd.get("rotation", 0) for sd in video.get("side_data_list", [])), 0)
    if abs(int(rotation)) % 180 == 90:
        w, h = h, w
    num, _, den = video.get("r_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    return {
        "path": path,
        "duration": float(data.get("format", {}).get("duration", 0) or 0),
        "width": w, "height": h,
        "fps": fps if 1 <= fps <= 240 else 30.0,
        "audio": any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
    }


def fmt_dur(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:04.1f}"


class FileList(QListWidget):
    """List that reorders by dragging and also accepts files dropped from outside."""

    filesDropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.source() is not self and e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.source() is not self and e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.source() is not self and e.mimeData().hasUrls():
            self.filesDropped.emit([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])
            e.acceptProposedAction()
        else:
            super().dropEvent(e)


class CombineTab(QWidget):
    combined = Signal(str, str)  # path of the joined video, folder to suggest for export

    def __init__(self):
        super().__init__()
        self.proc: QProcess | None = None
        self.last_dir = Path.home()
        self.temp_dir: Path | None = None  # per instance, removed on exit

        self.list = FileList()
        self.list.filesDropped.connect(self.add_files)
        self.list.model().rowsInserted.connect(self.update_summary)
        self.list.model().rowsRemoved.connect(self.update_summary)

        add_btn = QPushButton("Add videos…")
        add_btn.clicked.connect(self.browse)
        remove_btn = QPushButton("Remove")
        remove_btn.setShortcut("Delete")
        remove_btn.clicked.connect(self.remove_selected)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.list.clear)
        self.done_btn = QPushButton("Done – open in editor")
        self.done_btn.clicked.connect(self.done)

        self.summary = QLabel()
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        buttons = QHBoxLayout()
        buttons.addWidget(add_btn)
        buttons.addWidget(remove_btn)
        buttons.addWidget(clear_btn)
        buttons.addStretch()
        bottom = QHBoxLayout()
        bottom.addWidget(self.summary)
        bottom.addWidget(self.progress, 1)
        bottom.addWidget(self.done_btn)

        v = QVBoxLayout(self)
        v.addWidget(QLabel("Add videos, then drag them into the right order. "
                           "Files can also be dropped here from your file manager."))
        v.addLayout(buttons)
        v.addWidget(self.list, 1)
        v.addLayout(bottom)
        self.update_summary()

    # --- list ------------------------------------------------------------------
    def browse(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add videos", str(self.last_dir),
            "Video (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.mpg *.mpeg *.wmv);;All files (*)")
        self.add_files(files)

    def add_files(self, files):
        failed = []
        for f in files:
            p = Path(f)
            if p.is_dir():
                self.add_files(sorted(str(c) for c in p.iterdir() if c.suffix.lower() in VIDEO_EXTS))
                continue
            try:
                info = probe(str(p))
            except (subprocess.CalledProcessError, ValueError, KeyError, json.JSONDecodeError):
                failed.append(p.name)
                continue
            self.last_dir = p.parent
            audio = "" if info["audio"] else " · no audio"
            item = QListWidgetItem(f"{p.name}   —   {fmt_dur(info['duration'])} · "
                                   f"{info['width']}×{info['height']} · {info['fps']:.3g} fps{audio}")
            item.setData(Qt.UserRole, info)
            item.setToolTip(str(p))
            self.list.addItem(item)
        if failed:
            QMessageBox.warning(self, "Combine", "Not a readable video:\n" + "\n".join(failed))

    def remove_selected(self):
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def infos(self) -> list[dict]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]

    def update_summary(self, *_):
        infos = self.infos()
        if infos:
            first = infos[0]
            self.summary.setText(f"{len(infos)} videos · {fmt_dur(sum(i['duration'] for i in infos))} · "
                                 f"output {first['width']}×{first['height']} @ {first['fps']:.3g} fps")
        else:
            self.summary.setText("No videos yet")
        self.done_btn.setEnabled(bool(infos) and self.proc is None)

    # --- combine -------------------------------------------------------------------
    def done(self):
        infos = self.infos()
        self.export_dir = str(Path(infos[0]["path"]).parent)
        if len(infos) == 1:
            self.combined.emit(infos[0]["path"], self.export_dir)
            return
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            QMessageBox.critical(self, "Error", "ffmpeg was not found in PATH.")
            return

        # everything is scaled/padded to the first video's size and frame rate
        first = infos[0]
        w, h, fps = first["width"] // 2 * 2, first["height"] // 2 * 2, first["fps"]
        any_audio = any(i["audio"] for i in infos)
        args = ["-hide_banner", "-y"]
        for i in infos:
            args += ["-i", i["path"]]
        graph, concat_in = [], ""
        silent = len(infos)  # input index of the next silence source
        for n, i in enumerate(infos):
            graph.append(f"[{n}:v:0]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                         f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps:.6g},format=yuv420p[v{n}]")
            concat_in += f"[v{n}]"
            if any_audio:
                if i["audio"]:
                    src = f"{n}:a:0"
                else:
                    args += ["-f", "lavfi", "-t", f"{i['duration']:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
                    src = f"{silent}:a:0"
                    silent += 1
                graph.append(f"[{src}]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{n}]")
                concat_in += f"[a{n}]"
        graph.append(f"{concat_in}concat=n={len(infos)}:v=1:a={int(any_audio)}"
                     + ("[v][a]" if any_audio else "[v]"))
        args += ["-filter_complex", ";".join(graph), "-map", "[v]"]
        args += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"] if any_audio else ["-an"]
        # intermediate file: quality high enough to edit further without visible loss
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="videoedit_"))
        self.out = str(self.temp_dir / f"combined_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                 "-progress", "pipe:1", "-nostats", self.out]

        self.total = sum(i["duration"] for i in infos) or 1
        self.err = b""
        proc = QProcess(self)
        proc.readyReadStandardOutput.connect(self.on_output)
        proc.readyReadStandardError.connect(lambda: setattr(self, "err", self.err + proc.readAllStandardError().data()))
        proc.finished.connect(self.on_finished)
        self.proc = proc
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.done_btn.setText("Cancel")
        self.done_btn.clicked.disconnect()
        self.done_btn.clicked.connect(proc.kill)
        proc.start(ffmpeg, args)

    def on_output(self):
        for line in self.proc.readAllStandardOutput().data().decode(errors="ignore").splitlines():
            if line.startswith("out_time_us="):
                try:
                    self.progress.setValue(min(1000, int(int(line[12:]) / 1e6 / self.total * 1000)))
                except ValueError:
                    pass

    def on_finished(self, code, status):
        self.proc = None
        self.progress.setVisible(False)
        self.done_btn.setText("Done – open in editor")
        self.done_btn.clicked.disconnect()
        self.done_btn.clicked.connect(self.done)
        self.update_summary()
        if status == QProcess.NormalExit and code == 0:
            self.combined.emit(self.out, self.export_dir)
            return
        Path(self.out).unlink(missing_ok=True)
        if status != QProcess.CrashExit:
            err = self.err.decode(errors="ignore").strip().splitlines()[-15:]
            QMessageBox.critical(self, "Combine failed", "\n".join(err))

    def cleanup(self):
        """Remove intermediate files (on exit)."""
        if self.proc:
            self.proc.kill()
            self.proc.waitForFinished(3000)
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
