"""Tagger tabs: describe every video (or image) in a folder with a vision model in
LM Studio, writing the description to a .txt file with the same name next to it."""

import base64
import io
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps
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

DEFAULT_SERVER = "http://localhost:1234"
DEFAULT_PROMPT = (
    "These are frames sampled in chronological order from one video clip. "
    "Describe the video as one paragraph: the subjects and their appearance, "
    "their actions and motion, the setting, the lighting and the camera movement. "
    "Describe it as one continuous video, not as separate images. "
    "Output only the description."
)
# the first default asked for "one detailed paragraph", which fights a short word limit
OLD_DEFAULT_PROMPT = DEFAULT_PROMPT.replace("as one paragraph", "in one detailed paragraph")
DEFAULT_IMAGE_PROMPT = (
    "Describe this image as one paragraph: the subjects and their appearance, what they are doing, "
    "the setting, the composition, the lighting and the style of the image. "
    "Output only the description."
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# per tab: settings prefix, what the files are called, which files, default prompt
KINDS = {
    "video": {"key": "tagger", "noun": "video", "exts": VIDEO_EXTS, "prompt": DEFAULT_PROMPT},
    "image": {"key": "imgtagger", "noun": "image", "exts": IMAGE_EXTS, "prompt": DEFAULT_IMAGE_PROMPT},
}


def lms_path() -> str | None:
    return shutil.which("lms") or next(
        (str(p) for p in [Path.home() / ".lmstudio" / "bin" / "lms"] if p.exists()), None)


def list_models(server: str) -> list[dict]:
    """Vision-capable models known to LM Studio: [{key, name, loaded, instances}]."""
    try:
        with urllib.request.urlopen(f"{server}/api/v1/models", timeout=5) as r:
            data = json.load(r)
        return [{"key": m["key"], "name": m.get("display_name") or m["key"],
                 "loaded": bool(m.get("loaded_instances")),
                 "instances": [i["id"] for i in m.get("loaded_instances") or []],
                 "contexts": [(i.get("config") or {}).get("context_length") for i in m.get("loaded_instances") or []],
                 "max_context": m.get("max_context_length")}
                for m in data.get("models", [])
                if m.get("type") == "llm" and (m.get("capabilities") or {}).get("vision")]
    except urllib.error.HTTPError:
        # older LM Studio: beta REST API
        with urllib.request.urlopen(f"{server}/api/v0/models", timeout=5) as r:
            data = json.load(r)
        return [{"key": m["id"], "name": m["id"], "loaded": m.get("state") == "loaded", "instances": None,
                 "contexts": [], "max_context": m.get("max_context_length")}
                for m in data.get("data", []) if m.get("type") == "vlm"]


def unload_model(server: str, key: str) -> bool:
    """Unload every loaded instance of `key`; False if it was not loaded."""
    model = next((m for m in list_models(server) if m["key"] == key), None)
    if not model or not model["loaded"]:
        return False
    if model["instances"] is None:
        # older LM Studio without the v1 REST API: use the CLI
        lms = lms_path()
        if not lms:
            raise RuntimeError("cannot unload: LM Studio's `lms` command was not found")
        subprocess.run([lms, "unload", key], capture_output=True, check=True, timeout=60)
        return True
    for instance in model["instances"]:
        req = urllib.request.Request(f"{server}/api/v1/models/unload", json.dumps({"instance_id": instance}).encode(),
                                     {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=60).close()
        except urllib.error.HTTPError as e:
            raise RuntimeError(e.read().decode(errors="ignore")[:300]) from e
    return True


def load_model(server: str, key: str, context: int = 0):
    """Load `key`; context 0 uses LM Studio's default."""
    body = {"model": key, **({"context_length": context} if context else {})}
    req = urllib.request.Request(f"{server}/api/v1/models/load", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=900).close()
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode(errors="ignore")[:300]) from e


def ensure_loaded(server: str, key: str, context: int) -> str:
    """Make sure `key` is loaded with exactly `context` tokens (0 = leave it to LM Studio).

    Returns what was done, for the log."""
    if not context:
        return ""
    model = next((m for m in list_models(server) if m["key"] == key), None)
    if model is None:
        raise RuntimeError(f"{key} is not available in LM Studio")
    if model["instances"] is None:
        raise RuntimeError("setting the context needs a newer LM Studio (v1 REST API); set Context to Auto")
    if model["max_context"] and context > model["max_context"]:
        raise RuntimeError(f"{key} supports at most {model['max_context']} tokens of context")
    if model["loaded"] and all(c == context for c in model["contexts"]):
        return f"{key} is already loaded with context {context}."
    msg = f"Loaded {key} with context {context}."
    if model["loaded"]:
        unload_model(server, key)
        msg = f"Reloaded {key} with context {context} (was {', '.join(map(str, model['contexts']))})."
    load_model(server, key, context)
    return msg


def model_key_for_file(path: str) -> str | None:
    """Map a .gguf file to its LM Studio model key (via `lms ls --json`)."""
    lms = lms_path()
    if not lms:
        return None
    out = subprocess.run([lms, "ls", "--json"], capture_output=True, text=True, timeout=30).stdout
    p = Path(path).resolve().as_posix()
    for m in json.loads(out or "[]"):
        rel = m.get("path", "")
        # the model file itself, or its vision projector (mmproj) next to it
        if rel and (p.endswith("/" + rel) or Path(p).parent.as_posix().endswith("/" + str(Path(rel).parent))):
            return m.get("modelKey")
    return None


def extract_frames(video: str, count: int, max_side: int) -> list[bytes]:
    """`count` JPEG frames spread evenly over the video."""
    duration = probe(video)["duration"] or 1.0
    frames = []
    for i in range(count):
        t = duration * (i + 0.5) / count
        jpg = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
             "-vf", f"scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease",
             "-f", "image2", "-c:v", "mjpeg", "-q:v", "3", "-"],
            capture_output=True, check=True).stdout
        if jpg:
            frames.append(jpg)
    if not frames:
        raise ValueError("no frames could be read")
    return frames


def chat(server: str, model: str, content, thinking: bool, max_tokens: int) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "temperature": 0.3, "max_tokens": max_tokens + (3000 if thinking else 0)}
    if not thinking:
        body["reasoning_effort"] = "none"
    req = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode(errors="ignore")[:500]) from e
    text = data["choices"][0]["message"].get("content") or ""
    # some models put their reasoning inline
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return " ".join(text.split())


def image_frame(path: str, max_side: int) -> list[bytes]:
    """The image as one JPEG, upright and no larger than `max_side`."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, "white")  # transparent areas on white, not black
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return [buf.getvalue()]


def describe(server: str, model: str, prompt: str, frames: list[bytes], thinking: bool, max_words: int) -> str:
    # models only roughly follow a word count; a range works best
    content = [{"type": "text", "text": f"{prompt}\n\nLength requirement: {max(1, round(max_words * 0.75))}-{max_words} words."}]
    content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(f).decode()}}
                for f in frames]
    # generous token budget: the word limit is in the prompt, this only prevents runaways
    text = chat(server, model, content, thinking, max_words * 3 + 200)
    if not text:
        raise RuntimeError("the model returned an empty description")
    if len(text.split()) > max_words:
        # too long: let the model shorten its own text (cheap, no images), trim_words() is the last resort
        target = max(3, round(max_words * 0.8))  # aim lower: the model tends to overshoot
        shorter = chat(server, model, f"Shorten this description to at most {target} words, keeping "
                       f"the most important details and complete sentences. Output only the shortened "
                       f"description.\n\n{text}", thinking, max_words * 3 + 100)
        text = shorter or text
    return trim_words(text, max_words)


DANGLING = {"a", "an", "the", "at", "in", "on", "of", "to", "with", "and", "or", "as", "by", "for", "from",
            "into", "near", "while", "his", "her", "their", "its", "is", "are", "who", "which", "that"}


def trim_words(text: str, max_words: int) -> str:
    """Cut to at most `max_words` words, preferably at the end of a sentence."""
    words = text.split()
    if len(words) <= max_words:
        return text
    cut = " ".join(words[:max_words])
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind("."))
    if end > 0:  # a complete sentence beats a broken one
        return cut[:end + 1]
    # mid-sentence: don't end on a dangling "at the", "with a", ...
    words = cut.rstrip(",;:").split()
    while len(words) > 1 and words[-1].lower().strip(",;:") in DANGLING:
        words.pop()
    return " ".join(words).rstrip(",;:") + "."


class TagWorker(QThread):
    log = Signal(str)
    progress = Signal(int, int)  # done, total
    described = Signal(str, str)  # .txt path, description

    def __init__(self, parent, server, model, prompt, prefix, videos, frames, max_side, thinking, max_words,
                 context, targets=None, kind="video"):
        super().__init__(parent)
        self.max_words, self.context, self.kind = max_words, context, kind
        self.targets = targets or {}  # file -> .txt path, default: next to the file
        self.server, self.model, self.prompt, self.prefix = server, model, prompt, prefix
        self.videos, self.frames, self.max_side, self.thinking = videos, frames, max_side, thinking
        self.stopped = False

    def run(self):
        ok = failed = 0
        try:
            if msg := ensure_loaded(self.server, self.model, self.context):
                self.log.emit(msg)
        except Exception as e:
            self.log.emit(f"✗ Could not load the model: {e}")
            return
        for n, video in enumerate(self.videos):
            if self.stopped:
                self.log.emit("Stopped.")
                break
            self.progress.emit(n, len(self.videos))
            self.log.emit(f"▶ {video.name}")
            try:
                frames = (extract_frames(str(video), self.frames, self.max_side) if self.kind == "video"
                          else image_frame(str(video), self.max_side))
                text = describe(self.server, self.model, self.prompt, frames, self.thinking, self.max_words)
                if self.prefix:
                    text = f"{self.prefix} {text}"
                txt = self.targets.get(video, video.with_suffix(".txt"))
                tmp = txt.with_suffix(".txt.part")
                tmp.write_text(text + "\n", encoding="utf-8")
                tmp.replace(txt)
                self.log.emit(f"   {text}")
                self.described.emit(str(txt), text)
                ok += 1
            except Exception as e:
                self.log.emit(f"   ✗ {e}")
                failed += 1
        self.progress.emit(len(self.videos), len(self.videos))
        self.log.emit(f"Done: {ok} written, {failed} failed.")


class TaggerTab(QWidget):
    def __init__(self, kind: str = "video"):
        super().__init__()
        self.kind = kind
        self.cfg = KINDS[kind]
        self.worker: TagWorker | None = None
        self.preferred_model = ""  # restored from settings, selected once the list arrives

        self.server_edit = QLineEdit(DEFAULT_SERVER)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(320)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_models)
        browse_model_btn = QPushButton("Choose file…")
        browse_model_btn.setToolTip("Pick a .gguf model file from your LM Studio models folder")
        browse_model_btn.clicked.connect(self.browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(refresh_btn)
        model_row.addWidget(browse_model_btn)
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(0, 262144)
        self.ctx_spin.setSingleStep(1024)
        self.ctx_spin.setValue(40960)
        self.ctx_spin.setSpecialValueText("Auto")
        self.ctx_spin.setSuffix(" tokens")
        self.ctx_spin.setToolTip("Context length the model is loaded with. Auto leaves it to LM Studio.\n"
                                 "Smaller uses less GPU memory; 40K leaves plenty of room for many large frames.")
        model_row.addWidget(QLabel("Context"))
        model_row.addWidget(self.ctx_spin)
        load_btn = QPushButton("Load model")
        load_btn.setToolTip("Load the selected model in LM Studio with this context length")
        load_btn.clicked.connect(self.load)
        model_row.addWidget(load_btn)
        unload_btn = QPushButton("Unload model")
        unload_btn.setToolTip("Free the GPU memory used by the selected model in LM Studio")
        unload_btn.clicked.connect(self.unload)
        model_row.addWidget(unload_btn)

        self.folder_edit = QLineEdit()
        browse_folder_btn = QPushButton("Browse…")
        browse_folder_btn.clicked.connect(self.browse_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(browse_folder_btn)
        self.recursive_chk = QCheckBox("Include subfolders")
        self.overwrite_chk = QCheckBox("Overwrite existing .txt files")

        self.prompt_edit = QPlainTextEdit(self.cfg["prompt"])
        self.prompt_edit.setFixedHeight(90)
        reset_prompt_btn = QPushButton("Reset to default prompt")
        reset_prompt_btn.clicked.connect(lambda: self.prompt_edit.setPlainText(self.cfg["prompt"]))
        prompt_box = QVBoxLayout()
        prompt_box.addWidget(self.prompt_edit)
        reset_row = QHBoxLayout()
        reset_row.addStretch()
        reset_row.addWidget(reset_prompt_btn)
        prompt_box.addLayout(reset_row)
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("optional, e.g. a trigger word – put in front of every description")
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(1, 32)
        self.frames_spin.setValue(6)
        self.frames_spin.setToolTip("Frames sent to the model per video, spread evenly over the clip")
        self.size_spin = QSpinBox()
        self.size_spin.setRange(128, 2048)
        self.size_spin.setSingleStep(64)
        self.size_spin.setValue(768)
        self.size_spin.setSuffix(" px")
        self.size_spin.setToolTip(f"Longest side of each {'frame' if kind == 'video' else 'image'} sent to the model")
        self.words_spin = QSpinBox()
        self.words_spin.setRange(5, 1000)
        self.words_spin.setSingleStep(10)
        self.words_spin.setValue(120)
        self.words_spin.setToolTip("Maximum length of each description. The model is asked for 75–100% of this; "
                                   "longer output is trimmed to whole sentences")
        self.thinking_chk = QCheckBox("Thinking")
        self.thinking_chk.setToolTip("Let reasoning models think first: slower, sometimes more accurate")
        opts = QHBoxLayout()
        if kind == "video":
            opts.addWidget(QLabel("Frames"))
            opts.addWidget(self.frames_spin)
        opts.addWidget(QLabel("Frame size" if kind == "video" else "Image size"))
        opts.addWidget(self.size_spin)
        opts.addWidget(QLabel("Max words"))
        opts.addWidget(self.words_spin)
        opts.addWidget(self.thinking_chk)
        opts.addStretch()

        self.start_btn = QPushButton("Start tagging")
        self.start_btn.clicked.connect(self.start)
        self.progress = QProgressBar()
        self.status = QLabel()
        run_row = QHBoxLayout()
        run_row.addWidget(self.progress, 1)
        run_row.addWidget(self.start_btn)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)

        form = QFormLayout()
        form.addRow("LM Studio server", self.server_edit)
        form.addRow("Model", model_row)
        form.addRow(f"{self.cfg['noun'].capitalize()} folder", folder_row)
        form.addRow("", self.recursive_chk)
        form.addRow("", self.overwrite_chk)
        form.addRow("Prompt", prompt_box)
        form.addRow("Prefix", self.prefix_edit)
        form.addRow("", opts)
        v = QVBoxLayout(self)
        v.addLayout(form)
        v.addLayout(run_row)
        v.addWidget(self.status)
        v.addWidget(self.log_view, 1)

    def persist_widgets(self) -> dict:
        k = self.cfg["key"]
        widgets = {"server": self.server_edit, "folder": self.folder_edit, "recursive": self.recursive_chk,
                   "overwrite": self.overwrite_chk, "prompt": self.prompt_edit, "prefix": self.prefix_edit,
                   "context": self.ctx_spin, "size": self.size_spin, "words": self.words_spin,
                   "thinking": self.thinking_chk}
        if self.kind == "video":
            widgets["frames"] = self.frames_spin
        return {f"{k}/{name}": w for name, w in widgets.items()}

    def after_restore(self):
        if self.kind == "video" and self.prompt_edit.toPlainText().strip() == OLD_DEFAULT_PROMPT:
            self.prompt_edit.setPlainText(DEFAULT_PROMPT)

    def server(self) -> str:
        return self.server_edit.text().strip().rstrip("/") or DEFAULT_SERVER

    def current_model(self) -> str:
        return self.model_combo.currentData() or self.preferred_model

    # --- model ------------------------------------------------------------------
    def refresh_models(self):
        keep = self.current_model()
        self.model_combo.clear()
        try:
            models = list_models(self.server())
        except Exception as e:
            self.status.setText(f"LM Studio not reachable at {self.server()} ({e}). Is the server running?")
            if keep:
                self.model_combo.addItem(f"{keep} (not verified)", keep)
            return
        for m in models:
            self.model_combo.addItem(m["name"] + ("  ● loaded" if m["loaded"] else ""), m["key"])
        if keep and self.model_combo.findData(keep) < 0:
            self.model_combo.addItem(f"{keep} (not found in LM Studio)", keep)
        if keep:
            self.model_combo.setCurrentIndex(self.model_combo.findData(keep))
        self.status.setText(f"{len(models)} vision models found in LM Studio." if models else
                            "LM Studio has no vision-capable models.")

    def browse_model(self):
        start = Path.home() / ".lmstudio" / "models"
        fn, _ = QFileDialog.getOpenFileName(self, "Choose model", str(start if start.is_dir() else Path.home()),
                                            "GGUF model (*.gguf);;All files (*)")
        if not fn:
            return
        try:
            key = model_key_for_file(fn)
        except Exception:
            key = None
        if not key:
            QMessageBox.warning(self, "Model", "This file is not a model known to LM Studio.\n"
                                "Put it in LM Studio's models folder (or use `lms import`) and try again.")
            return
        self.preferred_model = key
        idx = self.model_combo.findData(key)
        if idx < 0:
            self.model_combo.addItem(key, key)
            idx = self.model_combo.count() - 1
        self.model_combo.setCurrentIndex(idx)

    def load(self):
        key = self.current_model()
        if not key or self.worker:
            return
        self.status.setText(f"Loading {key}…")
        self.repaint()
        try:
            if self.ctx_spin.value():
                msg = ensure_loaded(self.server(), key, self.ctx_spin.value())
            elif next((m for m in list_models(self.server()) if m["key"] == key), {}).get("loaded"):
                msg = f"{key} is already loaded."
            else:
                load_model(self.server(), key)
                msg = f"Loaded {key} with LM Studio's default context."
        except Exception as e:
            self.status.setText(f"Could not load {key}: {e}")
            return
        self.refresh_models()
        self.status.setText(msg)

    def unload(self):
        key = self.current_model()
        if not key:
            return
        if self.worker:
            QMessageBox.warning(self, "Tagger", "Stop tagging before unloading the model.")
            return
        try:
            done = unload_model(self.server(), key)
        except Exception as e:
            self.status.setText(f"Could not unload {key}: {e}")
            return
        self.refresh_models()
        self.status.setText(f"Unloaded {key}." if done else f"{key} was not loaded.")

    # --- run ----------------------------------------------------------------------
    def browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, f"{self.cfg['noun'].capitalize()} folder",
                                             self.folder_edit.text() or str(Path.home()))
        if d:
            self.folder_edit.setText(d)

    def files(self) -> list[Path]:
        folder = Path(self.folder_edit.text()).expanduser()
        found = folder.rglob("*") if self.recursive_chk.isChecked() else folder.iterdir()
        files = sorted(p for p in found if p.is_file() and p.suffix.lower() in self.cfg["exts"])
        if not self.overwrite_chk.isChecked():
            files = [p for p in files if not p.with_suffix(".txt").exists()]
        return files

    def start(self):
        if self.worker:
            self.worker.stopped = True
            self.start_btn.setEnabled(False)
            self.start_btn.setText(f"Stopping after this {self.cfg['noun']}…")
            return
        model = self.current_model()
        if not model:
            QMessageBox.warning(self, "Tagger", "Choose a model first (is LM Studio running? Click Refresh).")
            return
        if not Path(self.folder_edit.text()).expanduser().is_dir():
            QMessageBox.warning(self, "Tagger", f"Choose an existing {self.cfg['noun']} folder.")
            return
        files = self.files()
        if not files:
            QMessageBox.information(self, "Tagger", f"No {self.cfg['noun']}s to tag "
                                    "(all of them may already have a .txt file).")
            return
        self.start_worker(files)

    def tag_file(self, video: Path, txt: Path) -> TagWorker | None:
        """Tag one video (from the editor) with the current settings, writing `txt`."""
        if self.worker:
            QMessageBox.warning(self, "Tagger", "The tagger is already running.")
            return None
        if not self.current_model():
            self.refresh_models()
        if not self.current_model():
            QMessageBox.warning(self, "Tagger", "Choose a model in the Video Tagger tab first (is LM Studio running?).")
            return None
        return self.start_worker([video], {video: txt})

    def start_worker(self, videos: list[Path], targets=None) -> TagWorker:
        model = self.current_model()
        self.preferred_model = model
        self.log_view.clear()
        noun = self.cfg["noun"] + ("s" if len(videos) > 1 else "")
        self.log_view.appendPlainText(f"{len(videos)} {noun} · model {model}"
                                      " (LM Studio loads it on the first request if needed)")
        w = TagWorker(self, self.server(), model, self.prompt_edit.toPlainText().strip(),
                      self.prefix_edit.text().strip(), videos, self.frames_spin.value(),
                      self.size_spin.value(), self.thinking_chk.isChecked(), self.words_spin.value(),
                      self.ctx_spin.value(), targets, self.kind)
        w.log.connect(self.log_view.appendPlainText)
        w.progress.connect(lambda d, t: (self.progress.setRange(0, t), self.progress.setValue(d),
                                         self.status.setText(f"{d} / {t}")))
        w.finished.connect(self.on_finished)
        self.worker = w
        self.start_btn.setText("Stop")
        w.start()
        return w

    def on_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start tagging")

    def shutdown(self):
        if self.worker:
            self.worker.stopped = True
            # a request can take minutes; don't keep the app from closing
            if not self.worker.wait(2000):
                self.worker.terminate()
                self.worker.wait()
