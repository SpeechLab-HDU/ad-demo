"""
app.py
======
Flask web server that serves the AD recognition UI and handles API requests.

Architecture:
    Browser (HTML/JS)  <-->  Flask (app.py)  <-->  AD Predictor (predictor.py)
         UI                  Web Server                AI Model (Wav2Vec2 + SVM)
"""

from flask import Flask, request, jsonify, render_template_string
import os
import tempfile
import subprocess
from predictor import load_model, classify_audio, get_model_info, ModelNotLoadedError

# ── App Setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Load model once at startup. If weights are missing/corrupt, we still start the
# server so the UI loads; /predict returns a clear 503 explaining the problem.
print("[App] Loading AD recognition model at startup...")
MODEL = None
MODEL_LOAD_ERROR = None
try:
    MODEL = load_model("base")
    print("[App] Ready to serve requests.")
except ModelNotLoadedError as e:
    MODEL_LOAD_ERROR = str(e)
    print(f"[App] WARNING: {e}")
    print("[App] Server will start, but /predict will return 503 until the model is available.")
except Exception as e:
    MODEL_LOAD_ERROR = f"模型加载异常: {e}"
    print(f"[App] WARNING: {MODEL_LOAD_ERROR}")

# Allowed audio formats (handled via librosa + ffmpeg fallback)
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "flac", "mp4", "webm"}

# Accepted audio duration range (seconds), configurable via env vars.
MIN_AUDIO_SECONDS = float(os.environ.get("MIN_AUDIO_SECONDS", "10"))
MAX_AUDIO_SECONDS = float(os.environ.get("MAX_AUDIO_SECONDS", "60"))

# Cap upload size to avoid large-file abuse when exposed publicly (auto 413).
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _audio_duration_seconds(path: str) -> float:
    """Read audio duration in seconds via librosa. Raises on undecodable audio."""
    import librosa
    d, sr = librosa.load(path, sr=None, mono=True)
    return d.shape[0] / sr


def validate_duration(path: str) -> float:
    """
    Validate audio duration falls within [MIN_AUDIO_SECONDS, MAX_AUDIO_SECONDS].

    Returns the duration (seconds) on success, or raises ValueError with a
    user-facing Chinese message on failure.
    """
    try:
        duration = _audio_duration_seconds(path)
    except Exception:
        raise ValueError("无法解析音频文件，请检查文件是否完整。")
    if duration <= 0:
        raise ValueError("音频文件为空，无法识别。")
    if duration < MIN_AUDIO_SECONDS:
        raise ValueError(
            f"音频时长为 {duration:.1f} 秒，不足 {MIN_AUDIO_SECONDS:.0f} 秒，"
            f"请上传 {MIN_AUDIO_SECONDS:.0f}-{MAX_AUDIO_SECONDS:.0f} 秒的语音。"
        )
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"音频时长为 {duration:.1f} 秒，超过 {MAX_AUDIO_SECONDS:.0f} 秒，"
            f"请上传 {MIN_AUDIO_SECONDS:.0f}-{MAX_AUDIO_SECONDS:.0f} 秒的语音。"
        )
    return duration


def to_16k_wav(src_path: str, dst_path: str) -> str:
    """
    Convert any supported audio to 16kHz mono WAV (required by the model).
    Falls back to returning the source unchanged for .wav input.
    """

    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", dst_path],
        capture_output=True, check=True,
    )
    return dst_path


# ── HTML Template ─────────────────────────────────────────────────────────────
# Loaded from index.html (kept separate for cleanliness)
with open("index.html", "r") as f:
    HTML_TEMPLATE = f.read()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts: multipart form with 'audio' file, optional 'detail'
    Returns: JSON with AD classification result and metadata
    """
    if MODEL is None:
        return jsonify({"error": MODEL_LOAD_ERROR or "模型未加载"}), 503

    # Validate file exists in request
    if "audio" not in request.files:
        return jsonify({"error": "未提供音频文件 (字段名 audio)"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"不支持的格式。支持: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    detail = request.form.get("detail", "false") == "true"

    # Save uploaded file to a temp location
    suffix = "." + file.filename.rsplit(".", 1)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    converted = None
    try:
        # Enforce 10-60s duration limit (fail fast before expensive conversion)
        try:
            validate_duration(tmp_path)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Convert non-wav inputs to 16kHz mono wav
        if not tmp_path.lower().endswith(".wav"):
            converted = tmp_path + "_16k.wav"
            to_16k_wav(tmp_path, converted)

        wav_path = converted or tmp_path

        result = classify_audio(wav_path, MODEL, detail=detail)

        return jsonify({
            "success": True,
            "label": result["label"],
            "probs": result["probs"],
            "confidence": result["confidence"],
            "audio_duration": result["duration_seconds"],
            "processing_time": result["processing_time"],
            "detail": result.get("detail"),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Always clean up temp files
        for p in (tmp_path, converted):
            if p and os.path.exists(p):
                os.remove(p)


@app.route("/models")
def models():
    """Return available model info for UI display."""
    info = get_model_info()
    info["model_loaded"] = MODEL is not None
    info["load_error"] = MODEL_LOAD_ERROR
    return jsonify(info)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  🧠  AD 识别演示服务运行中!")
    print("  Open: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
