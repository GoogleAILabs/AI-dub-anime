import os
import sys
import gc
import json
import math
import re
import shutil
import hashlib
import tempfile
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import gradio as gr

# ================================================================
# ANIME DUB STUDIO PRO — KAGGLE LOCAL MAX / STORAGE-SAFE BUILD
# ================================================================
# Japanese/English -> Hindi anime dubbing pipeline.
#
# IMPORTANT STORAGE DESIGN
#   /kaggle/working  = persistent, small project data only
#   /kaggle/tmp      = heavy models + temporary audio/video + TTS cache
#
# This prevents the 19 GB /kaggle/working quota from being consumed by
# Hugging Face caches, Demucs stems, TTS WAVs and intermediate proxies.
#
# Persistent:
#   voicebank/       character references + embeddings + profile JSON
#   projects/        translation memory + character dialogue memory + manifest
#
# Scratch:
#   models/          HF / Transformers / Torch model caches
#   separation/      Demucs outputs
#   dialogue_cache/  per-session rendered TTS cache by default
#   tmp/             Gradio uploads (never bulk-deleted)
#   tmp/pipeline/    app-owned tempfile()/FFmpeg intermediates
#
# The pipeline is intentionally conservative about timing. Failed dialogue
# slots block final video export instead of silently trimming words or drifting.
# ================================================================

APP_TITLE = "Anime Dub Studio Pro — Kaggle Ultra FrameSync 100X"

# ---------------------------------------------------------------
# Storage
# ---------------------------------------------------------------

if os.path.isdir("/kaggle/working"):
    PERSIST_ROOT = os.environ.get(
        "ANIME_DUB_PERSIST_ROOT",
        "/kaggle/working/anime_dub_studio",
    )
    SCRATCH_ROOT = os.environ.get(
        "ANIME_DUB_SCRATCH_ROOT",
        "/kaggle/tmp/anime_dub_studio",
    )
else:
    PERSIST_ROOT = os.environ.get(
        "ANIME_DUB_PERSIST_ROOT",
        os.path.join(tempfile.gettempdir(), "anime_dub_studio"),
    )
    SCRATCH_ROOT = os.environ.get(
        "ANIME_DUB_SCRATCH_ROOT",
        os.path.join(tempfile.gettempdir(), "anime_dub_studio_scratch"),
    )

WORK_ROOT = PERSIST_ROOT
VOICEBANK_ROOT = os.environ.get(
    "VOICEBANK_ROOT", os.path.join(PERSIST_ROOT, "voicebank")
)
PROJECTS_ROOT = os.environ.get(
    "PROJECTS_ROOT", os.path.join(PERSIST_ROOT, "projects")
)
MODEL_CACHE = os.environ.get(
    "MODEL_CACHE", os.path.join(SCRATCH_ROOT, "models")
)
SEPARATION_ROOT = os.path.join(SCRATCH_ROOT, "separation")
DIALOGUE_CACHE_ROOT = os.path.join(SCRATCH_ROOT, "dialogue_cache")
TMP_ROOT = os.path.join(SCRATCH_ROOT, "tmp")
# IMPORTANT: Gradio uploads live directly under GRADIO_TEMP_DIR/TMP_ROOT.
# Never clean TMP_ROOT wholesale. Our own tempfile()/FFmpeg scratch goes here.
PIPE_TMP_ROOT = os.path.join(TMP_ROOT, "pipeline")
RESULT_ROOT = os.path.join(SCRATCH_ROOT, "results")

PERSIST_TTS_CACHE = os.environ.get("PERSIST_TTS_CACHE", "0") == "1"

for _p in [
    PERSIST_ROOT,
    SCRATCH_ROOT,
    VOICEBANK_ROOT,
    PROJECTS_ROOT,
    MODEL_CACHE,
    SEPARATION_ROOT,
    DIALOGUE_CACHE_ROOT,
    TMP_ROOT,
    PIPE_TMP_ROOT,
    RESULT_ROOT,
]:
    os.makedirs(_p, exist_ok=True)

# Route all common caches into scratch.
os.environ["TMPDIR"] = TMP_ROOT
os.environ["TEMP"] = TMP_ROOT
os.environ["TMP"] = TMP_ROOT
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(MODEL_CACHE, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(MODEL_CACHE, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(MODEL_CACHE, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(MODEL_CACHE, "datasets")
os.environ["TORCH_HOME"] = os.path.join(MODEL_CACHE, "torch")
os.environ["XDG_CACHE_HOME"] = os.path.join(MODEL_CACHE, "xdg")
os.environ["GRADIO_TEMP_DIR"] = TMP_ROOT

# IMPORTANT:
# Gradio uploaded inputs are stored under TMP_ROOT.
# Python/FFmpeg temporary files are stored under PIPE_TMP_ROOT.
# This separation prevents cleanup_intermediates() from deleting user uploads.
tempfile.tempdir = PIPE_TMP_ROOT

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()


def purge_legacy_working_caches():
    """Remove heavy folders left by older builds under persistent working storage."""
    legacy = [
        os.path.join(PERSIST_ROOT, "models"),
        os.path.join(PERSIST_ROOT, "separation"),
        os.path.join(PERSIST_ROOT, "tmp"),
    ]
    for path in legacy:
        if os.path.isdir(path):
            try:
                shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass

    # Older versions stored per-project TTS cache in persistent storage.
    if os.path.isdir(PROJECTS_ROOT):
        for project_dir in Path(PROJECTS_ROOT).iterdir():
            if not project_dir.is_dir():
                continue
            old_cache = project_dir / "dialogue_cache"
            if old_cache.is_dir() and not PERSIST_TTS_CACHE:
                try:
                    shutil.rmtree(old_cache, ignore_errors=True)
                except Exception:
                    pass


def disk_report(path):
    try:
        total, used, free = shutil.disk_usage(path)
        return {
            "total_gb": total / 1024**3,
            "used_gb": used / 1024**3,
            "free_gb": free / 1024**3,
        }
    except Exception:
        return None


purge_legacy_working_caches()

# ---------------------------------------------------------------
# Runtime / models
# ---------------------------------------------------------------

LIPSYNC_TARGET_FPS = float(os.environ.get("LIPSYNC_TARGET_FPS", "25"))
LIPSYNC_MAX_OFFSET_FRAMES = int(os.environ.get("LIPSYNC_MAX_OFFSET_FRAMES", "2"))
LIPSYNC_MIN_CONFIDENCE = float(os.environ.get("LIPSYNC_MIN_CONFIDENCE", "5.0"))
LIPSYNC_MAX_DRIFT_MS = float(os.environ.get("LIPSYNC_MAX_DRIFT_MS", "20"))
TTS_MAX_TEMPO_FAST = float(os.environ.get("TTS_MAX_TEMPO_FAST", "1.30"))
TTS_MAX_TEMPO_SLOW = float(os.environ.get("TTS_MAX_TEMPO_SLOW", "0.78"))
TTS_NATURALNESS_PASSES = int(os.environ.get("TTS_NATURALNESS_PASSES", "2"))
SYNCNET_WEIGHTS = os.environ.get(
    "SYNCNET_WEIGHTS",
    os.path.join(MODEL_CACHE, "syncnet", "syncnet_v2.model"),
)
SYNCNET_DIR = os.environ.get("SYNCNET_DIR", os.path.join(MODEL_CACHE, "syncnet"))

WAV2LIP_DIR = os.environ.get("WAV2LIP_DIR", os.path.join(MODEL_CACHE, "Wav2Lip"))
WAV2LIP_REPO = os.environ.get("WAV2LIP_REPO", WAV2LIP_DIR)
WAV2LIP_CHECKPOINT = os.environ.get(
    "WAV2LIP_CHECKPOINT",
    os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip.pth"),
)
WAV2LIP_GAN_CHECKPOINT = os.environ.get(
    "WAV2LIP_GAN_CHECKPOINT",
    os.path.join(WAV2LIP_REPO, "checkpoints", "wav2lip_gan.pth"),
)

LANGS = {"Hindi": "hi", "Hinglish": "hi"}

NLLB = {
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "hi": "hin_Deva",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
}

NLLB_REPO = os.environ.get("NLLB_REPO", "google/madlad400-3b-mt")
QWEN_REPO = os.environ.get("QWEN_REPO", "Qwen/Qwen3-4B")
PYANNOTE_REPO = os.environ.get(
    "PYANNOTE_REPO", "pyannote/speaker-diarization-community-1"
)
SPK_REPO = os.environ.get(
    "SPK_REPO", "speechbrain/spkrec-ecapa-voxceleb"
)
SENSEVOICE_REPO = os.environ.get("SENSEVOICE_REPO", "iic/SenseVoiceSmall")
CHATTERBOX_MODEL = os.environ.get("CHATTERBOX_T3", "v3")

MAX_CHARACTERS_DEFAULT = int(os.environ.get("MAX_CHARACTERS_DEFAULT", "160"))
MAX_REFERENCE_CLIPS = int(os.environ.get("MAX_REFERENCE_CLIPS", "12"))
VOICE_ID_SCORING_DEFAULT = os.environ.get("VOICE_ID_SCORING", "1") != "0"
SYNCNET_REQUIRED_DEFAULT = os.environ.get("SYNCNET_REQUIRED", "1") != "0"
SYNCNET_WINDOW_SEC = float(os.environ.get("SYNCNET_WINDOW_SEC", "1.60"))
SYNCNET_WINDOW_STEP_SEC = float(os.environ.get("SYNCNET_WINDOW_STEP_SEC", "0.80"))


def gpu_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def device():
    return "cuda" if gpu_available() else "cpu"


DEVICE = device()


def device_label():
    if DEVICE != "cuda":
        return "CPU"
    try:
        import torch
        return (
            f"CUDA • {torch.cuda.get_device_name(0)} • "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM"
        )
    except Exception:
        return "CUDA GPU"


DEVICE_LABEL = device_label()


def release_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def safe_int(value, default=None, minimum=None, maximum=None):
    """Robustly convert Gradio Number-like values to int."""
    if value is None:
        return default

    candidates = [value]
    if isinstance(value, dict):
        for key in ("value", "data", "number"):
            if key in value:
                candidates.append(value[key])

    for attr in ("value", "data"):
        try:
            if hasattr(value, attr):
                candidates.append(getattr(value, attr))
        except Exception:
            pass

    for candidate in candidates:
        try:
            if candidate is None:
                continue
            n = int(float(candidate))
            if minimum is not None:
                n = max(minimum, n)
            if maximum is not None:
                n = min(maximum, n)
            return n
        except Exception:
            continue
    return default


def safe_float(value, default=None, minimum=None, maximum=None):
    """Robustly convert Gradio Number-like values to float."""
    if value is None:
        return default
    candidates = [value]
    if isinstance(value, dict):
        for key in ("value", "data", "number"):
            if key in value:
                candidates.append(value[key])
    for attr in ("value", "data"):
        try:
            if hasattr(value, attr):
                candidates.append(getattr(value, attr))
        except Exception:
            pass
    for candidate in candidates:
        try:
            if candidate is None:
                continue
            n = float(candidate)
            if minimum is not None:
                n = max(minimum, n)
            if maximum is not None:
                n = min(maximum, n)
            return n
        except Exception:
            continue
    return default


def require_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            "FFmpeg/ffprobe नहीं मिला। Kaggle package cell में `apt-get install -y ffmpeg` चलाएँ।"
        )


def run_cmd(args, quiet=True, check=True):
    """
    Run a subprocess and preserve useful stderr.

    FFmpeg often explains the real failure only in stderr. The old version
    discarded that information and only reported "returned non-zero exit status".
    """
    stdout = subprocess.DEVNULL if quiet else None

    try:
        return subprocess.run(
            args,
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=True,
            check=check,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        cmd = " ".join(str(x) for x in args)

        if stderr:
            tail = stderr[-3000:]
            raise RuntimeError(
                f"Command failed (exit {e.returncode}):\n"
                f"{cmd}\n\nFFmpeg/tool stderr:\n{tail}"
            ) from e

        raise RuntimeError(
            f"Command failed (exit {e.returncode}): {cmd}"
        ) from e


def normalize_path(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("path") or x.get("name")
    return str(x)


def ffprobe_duration(path):
    require_ffmpeg()
    p = run_cmd(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ],
        quiet=False,
    )
    text = (p.stdout or "0").strip()
    return float(text or 0.0)


def mktemp_path(suffix="", prefix="tmp_"):
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return path


def _validate_input_file(path):
    """Validate a Gradio/user video before giving it to FFmpeg."""
    path = os.path.abspath(os.path.expanduser(str(path)))

    if not os.path.isfile(path):
        raise RuntimeError(
            f"Input video file does not exist: {path}"
        )

    size = os.path.getsize(path)

    if size < 1024:
        raise RuntimeError(
            f"Input video file is empty/too small ({size} bytes): {path}"
        )

    return path, size


def stage_input_video(video):
    """
    Copy the Gradio upload to an app-owned stable scratch location.

    Gradio may clean its upload directory after the request lifecycle.
    The staged copy is controlled by this pipeline and survives the full job.
    """
    video, size = _validate_input_file(video)

    stage_dir = os.path.join(
        PIPE_TMP_ROOT,
        "inputs",
    )
    os.makedirs(stage_dir, exist_ok=True)

    digest = hashlib.sha1(
        f"{video}|{size}|{os.path.getmtime(video):.6f}".encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()[:16]

    ext = Path(video).suffix.lower() or ".mp4"
    if ext not in {
        ".mp4", ".mkv", ".avi", ".mov", ".webm",
        ".m4v", ".ts", ".mts", ".m2ts", ".flv"
    }:
        ext = ".mp4"

    stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        Path(video).stem,
    ).strip("._-")[:70] or "source"

    staged = os.path.join(
        stage_dir,
        f"{stem}_{digest}{ext}",
    )

    if (
        os.path.exists(staged)
        and os.path.getsize(staged) == size
    ):
        return staged

    tmp_staged = os.path.join(
        stage_dir,
        f".{stem}_{digest}.part{ext}",
    )

    try:
        with open(video, "rb") as src_f, open(tmp_staged, "wb") as dst_f:
            shutil.copyfileobj(src_f, dst_f, length=16 * 1024 * 1024)
        os.replace(tmp_staged, staged)
    except Exception as e:
        try:
            if os.path.exists(tmp_staged):
                os.remove(tmp_staged)
        except Exception:
            pass
        raise RuntimeError(
            f"Failed to stage uploaded video into scratch storage: {e}"
        ) from e

    staged_size = os.path.getsize(staged)

    if staged_size != size:
        try:
            os.remove(staged)
        except Exception:
            pass
        raise RuntimeError(
            f"Staged video size mismatch: source={size} bytes, "
            f"staged={staged_size} bytes"
        )

    return staged


def extract_audio(video):
    require_ffmpeg()

    video, _ = _validate_input_file(video)

    # Prefer explicit first audio stream, but allow files without audio to
    # produce a clean error instead of an opaque FFmpeg exit code.
    try:
        probe = run_cmd(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=index",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video,
            ],
            quiet=False,
        )
        has_audio = bool((probe.stdout or "").strip())
    except Exception:
        has_audio = False

    if not has_audio:
        raise RuntimeError(
            f"No audio stream found in source video: {video}"
        )

    out = mktemp_path("_mono16k.wav", "extract_")

    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-v", "error",
            "-i", video,
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            out,
        ],
        quiet=False,
    )

    if not os.path.exists(out) or os.path.getsize(out) < 1024:
        raise RuntimeError(
            f"FFmpeg created an invalid/empty extracted audio file: {out}"
        )

    return out


def prepare_video(video):
    """
    Stage the original upload first, then create an MP4 proxy only when needed.

    We always work from the staged copy rather than directly from Gradio's
    transient upload path.
    """
    require_ffmpeg()

    staged = stage_input_video(video)

    # Probe the staged input before continuing.
    probe = run_cmd(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=format_name,duration",
            "-of", "json",
            staged,
        ],
        quiet=False,
    )

    try:
        data = json.loads(probe.stdout or "{}")
    except Exception as e:
        raise RuntimeError(
            f"Could not parse ffprobe output for staged source: {e}"
        ) from e

    duration = float(
        (data.get("format") or {}).get("duration") or 0.0
    )

    if duration <= 0.05:
        raise RuntimeError(
            f"Source video has invalid/zero duration: {staged}"
        )

    # MP4 is kept when FFprobe can read it successfully.
    if Path(staged).suffix.lower() == ".mp4":
        return staged

    out = mktemp_path("_source.mp4", "proxy_")

    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-v", "error",
            "-i", staged,
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "16",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "320k",
            out,
        ],
        quiet=False,
    )

    if not os.path.exists(out) or os.path.getsize(out) < 1024:
        raise RuntimeError(
            f"FFmpeg created an invalid proxy video: {out}"
        )

    return out


def load_audio(path, sr=16000):
    import soundfile as sf
    y, source_sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if source_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=source_sr, target_sr=sr)
    return np.asarray(y, dtype="float32"), sr


def save_audio(path, y, sr=16000):
    import soundfile as sf
    y = np.asarray(y, dtype="float32")
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.999:
        y = y / peak * 0.999
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_16")


def save_audio_float(path, y, sr=24000):
    import soundfile as sf
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(y, dtype="float32"), sr, subtype="FLOAT")


def cleanup_dir(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def prune_old_results(max_files=12):
    """Keep result scratch bounded without touching persistent project data."""
    try:
        files = [p for p in Path(RESULT_ROOT).glob("*") if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[max_files:]:
            try:
                p.unlink()
            except Exception:
                pass
    except Exception:
        pass


prune_old_results()


def cleanup_intermediates():
    """
    Delete only app-owned intermediate data.

    NEVER delete TMP_ROOT wholesale because GRADIO_TEMP_DIR points there and
    uploaded user files can live inside TMP_ROOT/<hash>/...
    """
    cleanup_dir(SEPARATION_ROOT)
    cleanup_dir(PIPE_TMP_ROOT)

    os.makedirs(SEPARATION_ROOT, exist_ok=True)
    os.makedirs(PIPE_TMP_ROOT, exist_ok=True)

    # Do not touch TMP_ROOT itself and do not touch Gradio upload folders.


# ---------------------------------------------------------------
# Audio statistics / separation
# ---------------------------------------------------------------

def audio_stats(y, sr=16000):
    y = np.asarray(y, dtype="float32")
    if len(y) == 0:
        return {"rms_db": -60.0, "pitch": 0.0, "duration": 0.0, "zcr": 0.0}
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-9)
    rms_db = 20.0 * math.log10(rms)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(y).astype(np.int8))))) if len(y) > 1 else 0.0
    pitch = 0.0
    try:
        import librosa
        sample = y[: min(len(y), sr * 5)]
        if len(sample) >= sr // 4:
            f0 = librosa.yin(sample, fmin=70, fmax=500, sr=sr)
            valid = f0[np.isfinite(f0)]
            if len(valid):
                pitch = float(np.median(valid))
    except Exception:
        pass
    return {
        "rms_db": rms_db,
        "pitch": pitch,
        "duration": len(y) / sr,
        "zcr": zcr,
    }


def separate_dialogue(audio_path, progress, log, model_name="htdemucs_ft"):
    """Recover dialogue/vocals and accompaniment into Kaggle scratch storage."""
    try:
        import demucs.separate
        progress(0.10, "Dialogue / BGM-SFX separation…")
        out_root = SEPARATION_ROOT
        os.makedirs(out_root, exist_ok=True)
        args = [
            "--two-stems", "vocals",
            "-n", model_name,
            "--out", out_root,
            "--segment", "7",
            audio_path,
        ]
        if DEVICE == "cuda":
            args[-1:-1] = ["--device", "cuda"]
        try:
            demucs.separate.main(args)
        except Exception:
            args = [
                "--two-stems", "vocals", "-n", model_name,
                "--out", out_root, "--segment", "7", audio_path,
            ]
            demucs.separate.main(args)
        stem_dir = os.path.join(out_root, model_name, Path(audio_path).stem)
        vocals = os.path.join(stem_dir, "vocals.wav")
        accomp = os.path.join(stem_dir, "no_vocals.wav")
        if not os.path.exists(vocals):
            raise RuntimeError("Demucs vocals stem नहीं मिला")
        log.append(f"✅ Separation complete: {model_name}")
        log.append("ℹ️ Accompaniment में BGM + बहुत-से SFX रहेंगे; ये recovered original bed है।")
        return vocals, accomp
    except Exception as e:
        log.append(f"⚠️ Separation failed: {str(e)[:180]}")
        return audio_path, None


def build_original_residual_bed(original_audio, vocals_audio, progress=None, log=None):
    try:
        if progress:
            progress(0.56, "Rebuilding original BGM/SFX bed…")
        orig, sr = load_audio(original_audio, 24000)
        voc, _ = load_audio(vocals_audio, 24000)
        n = max(len(orig), len(voc))
        orig = np.pad(orig, (0, max(0, n - len(orig))))[:n]
        voc = np.pad(voc, (0, max(0, n - len(voc))))[:n]
        residual = orig - voc
        peak = float(np.max(np.abs(residual))) if len(residual) else 0.0
        if peak > 1.20:
            residual *= 1.20 / peak
        out = mktemp_path("_original_residual_bed.wav", "bed_")
        save_audio_float(out, residual, sr)
        if log:
            log.append("✅ Original residual BGM/SFX bed reconstructed from source master")
        return out
    except Exception as e:
        if log:
            log.append(f"⚠️ Residual bed reconstruction failed: {str(e)[:120]}")
        return None

# ---------------------------------------------------------------
# WhisperX transcription + alignment
# ---------------------------------------------------------------

def transcribe_whisperx(audio_path, model_size, progress, log):
    progress(0.16, f"WhisperX {model_size} transcription…")
    try:
        import whisperx
        compute_type = "float16" if DEVICE == "cuda" else "int8"
        model = whisperx.load_model(
            model_size,
            DEVICE,
            compute_type=compute_type,
            download_root=os.path.join(MODEL_CACHE, "whisperx"),
            language=None,
            vad_method="silero",
        )
        audio = whisperx.load_audio(audio_path)
        batch_size = 8 if DEVICE == "cuda" else 1
        result = model.transcribe(
            audio,
            batch_size=batch_size,
            chunk_size=30,
            print_progress=False,
        )
        source_lang = result.get("language", "en") or "en"
        del model
        release_gpu()

        try:
            progress(0.22, "Word-level alignment…")
            align_model, metadata = whisperx.load_align_model(
                language_code=source_lang,
                device=DEVICE,
                model_dir=os.path.join(MODEL_CACHE, "alignment"),
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
            )
            del align_model
            release_gpu()
            log.append("✅ Word-level alignment complete")
        except Exception as e:
            log.append(f"⚠️ Alignment fallback: {str(e)[:120]}")

        segments = []
        for raw in result.get("segments", []):
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            words = []
            for w in raw.get("words", []) or []:
                if w.get("start") is None or w.get("end") is None:
                    continue
                words.append({
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "word": str(w.get("word", "")).strip(),
                })
            segments.append({
                "start": float(raw.get("start", 0.0)),
                "end": float(raw.get("end", 0.0)),
                "text": text,
                "words": words,
                "speaker": "SPEAKER_00",
                "language": source_lang,
            })
        log.append(f"✅ Source: {source_lang} • {len(segments)} segments")
        return source_lang, segments
    except Exception as e:
        log.append(f"⚠️ WhisperX unavailable: {str(e)[:140]}")
        log.append("↪ Using faster-whisper fallback")
        try:
            from faster_whisper import WhisperModel
            compute_type = "float16" if DEVICE == "cuda" else "int8"
            model = WhisperModel(
                model_size,
                device=DEVICE,
                compute_type=compute_type,
                download_root=os.path.join(MODEL_CACHE, "faster_whisper"),
            )
            it, info = model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
                condition_on_previous_text=False,
            )
            segs = []
            for s in it:
                text = s.text.strip()
                if not text:
                    continue
                words = []
                for w in s.words or []:
                    if w.start is not None and w.end is not None:
                        words.append({
                            "start": float(w.start),
                            "end": float(w.end),
                            "word": w.word.strip(),
                        })
                segs.append({
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": text,
                    "words": words,
                    "speaker": "SPEAKER_00",
                    "language": getattr(info, "language", "en") or "en",
                })
            source_lang = getattr(info, "language", "en") or "en"
            del model
            release_gpu()
            return source_lang, segs
        except Exception as e2:
            raise RuntimeError(f"ASR failed: {str(e2)[:180]}") from e2

# ---------------------------------------------------------------
# Speaker diarization — pyannote Community-1 compatible loader
# ---------------------------------------------------------------

def _load_pyannote_pipeline(log):
    from pyannote.audio import Pipeline

    cache_dir = os.path.join(MODEL_CACHE, "pyannote")
    os.makedirs(cache_dir, exist_ok=True)

    # Current Community-1 uses token=. Older pyannote versions may use
    # use_auth_token=. Some versions expose **kwargs and accept token silently.
    attempts = []
    if HF_TOKEN:
        attempts.extend([
            {"token": HF_TOKEN, "cache_dir": cache_dir},
            {"use_auth_token": HF_TOKEN, "cache_dir": cache_dir},
            {"token": HF_TOKEN},
            {"use_auth_token": HF_TOKEN},
        ])
    attempts.extend([{"cache_dir": cache_dir}, {}])

    last_type_error = None
    for kwargs in attempts:
        try:
            return Pipeline.from_pretrained(PYANNOTE_REPO, **kwargs)
        except TypeError as e:
            msg = str(e)
            if "unexpected keyword argument" in msg:
                last_type_error = e
                continue
            raise

    if last_type_error:
        raise last_type_error
    return Pipeline.from_pretrained(PYANNOTE_REPO)


def diarize(audio_path, progress, log, min_speakers=None, max_speakers=None):
    if not HF_TOKEN:
        log.append("⚠️ HF_TOKEN absent: multi-character identification disabled.")
        dur = ffprobe_duration(audio_path)
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": dur}]

    try:
        progress(0.25, "Character/speaker diarization…")
        import pyannote.audio

        pipe = _load_pyannote_pipeline(log)
        try:
            import torch
            pipe.to(torch.device(DEVICE))
        except Exception:
            pass

        kwargs = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            kwargs["max_speakers"] = int(max_speakers)

        output = pipe(audio_path, **kwargs)

        # Community-1 returns a DiarizeOutput. Older versions may return
        # Annotation directly. Prefer exclusive diarization when available
        # because it is easier to reconcile against transcription cues.
        ann = getattr(output, "exclusive_speaker_diarization", None)
        if ann is None:
            ann = getattr(output, "speaker_diarization", output)

        turns = []
        for turn, _, speaker in ann.itertracks(yield_label=True):
            turns.append({
                "speaker": str(speaker),
                "start": float(turn.start),
                "end": float(turn.end),
            })

        unique = sorted({t["speaker"] for t in turns})
        try:
            log.append(f"ℹ️ pyannote.audio version: {pyannote.audio.__version__}")
        except Exception:
            pass
        log.append(f"✅ Diarization: {len(unique)} source speaker(s)")

        del pipe
        release_gpu()

        if not turns:
            dur = ffprobe_duration(audio_path)
            turns = [{"speaker": "SPEAKER_00", "start": 0.0, "end": dur}]
        return turns
    except Exception as e:
        log.append(f"⚠️ Diarization failed: {str(e)[:200]}")
        log.append("↪ Falling back to one speaker")
        dur = ffprobe_duration(audio_path)
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": dur}]


def interval_overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(segments, turns):
    if not turns:
        for s in segments:
            s["speaker"] = "SPEAKER_00"
        return segments
    for seg in segments:
        candidates = []
        for t in turns:
            ov = interval_overlap(seg["start"], seg["end"], t["start"], t["end"])
            if ov > 0:
                candidates.append((ov, t))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            seg["speaker"] = candidates[0][1]["speaker"]
        else:
            mid = (seg["start"] + seg["end"]) / 2.0
            nearest = min(
                turns,
                key=lambda t: abs((t["start"] + t["end"]) / 2 - mid),
            )
            seg["speaker"] = nearest["speaker"]
    return segments

# ---------------------------------------------------------------
# Timing / frame lock
# ---------------------------------------------------------------

def probe_video_timing(video):
    require_ffmpeg()
    p = run_cmd(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames",
            "-of", "json", video,
        ],
        quiet=False,
    )
    data = json.loads(p.stdout or "{}")
    st = (data.get("streams") or [{}])[0]

    def fps_from(value):
        try:
            a, b = str(value).split("/")
            return float(a) / float(b)
        except Exception:
            try:
                return float(value)
            except Exception:
                return 25.0

    fps = fps_from(st.get("avg_frame_rate") or st.get("r_frame_rate") or "25/1")
    frames = None
    try:
        frames = int(st.get("nb_frames")) if st.get("nb_frames") else None
    except Exception:
        frames = None
    if not frames:
        dur = ffprobe_duration(video)
        frames = int(round(dur * fps))
    return {"fps": fps, "frames": frames, "duration": frames / max(fps, 1e-6)}


def snap_segments_to_frame_grid(video, segments, log):
    meta = probe_video_timing(video)
    fps = meta["fps"] if meta["fps"] > 1 else LIPSYNC_TARGET_FPS
    frame = 1.0 / fps
    for s in segments:
        start = max(0.0, float(s.get("start", 0.0)))
        end = max(start + frame, float(s.get("end", start + frame)))
        sf = int(round(start * fps))
        ef = max(sf + 1, int(round(end * fps)))
        s["source_start"] = start
        s["source_end"] = end
        s["frame_fps"] = fps
        s["start_frame"] = sf
        s["end_frame"] = ef
        s["start"] = sf / fps
        s["end"] = ef / fps
        s["cue_duration"] = max(frame, s["end"] - s["start"])

    ordered = sorted(segments, key=lambda x: x["start"])
    for i in range(1, len(ordered)):
        prev = ordered[i - 1]
        cur = ordered[i]
        if cur["start"] < prev["end"]:
            cur["start"] = prev["end"]
            cur["start_frame"] = int(round(cur["start"] * fps))
            if cur["end"] <= cur["start"]:
                cur["end"] = min(meta["duration"], cur["start"] + frame)
                cur["end_frame"] = int(round(cur["end"] * fps))
            cur["cue_duration"] = max(frame, cur["end"] - cur["start"])

    log.append(
        f"🎞️ Frame grid locked: source FPS={fps:.3f}, boundary quantization ≤{500.0/fps:.1f}ms"
    )
    return segments, meta


def lock_cues_to_aligned_words(segments, log):
    changed = 0
    for s in segments:
        words = [
            w for w in (s.get("words") or [])
            if w.get("start") is not None and w.get("end") is not None
        ]
        if not words:
            continue
        outer_start = float(s.get("start", 0.0))
        outer_end = float(s.get("end", outer_start))
        w0 = max(outer_start, min(float(w["start"]) for w in words))
        w1 = min(outer_end, max(float(w["end"]) for w in words))
        if w1 - w0 >= 0.08:
            ns = max(outer_start, w0 - 0.012)
            ne = min(outer_end, w1 + 0.018)
            if ne > ns:
                s["aligned_start"] = ns
                s["aligned_end"] = ne
                s["start"] = ns
                s["end"] = ne
                changed += 1
    log.append(f"✅ Word-boundary cue lock: {changed}/{len(segments)} segments refined")
    return segments


def refine_cue_with_energy(audio_path, seg, sr=16000):
    try:
        y, _ = load_audio(audio_path, sr)
        a = max(0, int(float(seg["start"]) * sr))
        b = min(len(y), int(float(seg["end"]) * sr))
        if b <= a + int(0.06 * sr):
            return seg
        clip = y[a:b]
        import librosa
        intervals = librosa.effects.split(
            clip,
            top_db=35,
            frame_length=512,
            hop_length=128,
        )
        if len(intervals) == 0:
            return seg
        on = max(0, int(intervals[0][0])) / sr
        off = min(len(clip), int(intervals[-1][1])) / sr
        refined_start = max(seg["start"], seg["start"] + max(0.0, on - 0.012))
        refined_end = min(
            seg["end"],
            seg["start"] + min(off + 0.018, seg["end"] - seg["start"]),
        )
        if refined_end - refined_start >= 0.08:
            seg["speech_onset"] = refined_start
            seg["speech_offset"] = refined_end
        return seg
    except Exception:
        return seg


def refine_all_cues(audio_path, segments, progress, log):
    out = []
    for i, s in enumerate(segments):
        out.append(refine_cue_with_energy(audio_path, s))
        progress(
            0.20 + 0.03 * ((i + 1) / max(1, len(segments))),
            f"Cue timing refine {i+1}/{len(segments)}…",
        )
    log.append("✅ Source dialogue onset/offset refined before Hindi rendering")
    return out

# ---------------------------------------------------------------
# Emotion / prosody
# ---------------------------------------------------------------

def hindi_speech_budget(text):
    text = re.sub(r"\s+", " ", str(text or "").strip())
    words = len(re.findall(r"\S+", text))
    letters = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    syllable_proxy = max(words, int(letters * 0.52) + int(latin * 0.45))
    return {"words": words, "syllable_proxy": syllable_proxy}


def hindi_delivery_load(text):
    text = re.sub(r"\s+", " ", str(text or "").strip())
    fallback = hindi_speech_budget(text)
    if not text:
        return {"words": 0, "phonemes": 0, "method": "empty"}
    if shutil.which("espeak-ng"):
        try:
            cp = subprocess.run(
                ["espeak-ng", "-q", "-v", "hi", "--ipa", text],
                capture_output=True,
                text=True,
                check=True,
            )
            ipa = re.sub(r"\s+", "", cp.stdout or "")
            phones = len([
                ch for ch in ipa
                if ch.isalpha() or ch in "ɐəɛɪɔʊʃʒŋɲɳʈɖɽɻʂɦ"
            ])
            return {
                "words": fallback["words"],
                "phonemes": max(phones, fallback["syllable_proxy"]),
                "method": "espeak-ng",
            }
        except Exception:
            pass
    return {
        "words": fallback["words"],
        "phonemes": fallback["syllable_proxy"],
        "method": "grapheme-fallback",
    }


def analyze_segment_prosody(source_audio, segments):
    y, sr = load_audio(source_audio, 16000)
    global_e = []
    global_p = []
    data = []
    for s in segments:
        a = max(0, int(s["start"] * sr))
        b = min(len(y), int(s["end"] * sr))
        chunk = y[a:b]
        st = audio_stats(chunk, sr)
        data.append(st)
        global_e.append(st["rms_db"])
        if st["pitch"] > 0:
            global_p.append(st["pitch"])
    em = np.median(global_e) if global_e else -24.0
    es = np.std(global_e) + 1e-6
    pm = np.median(global_p) if global_p else 180.0
    ps = np.std(global_p) + 1e-6
    for s, st in zip(segments, data):
        ez = (st["rms_db"] - em) / es
        pz = (st["pitch"] - pm) / ps if st["pitch"] > 0 else 0.0
        if ez > 1.15 and pz > 0.6:
            emotion = "excited"
        elif ez > 1.2:
            emotion = "angry"
        elif ez < -0.9 and pz < -0.35:
            emotion = "sad"
        elif pz > 1.3 and ez < 0.7:
            emotion = "surprised"
        elif ez < -0.45:
            emotion = "calm"
        else:
            emotion = "neutral"
        s["source_stats"] = st
        s["emotion"] = emotion
    return segments


SENSEVOICE = None


def load_sensevoice(progress, log):
    global SENSEVOICE
    if SENSEVOICE is not None:
        return SENSEVOICE
    try:
        from funasr import AutoModel
        if progress:
            progress(0.23, "Loading local SenseVoice emotion model…")
        SENSEVOICE = AutoModel(
            model=SENSEVOICE_REPO,
            device=DEVICE,
            disable_update=True,
            disable_pbar=True,
        )
        log.append("✅ SenseVoiceSmall loaded locally for emotion/event cues")
        return SENSEVOICE
    except Exception as e:
        log.append(f"⚠️ SenseVoice unavailable: {str(e)[:130]}")
        return None


def sensevoice_emotion(path, source_lang, log, model=None):
    model = model or load_sensevoice(None, log)
    if model is None or source_lang not in {"en", "ja", "ko", "zh", "yue"}:
        return None, []
    try:
        result = model.generate(
            input=path,
            language=source_lang,
            use_itn=True,
            output_timestamp=False,
        )
        if isinstance(result, list):
            result = result[0] if result else {}
        tagged = str(result.get("text", "")) if isinstance(result, dict) else str(result)
        tags = re.findall(r"<\|([A-Za-z_]+)\|>", tagged)
        upper = {t.upper() for t in tags}
        emotion = None
        if "HAPPY" in upper:
            emotion = "happy"
        elif "SAD" in upper:
            emotion = "sad"
        elif "ANGRY" in upper:
            emotion = "angry"
        elif "NEUTRAL" in upper:
            emotion = "neutral"
        events = [
            t.lower() for t in tags
            if t.upper() in {"LAUGH", "CRY", "SPEECH", "BREATH"}
        ]
        return emotion, events
    except Exception:
        return None, []


def _map_emotion(label):
    if label == "happy":
        return "excited"
    if label == "sad":
        return "sad"
    if label == "angry":
        return "angry"
    return "neutral"


def classify_emotions(source_audio, source_lang, segments, progress, log, use_model=True):
    segments = analyze_segment_prosody(source_audio, segments)
    sv = load_sensevoice(progress, log) if use_model else None
    total = len(segments)
    for i, s in enumerate(segments):
        tag = None
        events = []
        if sv is not None and source_lang in {"en", "ja", "ko", "zh", "yue"}:
            start = max(0.0, s["start"] - 0.03)
            end = max(start + 0.25, s["end"] + 0.03)
            clip = mktemp_path("_emo.wav", "emo_")
            try:
                run_cmd([
                    "ffmpeg", "-y", "-i", source_audio,
                    "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                    "-ac", "1", "-ar", "16000", clip,
                ])
                tag, events = sensevoice_emotion(clip, source_lang, log, sv)
            except Exception:
                pass
            finally:
                try:
                    os.remove(clip)
                except Exception:
                    pass
        acoustic = s.get("emotion", "neutral")
        s["model_emotion"] = _map_emotion(tag) if tag else None
        s["events"] = events
        s["emotion"] = _map_emotion(tag) if tag else acoustic
        s["emotion_confidence"] = 0.86 if tag else 0.48
        if tag is None and acoustic in {"angry", "excited"}:
            s["emotion_confidence"] = 0.55
        progress(
            0.23 + 0.07 * ((i + 1) / max(1, total)),
            f"Emotion analysis {i+1}/{total}…",
        )
    return segments

# ---------------------------------------------------------------
# Persistent character voice bank
# ---------------------------------------------------------------

def safe_project_id(name):
    name = (name or "anime").strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    return name[:80] or "anime"


def bank_dir(project_id):
    path = os.path.join(VOICEBANK_ROOT, safe_project_id(project_id))
    os.makedirs(path, exist_ok=True)
    return path


def memory_dir(project_id):
    path = os.path.join(PROJECTS_ROOT, safe_project_id(project_id))
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


SPK_MODEL = None


def get_speaker_encoder(log):
    global SPK_MODEL
    if SPK_MODEL is not None:
        return SPK_MODEL
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        SPK_MODEL = EncoderClassifier.from_hparams(
            source=SPK_REPO,
            savedir=os.path.join(MODEL_CACHE, "speaker_encoder"),
            run_opts={"device": os.environ.get("SPK_DEVICE", "cpu")},
        )
        log.append("✅ Speaker identity encoder loaded")
        return SPK_MODEL
    except Exception as e:
        log.append(f"⚠️ Speaker embedding model unavailable: {str(e)[:140]}")
        return None


def speaker_embedding(wav_path, log):
    model = get_speaker_encoder(log)
    if model is None:
        return None
    try:
        import torch
        y, _ = load_audio(wav_path, 16000)
        if len(y) < 16000:
            y = np.pad(y, (0, 16000 - len(y)))
        tensor = torch.from_numpy(y).float().unsqueeze(0)
        with torch.inference_mode():
            emb = (
                model.encode_batch(tensor)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype("float32")
            )
        norm = float(np.linalg.norm(emb)) + 1e-9
        return emb / norm
    except Exception as e:
        log.append(f"⚠️ Speaker embedding failed: {str(e)[:110]}")
        return None


def unload_speaker_encoder():
    global SPK_MODEL
    SPK_MODEL = None
    release_gpu()


def unload_sensevoice():
    global SENSEVOICE
    SENSEVOICE = None
    release_gpu()


def cosine(a, b):
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def extract_speaker_refs(audio_path, turns, speaker, max_refs=4, max_each=8.0):
    y, sr = load_audio(audio_path, 16000)
    candidates = []
    for t in turns:
        if t["speaker"] != speaker:
            continue
        duration = float(t["end"] - t["start"])
        if duration < 0.9:
            continue
        start = max(0, int(t["start"] * sr))
        end = min(len(y), int(t["end"] * sr))
        if end <= start:
            continue
        chunk = y[start:end]
        st = audio_stats(chunk, sr)
        st["start"] = float(t["start"])
        st["end"] = float(t["end"])
        if st["rms_db"] < -45:
            continue
        use = chunk[: int(max_each * sr)]
        loudness_score = max(0.0, 1.0 - abs(st["rms_db"] + 22.0) / 18.0)
        duration_score = min(1.0, duration / max_each)
        score = 0.65 * duration_score + 0.35 * loudness_score
        candidates.append((score, use, st))
    candidates.sort(key=lambda x: x[0], reverse=True)
    refs = []
    used = []
    for _, chunk, st in candidates:
        if len(refs) >= max_refs:
            break
        sig = hashlib.sha1(chunk[: min(len(chunk), sr)]).hexdigest()
        if sig in used:
            continue
        used.append(sig)
        out = mktemp_path("_speaker_ref.wav", "spkref_")
        save_audio(out, chunk, sr)
        refs.append((out, st))
    return refs


def list_profiles(project_id):
    root = bank_dir(project_id)
    profiles = []
    for p in sorted(Path(root).glob("char_*.json")):
        data = load_json(str(p), {})
        if data:
            profiles.append(data)
    return profiles


def save_profile(project_id, profile):
    root = bank_dir(project_id)
    save_json(os.path.join(root, f"{profile['id']}.json"), profile)


def load_profile(project_id, cid):
    return load_json(os.path.join(bank_dir(project_id), f"{cid}.json"), {})


def profile_reference_paths(profile, project_id):
    root = bank_dir(project_id)
    paths = []
    for item in profile.get("references", []):
        fn = item.get("file") if isinstance(item, dict) else item
        if fn:
            p = os.path.join(root, fn)
            if os.path.exists(p):
                paths.append((p, item))
    return paths


def _interval_emotion(segments, start, end):
    best = None
    best_ov = 0.0
    for s in segments:
        ov = interval_overlap(start, end, s["start"], s["end"])
        if ov > best_ov:
            best_ov = ov
            best = s
    return best.get("emotion", "neutral") if best else "neutral"


def _next_character_id(profiles):
    nums = []
    for p in profiles:
        m = re.search(r"char_(\d+)$", p.get("id", ""))
        if m:
            nums.append(int(m.group(1)))
    return f"char_{(max(nums) + 1 if nums else 1):03d}"


def build_voice_bank(
    project_id,
    turns,
    source_audio,
    threshold,
    progress,
    log,
    segments=None,
    max_characters=160,
):
    profiles = list_profiles(project_id)
    root = bank_dir(project_id)
    speaker_ids = sorted({t["speaker"] for t in turns})
    mapping = {}
    report = []
    segments = segments or []
    progress(0.30, f"Learning {len(speaker_ids)} character voice identities…")

    for idx, speaker in enumerate(speaker_ids):
        refs = extract_speaker_refs(
            source_audio,
            turns,
            speaker,
            max_refs=6,
            max_each=10.0,
        )
        if not refs:
            continue

        embs = []
        ref_records = []
        for path, st in refs:
            emb = speaker_embedding(path, log)
            ref_emotion = _interval_emotion(
                segments,
                st.get("start", 0.0),
                st.get("end", 0.0),
            )
            if emb is not None:
                embs.append(emb)
            ref_records.append((path, st, ref_emotion, emb))

        emb_mean = None
        if embs:
            emb_mean = np.mean(np.stack(embs), axis=0)
            emb_mean = emb_mean / (np.linalg.norm(emb_mean) + 1e-9)

        chosen = None
        score = 0.0
        margin = 0.0
        if emb_mean is not None and profiles:
            ranked = []
            for prof in profiles:
                sims = []
                if prof.get("embedding"):
                    sims.append(
                        cosine(
                            emb_mean,
                            np.asarray(prof["embedding"], dtype="float32"),
                        )
                    )
                for hist in prof.get("prototype_embeddings", [])[-12:]:
                    sims.append(
                        cosine(emb_mean, np.asarray(hist, dtype="float32"))
                    )
                if sims:
                    emb_score = 0.70 * max(sims) + 0.30 * float(
                        np.mean(sorted(sims, reverse=True)[:3])
                    )
                    p0 = float(prof.get("canonical_pitch", 0.0))
                    p1 = (
                        float(
                            np.median([
                                st.get("pitch", 0.0)
                                for _, st, _, _ in ref_records
                            ])
                        )
                        if ref_records
                        else 0.0
                    )
                    if p0 > 0 and p1 > 0:
                        pitch_sim = math.exp(
                            -abs(math.log((p0 + 1e-6) / (p1 + 1e-6))) / 0.35
                        )
                    else:
                        pitch_sim = 0.5
                    pscore = 0.84 * emb_score + 0.16 * pitch_sim
                    ranked.append((pscore, prof))
            ranked.sort(key=lambda x: x[0], reverse=True)
            if ranked:
                score, chosen = ranked[0]
                second = ranked[1][0] if len(ranked) > 1 else 0.0
                margin = score - second
                if score < threshold or (len(ranked) > 1 and margin < 0.045):
                    chosen = None

        if chosen is None:
            if len(profiles) >= max_characters:
                report.append(
                    f"{speaker} → UNRESOLVED (voice-bank limit {max_characters})"
                )
                continue
            cid = _next_character_id(profiles)
            chosen = {
                "id": cid,
                "name": f"Character {int(cid.split('_')[1]):02d}",
                "created": datetime.utcnow().isoformat() + "Z",
                "episodes_seen": 0,
                "embedding": emb_mean.tolist() if emb_mean is not None else None,
                "prototype_embeddings": [e.tolist() for e in embs[-8:]],
                "canonical_pitch": float(
                    np.median([
                        st.get("pitch", 0.0)
                        for _, st, _, _ in ref_records
                    ])
                ) if ref_records else 0.0,
                "canonical_rms_db": float(
                    np.median([
                        st.get("rms_db", -25.0)
                        for _, st, _, _ in ref_records
                    ])
                ) if ref_records else -25.0,
                "references": [],
                "reference_emotions": {},
                "source_speakers": [],
                "locked": True,
            }
            profiles.append(chosen)
            action = "NEW"
        else:
            action = f"MATCH {score:.3f} margin={margin:.3f}"
            if emb_mean is not None:
                chosen.setdefault("prototype_embeddings", []).append(
                    emb_mean.tolist()
                )
                chosen["prototype_embeddings"] = chosen[
                    "prototype_embeddings"
                ][-12:]
                if not chosen.get("embedding"):
                    chosen["embedding"] = emb_mean.tolist()

        chosen["episodes_seen"] = int(chosen.get("episodes_seen", 0)) + 1
        chosen.setdefault("source_speakers", []).append({
            "speaker": speaker,
            "seen": datetime.utcnow().isoformat() + "Z",
            "match_score": float(score),
            "margin": float(margin),
        })

        existing_files = {
            item.get("file")
            for item in chosen.get("references", [])
            if isinstance(item, dict)
        }
        ref_quota = min(MAX_REFERENCE_CLIPS, 12)
        for src_path, st, ref_emotion, emb in sorted(
            ref_records,
            key=lambda x: x[1].get("duration", 0),
            reverse=True,
        ):
            if len(existing_files) >= ref_quota:
                break
            dest_name = f"{chosen['id']}_ref_{len(existing_files)+1:02d}.wav"
            dest = os.path.join(root, dest_name)
            if dest_name in existing_files:
                continue
            try:
                shutil.copy2(src_path, dest)
                chosen.setdefault("references", []).append({
                    "file": dest_name,
                    "rms_db": float(st.get("rms_db", -25.0)),
                    "pitch": float(st.get("pitch", 0.0)),
                    "duration": float(st.get("duration", 0.0)),
                    "emotion": ref_emotion,
                    "embedding": emb.tolist() if emb is not None else None,
                })
                chosen.setdefault("reference_emotions", {}).setdefault(
                    ref_emotion, []
                ).append(dest_name)
                existing_files.add(dest_name)
            except Exception:
                pass

        chosen["updated"] = datetime.utcnow().isoformat() + "Z"
        save_profile(project_id, chosen)
        mapping[speaker] = chosen["id"]
        report.append(
            f"{speaker} → {chosen['id']} ({chosen.get('name')}) "
            f"[{action}] refs={len(chosen.get('references', []))}"
        )
        progress(
            0.30 + 0.09 * ((idx + 1) / max(1, len(speaker_ids))),
            f"Character {idx+1}/{len(speaker_ids)}…",
        )

    log.append("🎭 Persistent Character Voice Bank")
    log.extend(["  • " + x for x in report])
    return mapping, report


def resolve_reference(project_id, profile, target_stats=None, emotion="neutral"):
    refs = profile_reference_paths(profile, project_id)
    if not refs:
        return None
    candidates = refs
    emo_pool = [
        r for r in refs
        if str(r[1].get("emotion", "neutral")) == str(emotion)
    ]
    if emo_pool:
        candidates = emo_pool
    if not target_stats:
        return candidates[0][0]

    def score(item):
        _, meta = item
        e = (
            abs(float(meta.get("rms_db", -25.0)) - float(target_stats.get("rms_db", -25.0)))
            / 18.0
        )
        p0 = float(meta.get("pitch", 0.0))
        p1 = float(target_stats.get("pitch", 0.0))
        p = (
            abs(math.log((p0 + 1e-6) / (p1 + 1e-6)))
            if p0 > 0 and p1 > 0
            else 0.25
        )
        emo_bonus = -0.35 if str(meta.get("emotion", "neutral")) == str(emotion) else 0.0
        return e + 0.30 * p + emo_bonus

    return min(candidates, key=score)[0]


def export_voicebank(project_id):
    root = bank_dir(project_id)
    out = os.path.join(
        RESULT_ROOT,
        f"{safe_project_id(project_id)}_voicebank.zip",
    )
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in Path(root).rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(root))
    return out

# ---------------------------------------------------------------
# Translation memory / MADLAD / Qwen
# ---------------------------------------------------------------

def normalize_source_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def load_translation_memory(project_id):
    return load_json(
        os.path.join(memory_dir(project_id), "translation_memory.json"),
        {},
    )


def save_translation_memory(project_id, memory):
    save_json(
        os.path.join(memory_dir(project_id), "translation_memory.json"),
        memory,
    )


def load_local_nllb(progress, log):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    progress(0.38, f"Loading local translation model {NLLB_REPO}…")
    tokenizer = AutoTokenizer.from_pretrained(
        NLLB_REPO,
        cache_dir=os.path.join(MODEL_CACHE, "nllb"),
    )
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_REPO,
        cache_dir=os.path.join(MODEL_CACHE, "nllb"),
        torch_dtype=dtype,
    ).to(DEVICE)
    model.eval()
    log.append(f"✅ Local translation model loaded: {NLLB_REPO}")
    return tokenizer, model


CURRENT_PROJECT_ID = "anime"


def current_project_for_memory():
    return CURRENT_PROJECT_ID


def nllb_translate(segments, source_lang, target_lang, progress, log):
    """Translate to Hindi with MADLAD-400 locally. Target prefix stays <2hi>."""
    actual_target = LANGS.get(target_lang, "hi")
    if actual_target != "hi":
        raise RuntimeError("इस MAX build का primary target Hindi/Hinglish है।")
    if source_lang == "hi":
        for s in segments:
            s["translation"] = s["text"]
        return segments

    memory = load_translation_memory(current_project_for_memory())
    pending = []
    for s in segments:
        key = normalize_source_text(s["text"])
        cached = memory.get(key)
        if cached:
            s["translation"] = cached
        else:
            pending.append(s)
    if not pending:
        log.append("✅ Translation memory hit — no new translation inference needed")
        return segments

    tokenizer = model = None
    try:
        import torch
        tokenizer, model = load_local_nllb(progress, log)
        target_tag = "<2hi>"
        batch_size = 8 if DEVICE == "cuda" else 2
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            texts = [f"{target_tag} {s['text']}" for s in batch]
            tokens = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            with torch.inference_mode():
                out = model.generate(
                    **tokens,
                    max_new_tokens=180,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    length_penalty=0.95,
                )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            for s, text in zip(batch, decoded):
                s["translation"] = text.strip()
            progress(
                0.40 + 0.08 * min(1.0, (i + len(batch)) / len(pending)),
                "MADLAD translating…",
            )
        for s in segments:
            memory[normalize_source_text(s["text"])] = s.get(
                "translation", s["text"]
            )
        save_translation_memory(current_project_for_memory(), memory)
        log.append(
            f"✅ MADLAD-400 3B: {source_lang} → Hindi | {len(pending)} new lines"
        )
        return segments
    finally:
        try:
            del model, tokenizer
        except Exception:
            pass
        release_gpu()


QWEN_CACHE = None
QWEN_TOKENIZER = None


def load_qwen(progress, log):
    global QWEN_CACHE, QWEN_TOKENIZER
    if QWEN_CACHE is not None:
        return QWEN_TOKENIZER, QWEN_CACHE
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        progress(0.49, f"Loading local dialogue director {QWEN_REPO}…")
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(
            QWEN_REPO,
            cache_dir=os.path.join(MODEL_CACHE, "qwen"),
        )
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_REPO,
            cache_dir=os.path.join(MODEL_CACHE, "qwen"),
            torch_dtype=dtype,
            device_map="auto" if DEVICE == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if DEVICE != "cuda":
            model = model.to(DEVICE)
        model.eval()
        QWEN_TOKENIZER = tok
        QWEN_CACHE = model
        log.append(f"✅ Local Qwen dialogue director loaded: {QWEN_REPO}")
        return tok, model
    except Exception as e:
        raise RuntimeError(f"Qwen local model load failed: {str(e)[:180]}")


def unload_qwen():
    global QWEN_CACHE, QWEN_TOKENIZER
    QWEN_CACHE = None
    QWEN_TOKENIZER = None
    release_gpu()


def clean_llm_output(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.replace("<|assistant|>", "").strip()
    if "\n" in text:
        text = text.splitlines()[0].strip()
    return text.strip(" \\\"'“”‘’")


def load_character_dialogue_memory(project_id):
    return load_json(
        os.path.join(memory_dir(project_id), "character_dialogue_memory.json"),
        {},
    )


def save_character_dialogue_memory(project_id, memory):
    save_json(
        os.path.join(memory_dir(project_id), "character_dialogue_memory.json"),
        memory,
    )


def qwen_polish(segments, project_id, progress, log, glossary=""):
    tok = model = None
    try:
        tok, model = load_qwen(progress, log)
        import torch
        total = len(segments)
        character_memory = load_character_dialogue_memory(project_id)
        for i, s in enumerate(segments):
            base = s.get("translation", "").strip()
            if not base:
                continue
            prev_text = segments[i - 1]["text"] if i > 0 else ""
            next_text = segments[i + 1]["text"] if i + 1 < total else ""
            duration = max(0.5, float(s["end"] - s["start"]))
            max_words = max(2, min(28, int(duration * 2.8) + 2))
            delivery_load = hindi_delivery_load(base)
            max_phonemes = max(3, min(96, int(duration * 10.0) + 4))
            style = s.get("style", "natural")
            speaker_name = s.get("character_name", "Character")
            cid = s.get("character_id", "UNRESOLVED")
            history = character_memory.get(cid, [])[-4:]
            history_text = " | ".join(
                [str(h.get("hi", "")) for h in history]
            )[:1200]
            prompt = f"""/no_think
You are an anime Hindi dubbing dialogue director.
Rewrite the BASE Hindi line into natural spoken Hindi for voice acting.
Do not change facts, names, intent, threat level, relationship, gendered meaning, or scene context.
Do not add information. Do not explain anything. Output ONLY the final Hindi dialogue.
Keep it short enough to speak naturally inside about {duration:.2f} seconds; target <= {max_words} words and roughly <= {max_phonemes} Hindi phoneme units.
Never sacrifice meaning, names, or emotional intent merely to hit the number.
Preserve anime terms/proper names and emotional intensity.
Character: {speaker_name}
Style: {style}
Character's recent Hindi dialogue: {history_text}
Previous source context: {prev_text}
Current source: {s['text']}
Next source context: {next_text}
BASE Hindi: {base}
Current Hindi delivery load: {delivery_load}
Glossary: {glossary[:1500]}
"""
            messages = [{"role": "user", "content": prompt}]
            try:
                rendered = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ) + "\n/no_think"
            inputs = tok([rendered], return_tensors="pt")
            if DEVICE == "cuda":
                try:
                    model_device = model.device
                except Exception:
                    model_device = next(model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    do_sample=True,
                    repetition_penalty=1.08,
                )
            generated = out[0][inputs["input_ids"].shape[-1]:]
            final = clean_llm_output(
                tok.decode(generated, skip_special_tokens=True)
            )
            if final:
                s["translation"] = final
                character_memory.setdefault(cid, []).append({
                    "source": s.get("text", ""),
                    "hi": final,
                    "emotion": s.get("emotion", "neutral"),
                })
                character_memory[cid] = character_memory[cid][-12:]
            progress(
                0.50 + 0.08 * ((i + 1) / max(1, total)),
                f"Hindi dialogue polish {i+1}/{total}…",
            )
        save_character_dialogue_memory(project_id, character_memory)
        log.append("✅ Qwen Hindi dialogue polish complete + character dialogue memory updated")
        return segments
    except Exception as e:
        log.append(f"⚠️ Qwen polish skipped: {str(e)[:180]}")
        return segments
    finally:
        unload_qwen()

# ---------------------------------------------------------------
# Chatterbox Multilingual V3 TTS
# ---------------------------------------------------------------

CHATTERBOX = None


def load_chatterbox(progress, log):
    global CHATTERBOX
    if CHATTERBOX is not None:
        return CHATTERBOX
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        progress(0.60, "Loading Chatterbox Multilingual V3…")
        model = ChatterboxMultilingualTTS.from_pretrained(
            device=DEVICE,
            t3_model=CHATTERBOX_MODEL,
        )
        if hasattr(model, "to"):
            model.to(DEVICE)
        CHATTERBOX = model
        log.append(
            f"✅ Chatterbox Multilingual {CHATTERBOX_MODEL} loaded on {DEVICE}"
        )
        return CHATTERBOX
    except Exception as e:
        raise RuntimeError(f"Chatterbox load failed: {str(e)[:190]}")


def unload_chatterbox():
    global CHATTERBOX
    CHATTERBOX = None
    release_gpu()


def emotion_params(seg):
    emotion = seg.get("emotion", "neutral")
    table = {
        "neutral": (0.48, 0.00, 0.80),
        "calm": (0.38, -0.05, 0.76),
        "sad": (0.58, -0.10, 0.72),
        "angry": (0.78, 0.08, 0.72),
        "excited": (0.72, 0.10, 0.74),
        "surprised": (0.70, 0.07, 0.74),
        "fearful": (0.65, 0.12, 0.73),
    }
    return table.get(emotion, table["neutral"])


def tts_expressive_settings(seg, attempt=0):
    emotion = seg.get("emotion", "neutral")
    base = {
        "neutral": (0.46, 0.76, 0.94),
        "calm": (0.38, 0.72, 0.97),
        "sad": (0.58, 0.73, 0.99),
        "angry": (0.72, 0.70, 0.92),
        "excited": (0.68, 0.70, 0.90),
        "surprised": (0.64, 0.72, 0.91),
        "fearful": (0.64, 0.73, 0.94),
    }.get(emotion, (0.46, 0.76, 0.94))
    ex, temp, minp = base
    if attempt == 1:
        ex *= 0.92
        temp *= 0.94
    elif attempt >= 2:
        ex *= 0.88
        temp *= 0.90
    return ex, temp, minp


def synthesize_segment(text, reference, seg, seed=1, attempt=0, log=None):
    if not reference:
        raise RuntimeError("Character reference voice missing")
    model = load_chatterbox(lambda *_: None, log or [])
    set_seed(seed)
    exaggeration, temperature, min_p = tts_expressive_settings(seg, attempt)
    if seg.get("events") and "laugh" in seg["events"] and not text.endswith(("!", "!!")):
        text += "!"
    wav = model.generate(
        text[:300],
        language_id="hi",
        audio_prompt_path=reference,
        exaggeration=float(exaggeration),
        cfg_weight=0.0,
        temperature=float(temperature),
        min_p=float(min_p),
        top_p=0.95,
        repetition_penalty=1.18,
    )
    sr = int(model.sr)
    arr = wav.squeeze(0).detach().cpu().numpy().astype("float32")
    out = mktemp_path("_tts.wav", "tts_")
    save_audio(out, arr, sr)
    return out


def atempo_chain(factor):
    factor = float(factor)
    if factor <= 0:
        raise ValueError("Invalid tempo factor")
    parts = []
    while factor > 2.0:
        parts.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        parts.append("atempo=0.5")
        factor /= 0.5
    parts.append(f"atempo={factor:.6f}")
    return ",".join(parts)


def trim_silence_precise(audio_path, keep_head=0.018, keep_tail=0.030):
    out = mktemp_path("_nosilence.wav", "nosil_")
    try:
        run_cmd([
            "ffmpeg", "-y", "-i", audio_path,
            "-af",
            (
                "silenceremove=start_periods=1:start_duration=0.025:"
                "start_threshold=-42dB:stop_periods=1:stop_duration=0.045:"
                "stop_threshold=-42dB"
            ),
            "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", out,
        ])
        return out
    except Exception:
        return audio_path


def fit_audio_strict(audio_path, target_duration, max_tempo_change=1.18):
    cleaned = trim_silence_precise(audio_path)
    src = ffprobe_duration(cleaned)
    target = max(0.12, float(target_duration))
    if src <= 0.01:
        return None, 1.0, src
    factor = src / target
    if factor < 1.0 / max_tempo_change or factor > max_tempo_change:
        return None, factor, src
    out = mktemp_path("_fit.wav", "fit_")
    filt = atempo_chain(factor) + f",apad,atrim=duration={target:.3f}"
    run_cmd([
        "ffmpeg", "-y", "-i", cleaned,
        "-af", filt,
        "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", out,
    ])
    return out, factor, src


def exact_duration_audio(audio_path, target_duration):
    out = mktemp_path("_exact.wav", "exact_")
    run_cmd([
        "ffmpeg", "-y", "-i", audio_path,
        "-af", f"apad,atrim=duration={float(target_duration):.6f}",
        "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", out,
    ])
    return out


def trim_with_fades(y, sr, fade_ms=18):
    n = len(y)
    f = min(int(sr * fade_ms / 1000), max(1, n // 4))
    if f > 1:
        ramp = np.linspace(0.0, 1.0, f, dtype="float32")
        y[:f] *= ramp
        y[-f:] *= ramp[::-1]
    return y


def prosody_vector(path):
    y, sr = load_audio(path, 16000)
    st = audio_stats(y, sr)
    try:
        import librosa
        f0, _, _ = librosa.pyin(
            y,
            fmin=70,
            fmax=500,
            sr=sr,
            frame_length=1024,
            hop_length=256,
        )
        f0 = np.asarray(f0, dtype=np.float32)
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
        med_pitch = float(np.median(f0)) if len(f0) else 0.0
        pitch_spread = float(np.std(f0)) if len(f0) else 0.0
    except Exception:
        med_pitch = float(st.get("pitch", 0.0))
        pitch_spread = 0.0
    return {
        "rms_db": float(st.get("rms_db", -30.0)),
        "pitch": med_pitch,
        "pitch_spread": pitch_spread,
        "zcr": float(st.get("zcr", 0.0)),
    }


def prosody_candidate_score(source_stats, candidate_path, target_duration, factor):
    try:
        c = prosody_vector(candidate_path)
        src_pitch = float(source_stats.get("pitch", 0.0) or 0.0)
        tgt_rms = float(source_stats.get("rms_db", -24.0))
        score = 0.0
        if src_pitch > 0 and c["pitch"] > 0:
            score += max(
                0.0,
                1.0 - abs(np.log((c["pitch"] + 1e-3) / (src_pitch + 1e-3)))
            ) * 0.30
        score += max(
            0.0,
            1.0 - min(1.0, abs(c["rms_db"] - tgt_rms) / 18.0),
        ) * 0.20
        score += max(
            0.0,
            1.0 - min(1.0, abs(np.log(max(factor, 1e-6))) / 0.28),
        ) * 0.35
        score += max(
            0.0,
            1.0 - min(
                1.0,
                abs(ffprobe_duration(candidate_path) - target_duration)
                / max(target_duration, 0.25),
            ),
        ) * 0.15
        return float(score)
    except Exception:
        return 0.0


def voice_identity_candidate_score(profile, candidate_path, log):
    if not VOICE_ID_SCORING_DEFAULT:
        return 0.5
    base = profile.get("embedding") if isinstance(profile, dict) else None
    if not base:
        return 0.5
    try:
        emb = speaker_embedding(candidate_path, log)
        if emb is None:
            return 0.5
        return float(
            max(
                0.0,
                min(1.0, (cosine(emb, np.asarray(base, dtype="float32")) + 1.0) / 2.0),
            )
        )
    except Exception:
        return 0.5


def time_slot_constraint(seg):
    d = max(0.08, float(seg["end"] - seg["start"]))
    budget = hindi_speech_budget(seg.get("translation", ""))
    load = hindi_delivery_load(seg.get("translation", ""))
    return {
        "duration": d,
        "max_words": max(1, min(32, int(d * 3.25) + 2)),
        "max_syllable_proxy": max(2, min(56, int(d * 5.7) + 3)),
        "phonemes": int(load.get("phonemes", budget["syllable_proxy"])),
        "phoneme_method": load.get("method", "fallback"),
        "max_phonemes": max(3, min(96, int(d * 10.0) + 4)),
    }


def _compact_hindi_rule(text, duration, seg):
    t = re.sub(r"\s+", " ", text.strip())
    replacements = [
        ("लेकिन फिर भी", "फिर भी"),
        ("क्योंकि इसी वजह से", "क्योंकि"),
        ("मैं तुम्हें बता रहा हूँ", "मैं बता रहा हूँ"),
        ("तुम लोग सभी", "तुम लोग"),
        ("अभी इसी समय", "अभी"),
        ("यह बात सच है कि", "सच यह है कि"),
        ("मेरे ख्याल से", "मेरे हिसाब से"),
    ]
    for a, b in replacements:
        t = t.replace(a, b)
    return t


def render_dub_track(
    project_id,
    segments,
    total_duration,
    character_map,
    profiles,
    source_audio,
    progress,
    log,
    quality_passes=2,
):
    sr = 24000
    total_n = max(1, int(math.ceil(total_duration * sr)))
    track = np.zeros(total_n, dtype="float32")
    cache_base = (
        os.path.join(PROJECTS_ROOT, safe_project_id(project_id), "dialogue_cache")
        if PERSIST_TTS_CACHE
        else os.path.join(DIALOGUE_CACHE_ROOT, safe_project_id(project_id))
    )
    os.makedirs(cache_base, exist_ok=True)
    load_chatterbox(progress, log)

    for i, s in enumerate(segments):
        text = re.sub(r"\s+", " ", str(s.get("translation", "")).strip())
        if not text:
            continue
        cid = character_map.get(s.get("speaker"))
        if not cid or cid not in profiles:
            s["timing_ok"] = False
            log.append(
                f"❌ SYNC GUARD segment {i+1}: no verified character identity for {s.get('speaker')}"
            )
            continue
        profile = profiles[cid]
        emotion = s.get("emotion", "neutral")
        ref = resolve_reference(
            project_id,
            profile,
            s.get("source_stats"),
            emotion=emotion,
        )
        if not ref:
            s["timing_ok"] = False
            log.append(
                f"❌ {cid}: no verified reference for {emotion}; segment {i+1} blocked"
            )
            continue

        start = max(0.0, float(s["start"]))
        end = min(float(total_duration), float(s["end"]))
        duration = max(0.20, end - start)
        frame = 1.0 / max(float(s.get("frame_fps", LIPSYNC_TARGET_FPS)), 1.0)
        safe_duration = max(frame, duration - 0.5 * frame)
        seed_key = f"{project_id}|{cid}|{emotion}|{i}|{text}"
        seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
        cache_key = hashlib.sha1(
            f"v5|{project_id}|{cid}|{start:.4f}|{end:.4f}|{emotion}|{text}|passes={quality_passes}".encode(
                "utf-8"
            )
        ).hexdigest()
        cache_wav = os.path.join(cache_base, f"{cache_key}.wav")

        try:
            fitted = cache_wav if os.path.exists(cache_wav) else None
            actual_text = text
            actual_factor = 1.0
            source_tts_duration = 0.0
            candidate_records = []
            passes = max(1, min(3, int(quality_passes)))

            if fitted is None:
                budget = time_slot_constraint(s)
                s["hindi_budget"] = budget
                if (
                    budget["words"] > budget["max_words"]
                    or budget["syllable_proxy"] > budget["max_syllable_proxy"]
                ):
                    actual_text = _compact_hindi_rule(
                        actual_text,
                        safe_duration,
                        s,
                    )

                for attempt in range(passes):
                    raw = synthesize_segment(
                        actual_text,
                        ref,
                        s,
                        seed=seed + attempt * 997,
                        attempt=attempt,
                        log=log,
                    )
                    raw = trim_silence_precise(raw)
                    raw_dur = ffprobe_duration(raw)
                    fitted_try, factor, _ = fit_audio_strict(
                        raw,
                        safe_duration,
                        max_tempo_change=max(
                            TTS_MAX_TEMPO_FAST,
                            1.0 / max(TTS_MAX_TEMPO_SLOW, 0.01),
                        ),
                    )
                    if fitted_try:
                        ps = prosody_candidate_score(
                            s.get("source_stats", {}),
                            fitted_try,
                            safe_duration,
                            factor,
                        )
                        ids = voice_identity_candidate_score(
                            profile,
                            fitted_try,
                            log,
                        )
                        score = 0.68 * ps + 0.32 * ids
                        candidate_records.append(
                            (
                                score,
                                fitted_try,
                                factor,
                                raw_dur,
                                attempt,
                                ps,
                                ids,
                            )
                        )

                if candidate_records:
                    candidate_records.sort(key=lambda x: x[0], reverse=True)
                    (
                        _, fitted,
                        actual_factor,
                        source_tts_duration,
                        chosen_attempt,
                        ps,
                        ids,
                    ) = candidate_records[0]
                    log.append(
                        f"🎙️ {cid} {emotion}: candidate {chosen_attempt+1}/{passes} "
                        f"score={candidate_records[0][0]:.3f} prosody={ps:.3f} "
                        f"voiceID={ids:.3f} tempo={actual_factor:.3f}x"
                    )
                    yy, _ = load_audio(fitted, sr)
                    yy = trim_with_fades(yy, sr, fade_ms=9)
                    save_audio(cache_wav, yy, sr)
                    fitted = cache_wav
                else:
                    shorter = _compact_hindi_rule(actual_text, safe_duration, s)
                    if shorter != actual_text:
                        raw = synthesize_segment(
                            shorter,
                            ref,
                            s,
                            seed=seed + 1999,
                            attempt=2,
                            log=log,
                        )
                        raw = trim_silence_precise(raw)
                        fitted_try, factor, raw_dur = fit_audio_strict(
                            raw,
                            safe_duration,
                            max_tempo_change=1.20,
                        )
                        if fitted_try:
                            actual_text = shorter
                            fitted = fitted_try
                            actual_factor = factor
                            source_tts_duration = raw_dur
                            yy, _ = load_audio(fitted, sr)
                            yy = trim_with_fades(yy, sr, fade_ms=9)
                            save_audio(cache_wav, yy, sr)
                            fitted = cache_wav

            if not fitted:
                raise RuntimeError(
                    f"Hindi delivery cannot safely fit source frame slot {duration:.3f}s"
                )

            yy, _ = load_audio(fitted, sr)
            a = max(0, int(round(start * sr)))
            slot_len = max(1, int(round(duration * sr)))
            yy = np.pad(yy, (0, max(0, slot_len - len(yy))))[:slot_len]
            b = min(total_n, a + len(yy))
            if b > a:
                peak = float(np.max(np.abs(yy))) if len(yy) else 0.0
                if peak > 0.98:
                    yy *= 0.98 / peak
                track[a:b] += yy[: b - a]

            s["audio_cache"] = os.path.relpath(cache_wav, memory_dir(project_id))
            s["tts_duration"] = float(len(yy) / sr)
            s["source_tts_duration"] = float(source_tts_duration)
            s["timing_factor"] = float(actual_factor)
            s["rendered_text"] = actual_text
            s["timing_ok"] = True
            s["cue_start"] = start
            s["cue_end"] = end
            s["cue_duration"] = duration
            s["frame_locked"] = True
        except Exception as e:
            s["timing_ok"] = False
            log.append(
                f"❌ SYNC GUARD segment {i+1} [{cid}] {emotion}: {str(e)[:190]}"
            )

        progress(
            0.60 + 0.15 * ((i + 1) / max(1, len(segments))),
            f"Voice + frame-locked timing {i+1}/{len(segments)}…",
        )

    peak = float(np.max(np.abs(track))) if len(track) else 0.0
    if peak > 0.94:
        track *= 0.94 / peak
    out = mktemp_path("_dub_dialogue_24k.wav", "dubtrack_")
    save_audio(out, track, sr)
    return out


def preserve_accompaniment_mix(
    dub_path,
    accomp_path,
    original_audio_path,
    vocals_path,
    duck_mode,
    duck_strength,
    segments,
    progress,
    log,
):
    sr = 24000
    if duck_mode == "Original residual (maximum fidelity)":
        residual = build_original_residual_bed(
            original_audio_path,
            vocals_path,
            progress,
            log,
        )
        if residual:
            accomp_path = residual

    dub, _ = load_audio(dub_path, sr)
    total_duration = ffprobe_duration(original_audio_path)
    total_n = max(1, int(math.ceil(total_duration * sr)))
    dub = np.pad(dub, (0, max(0, total_n - len(dub))))[:total_n]

    if not accomp_path or not os.path.exists(accomp_path):
        log.append("⚠️ No accompaniment stem; returning dialogue-only mix.")
        out = mktemp_path("_dialogue_only.wav", "mix_")
        save_audio(out, dub, sr)
        return out

    accomp, _ = load_audio(accomp_path, sr)
    accomp = np.pad(accomp, (0, max(0, total_n - len(accomp))))[:total_n]

    gain = np.ones(total_n, dtype="float32")
    if duck_mode == "Smart duck":
        lowered = max(0.25, 1.0 - float(duck_strength))
        fade = int(0.06 * sr)
        for s in segments:
            a = max(0, int(s["start"] * sr))
            b = min(total_n, int(s["end"] * sr))
            if b <= a:
                continue
            left = max(0, a - fade)
            right = min(total_n, b + fade)
            if a > left:
                gain[left:a] = np.linspace(1.0, lowered, a - left, dtype="float32")
            gain[a:b] = lowered
            if right > b:
                gain[b:right] = np.linspace(lowered, 1.0, right - b, dtype="float32")
        log.append(f"✅ Smart dialogue ducking: {lowered:.2f} bed gain during speech")
    else:
        log.append("✅ Background bed kept at source gain; no global BGM/SFX normalization")

    mixed = accomp * gain + 0.95 * dub
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 0.99:
        mixed *= 0.985 / peak
        log.append("ℹ️ Final limiter safety reduction applied to avoid clipping")
    out = mktemp_path("_final_mix.wav", "mix_")
    save_audio(out, mixed, sr)
    return out

# ---------------------------------------------------------------
# Subtitles / manifest / mux
# ---------------------------------------------------------------

def tc_srt(sec):
    sec = max(0.0, float(sec))
    base = int(sec)
    ms = int(round((sec - base) * 1000))
    if ms >= 1000:
        base += 1
        ms = 0
    h = base // 3600
    m = (base % 3600) // 60
    s = base % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(project_id, segments, character_names):
    out = os.path.join(
        RESULT_ROOT,
        f"{safe_project_id(project_id)}_dubbed_hindi.srt",
    )
    rows = []
    for i, s in enumerate(segments, 1):
        cid = s.get("character_id", "UNRESOLVED")
        name = character_names.get(cid, cid)
        rows.extend([
            str(i),
            f"{tc_srt(s['start'])} --> {tc_srt(s['end'])}",
            f"[{name}] {s.get('translation', '').strip()}",
            "",
        ])
    Path(out).write_text("\n".join(rows), encoding="utf-8")
    return out


def write_ass(project_id, segments, character_names):
    out = os.path.join(
        RESULT_ROOT,
        f"{safe_project_id(project_id)}_dubbed_hindi.ass",
    )
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,52,&H00FFFFFF,&H00FFFFFF,&H00000000,&H66000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,50,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    def ass_tc(x):
        x = max(0.0, float(x))
        h = int(x // 3600)
        m = int((x % 3600) // 60)
        s = int(x % 60)
        cs = int(round((x - int(x)) * 100))
        if cs >= 100:
            s += 1
            cs = 0
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    for s in segments:
        cid = s.get("character_id", "UNRESOLVED")
        name = character_names.get(cid, cid)
        text = (
            s.get("translation", "")
            .replace("{", "\\{")
            .replace("}", "\\}")
        )
        lines.append(
            f"Dialogue: 0,{ass_tc(s['start'])},{ass_tc(s['end'])},Default,{name},0,0,0,,{text}"
        )
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    return out


def mux_final_video(video, audio, subtitle_path=None, hardcode_subs=False, lipsync_video=None):
    require_ffmpeg()
    source_video = lipsync_video or video
    temp_out = mktemp_path("_mux.mp4", "mux_")
    if hardcode_subs and subtitle_path:
        run_cmd([
            "ffmpeg", "-y",
            "-i", source_video,
            "-i", audio,
            "-vf", f"subtitles={subtitle_path}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "320k", "-shortest", temp_out,
        ])
        return temp_out

    if lipsync_video is None:
        try:
            run_cmd([
                "ffmpeg", "-y",
                "-i", source_video,
                "-i", audio,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "320k",
                "-shortest", temp_out,
            ])
            return temp_out
        except Exception:
            pass

    run_cmd([
        "ffmpeg", "-y",
        "-i", source_video,
        "-i", audio,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "320k", "-shortest", temp_out,
    ])
    return temp_out


def sync_locked_final_audio(final_audio, video, segments, log):
    video_dur = ffprobe_duration(video)
    exact = exact_duration_audio(final_audio, video_dur)
    check = validate_episode_sync(
        video,
        exact,
        segments,
        tolerance_ms=LIPSYNC_MAX_DRIFT_MS,
    )
    if not check["duration_ok"]:
        raise RuntimeError(
            f"Audio/video duration lock failed: {check['duration_error_ms']:.1f} ms"
        )
    return exact, check


def validate_episode_sync(video, final_audio, segments, tolerance_ms=45):
    video_dur = ffprobe_duration(video)
    audio_dur = ffprobe_duration(final_audio)
    duration_error_ms = abs(video_dur - audio_dur) * 1000.0
    bad = [s for s in segments if not s.get("timing_ok", True)]
    return {
        "video_duration": video_dur,
        "audio_duration": audio_dur,
        "duration_error_ms": duration_error_ms,
        "duration_ok": duration_error_ms <= tolerance_ms,
        "bad_segments": len(bad),
        "segment_timing_ok": len(bad) == 0,
        "safe_to_mux": duration_error_ms <= tolerance_ms and len(bad) == 0,
    }

# ---------------------------------------------------------------
# Lip-sync / SyncNet gate
# ---------------------------------------------------------------

SYNCNET_MODEL = None


def _get_syncnet(log):
    global SYNCNET_MODEL
    if SYNCNET_MODEL is not None:
        return SYNCNET_MODEL
    if not os.path.exists(SYNCNET_WEIGHTS):
        log.append(
            "ℹ️ SyncNet gate unavailable: set SYNCNET_WEIGHTS to a local pretrained model."
        )
        return None
    try:
        from syncnet_python import SyncNetInstance
        model = SyncNetInstance()
        model.loadParameters(SYNCNET_WEIGHTS)
        SYNCNET_MODEL = model
        return model
    except Exception as e:
        log.append(f"⚠️ SyncNet load failed: {str(e)[:160]}")
        return None


def _syncnet_score(video_path, audio_path=None, log=None):
    log = log or []
    syncnet = _get_syncnet(log)
    if syncnet is None:
        return None
    try:
        class Args:
            tmp_dir = os.path.join(SCRATCH_ROOT, "syncnet_tmp")
            reference = "anime_dub"
            batch_size = 20
            vshift = 10
        os.makedirs(Args.tmp_dir, exist_ok=True)
        probe = video_path
        muxed = None
        if audio_path:
            muxed = mktemp_path("_syncnet_probe.mp4", "syncmux_")
            run_cmd([
                "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", muxed,
            ])
            probe = muxed
        offset, conf, _ = syncnet.evaluate(Args(), probe)
        log.append(
            f"🔎 SyncNet: offset={int(offset)} frame(s), confidence={float(conf):.3f}"
        )
        if muxed:
            try:
                os.remove(muxed)
            except Exception:
                pass
        return int(offset), float(conf)
    except Exception as e:
        log.append(f"⚠️ SyncNet validator unavailable: {str(e)[:160]}")
        return None


def _syncnet_accept(result):
    if result is None:
        return not SYNCNET_REQUIRED_DEFAULT
    offset, conf = result
    return (
        abs(offset) <= LIPSYNC_MAX_OFFSET_FRAMES
        and conf >= LIPSYNC_MIN_CONFIDENCE
    )


def make_frame_locked_dialogue_audio(dub_path, total_duration, fps):
    out = mktemp_path("_dialogue_frame_locked.wav", "frameaudio_")
    run_cmd([
        "ffmpeg", "-y", "-i", dub_path,
        "-af", f"apad,atrim=duration={float(total_duration):.6f}",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out,
    ])
    return out


def shift_audio_by_frames(audio_path, fps, frames, total_duration):
    shift = int(frames)
    out = mktemp_path(f"_offset_{shift:+d}f.wav", "offset_")
    if shift == 0:
        shutil.copy2(audio_path, out)
        return out
    sec = abs(shift) / max(fps, 1e-6)
    if shift > 0:
        af = (
            f"adelay={int(round(sec*1000))}:all=1,"
            f"atrim=duration={float(total_duration):.6f}"
        )
    else:
        af = (
            f"atrim=start={sec:.6f},apad,"
            f"atrim=duration={float(total_duration):.6f}"
        )
    run_cmd([
        "ffmpeg", "-y", "-i", audio_path,
        "-af", af,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out,
    ])
    return out


def prepare_visual_lipsync_source(video, total_duration):
    meta = probe_video_timing(video)
    fps = float(meta.get("fps") or LIPSYNC_TARGET_FPS)
    out = mktemp_path("_lipsync_cfr.mp4", "lipproxy_")
    try:
        run_cmd([
            "ffmpeg", "-y", "-i", video,
            "-vf", f"fps=fps={fps:.6f}",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
            "-pix_fmt", "yuv420p", out,
        ])
        probe = probe_video_timing(out)
        log_msg = (
            f"Frame-stable lip-sync proxy: source={meta['fps']:.3f}fps → "
            f"CFR={probe['fps']:.3f}fps"
        )
        return out, {
            "fps": probe["fps"],
            "frames": probe["frames"],
            "duration": probe["duration"],
            "log": log_msg,
        }
    except Exception:
        try:
            os.remove(out)
        except Exception:
            pass
        return video, meta


def visual_lipsync_preflight(video, dialogue_audio, segments, log):
    fps_meta = probe_video_timing(video)
    bad = []
    for s in segments:
        if not s.get("timing_ok", True):
            bad.append(s)
            continue
        a = float(s.get("start", 0.0))
        b = float(s.get("end", a))
        if b <= a or a < 0 or b > fps_meta["duration"] + 0.05:
            bad.append(s)
    if bad:
        raise RuntimeError(
            f"Visual lip-sync preflight blocked: {len(bad)} invalid dialogue cue(s)"
        )
    ad = ffprobe_duration(dialogue_audio)
    vd = ffprobe_duration(video)
    drift = abs(ad - vd) * 1000.0
    if drift > LIPSYNC_MAX_DRIFT_MS:
        raise RuntimeError(
            f"Visual lip-sync preflight blocked: dialogue audio drift={drift:.1f}ms"
        )
    log.append(
        f"🛡️ Lip-sync preflight PASS • {fps_meta['fps']:.3f}fps • "
        f"{fps_meta['frames']} frames • dialogue audio drift={drift:.1f}ms"
    )
    return fps_meta


def run_wav2lip(video, dialogue_audio, progress, log, checkpoint=None, fps=25.0):
    inference = os.path.join(WAV2LIP_DIR, "inference.py")
    checkpoint = checkpoint or WAV2LIP_GAN_CHECKPOINT
    if not os.path.exists(inference) or not os.path.exists(checkpoint):
        log.append("⚠️ Wav2Lip skipped: local repo/checkpoint not found.")
        return None
    progress(0.90, "Frame-accurate visual lip-sync…")
    out = mktemp_path("_lipsync.mp4", "lipsync_")
    try:
        run_cmd([
            sys.executable,
            inference,
            "--checkpoint_path", checkpoint,
            "--face", video,
            "--audio", dialogue_audio,
            "--outfile", out,
            "--pads", "0", "18", "0", "0",
            "--face_det_batch_size", "32",
            "--wav2lip_batch_size", "64",
            "--fps", str(fps),
            "--nosmooth",
        ], quiet=False)
        if os.path.exists(out):
            src_d = ffprobe_duration(video)
            out_d = ffprobe_duration(out)
            drift = abs(src_d - out_d) * 1000.0
            if drift <= LIPSYNC_MAX_DRIFT_MS:
                log.append(
                    f"✅ Wav2Lip frame render complete • duration drift {drift:.1f}ms"
                )
                return out
            log.append(f"🛑 Wav2Lip rejected: duration drift {drift:.1f}ms")
    except Exception as e:
        log.append(f"⚠️ Wav2Lip error: {str(e)[:180]}")
    return None


def syncnet_sliding_validate(video_path, dialogue_audio, segments, fps, log, strict=True):
    syncnet = _get_syncnet(log)
    if syncnet is None:
        # Strict QA means "fail if the available SyncNet gate fails".
        # When weights are not installed, there is no gate to run, so the
        # renderer remains usable but the report explicitly says unverified.
        return {"available": False, "verified": True, "unverified": True, "windows": []}
    dur = ffprobe_duration(video_path)
    windows = []
    for s in sorted(segments, key=lambda x: float(x.get("start", 0.0))):
        a = max(0.0, float(s.get("start", 0.0)) - 0.35)
        b = min(dur, float(s.get("end", a)) + 0.35)
        if b - a < 0.75:
            continue
        if windows and a <= windows[-1][1] + 0.15:
            windows[-1] = (windows[-1][0], max(windows[-1][1], b))
        else:
            windows.append((a, b))

    split = []
    for a, b in windows:
        cur = a
        while cur < b:
            end = min(b, cur + SYNCNET_WINDOW_SEC)
            if end - cur >= 0.75:
                split.append((cur, end))
            cur += SYNCNET_WINDOW_STEP_SEC
            if cur >= b:
                break

    if len(split) > 48:
        idx = np.linspace(0, len(split) - 1, 48).astype(int).tolist()
        split = [split[i] for i in idx]

    reports = []
    all_ok = True
    for i, (a, b) in enumerate(split):
        vclip = mktemp_path("_syncwin.mp4", "syncwin_")
        aclip = mktemp_path("_syncwin.wav", "syncaudio_")
        try:
            run_cmd([
                "ffmpeg", "-y",
                "-ss", f"{a:.3f}", "-i", video_path,
                "-t", f"{b-a:.3f}",
                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", vclip,
            ])
            run_cmd([
                "ffmpeg", "-y",
                "-ss", f"{a:.3f}", "-i", dialogue_audio,
                "-t", f"{b-a:.3f}",
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", aclip,
            ])
            result = _syncnet_score(vclip, aclip, log)
            ok = _syncnet_accept(result)
            reports.append({"start": a, "end": b, "result": result, "ok": ok})
            if not ok:
                all_ok = False
        except Exception as e:
            reports.append({
                "start": a,
                "end": b,
                "result": None,
                "ok": False,
                "error": str(e)[:120],
            })
            all_ok = False
        finally:
            for f in (vclip, aclip):
                try:
                    os.remove(f)
                except Exception:
                    pass
        log.append(
            f"🧪 SyncNet window {i+1}/{len(split)}: "
            f"{'PASS' if reports[-1]['ok'] else 'FAIL'}"
        )
    return {
        "available": True,
        "verified": all_ok,
        "windows": reports,
        "window_count": len(reports),
    }


def run_visual_lipsync(
    video,
    dialogue_audio,
    final_audio,
    engine,
    progress,
    log,
    total_duration,
    segments=None,
    strict_syncnet=True,
):
    engine = (engine or "Anime-safe timing lock").strip()
    if engine == "Anime-safe timing lock":
        log.append(
            "✅ Visual edits disabled; original frames preserved, audio remains frame-locked."
        )
        return None, {
            "engine": engine,
            "verified": True,
            "visual_changed": False,
        }

    visual_lipsync_preflight(video, dialogue_audio, segments or [], log)
    visual_input, meta = prepare_visual_lipsync_source(video, total_duration)
    log.append(meta.get("log", f"Lip-sync FPS={meta['fps']:.3f}"))
    dialogue_16k = make_frame_locked_dialogue_audio(
        dialogue_audio,
        total_duration,
        meta.get("fps", LIPSYNC_TARGET_FPS),
    )

    if engine.startswith("Wav2Lip"):
        base = run_wav2lip(
            visual_input,
            dialogue_16k,
            progress,
            log,
            checkpoint=WAV2LIP_GAN_CHECKPOINT,
            fps=meta.get("fps", LIPSYNC_TARGET_FPS),
        )
        if not base:
            return None, {
                "engine": engine,
                "verified": False,
                "reason": "renderer unavailable/rejected",
            }

        sync = _syncnet_score(base, dialogue_16k, log)
        if sync is not None and not _syncnet_accept(sync):
            offset, conf = sync
            log.append(
                f"⚙️ Sync correction requested: {offset:+d} frame(s), confidence={conf:.3f}"
            )
            correction = int(
                np.clip(
                    offset,
                    -LIPSYNC_MAX_OFFSET_FRAMES,
                    LIPSYNC_MAX_OFFSET_FRAMES,
                )
            )
            if correction != 0:
                shifted = shift_audio_by_frames(
                    dialogue_16k,
                    meta.get("fps", LIPSYNC_TARGET_FPS),
                    correction,
                    total_duration,
                )
                corrected = run_wav2lip(
                    visual_input,
                    shifted,
                    progress,
                    log,
                    checkpoint=WAV2LIP_GAN_CHECKPOINT,
                    fps=meta.get("fps", LIPSYNC_TARGET_FPS),
                )
                if corrected:
                    sync2 = _syncnet_score(corrected, shifted, log)
                    if sync2 is not None and not _syncnet_accept(sync2):
                        log.append(
                            "🛑 SyncNet still below gate after correction; original visual frames retained."
                        )
                        return None, {
                            "engine": engine,
                            "verified": False,
                            "sync_initial": sync,
                            "sync_after": sync2,
                        }
                    if sync2 is not None:
                        log.append(f"✅ Frame offset corrected: {sync2[0]:+d} frame(s)")
                    base = corrected

        drift = abs(ffprobe_duration(video) - ffprobe_duration(base)) * 1000.0
        verified = drift <= LIPSYNC_MAX_DRIFT_MS
        sync_windows = syncnet_sliding_validate(
            base,
            dialogue_16k,
            segments or [],
            meta.get("fps", LIPSYNC_TARGET_FPS),
            log,
            strict=strict_syncnet,
        )
        if not verified:
            log.append(
                f"🛑 Visual lipsync rejected by final duration gate: {drift:.1f}ms"
            )
            return None, {
                "engine": engine,
                "verified": False,
                "duration_drift_ms": drift,
                "syncnet": sync_windows,
            }
        if (
            strict_syncnet
            and sync_windows.get("available", False)
            and not sync_windows.get("verified", False)
        ):
            log.append(
                "🛑 Visual lip-sync rejected: one or more local SyncNet QA windows failed."
            )
            return None, {
                "engine": engine,
                "verified": False,
                "duration_drift_ms": drift,
                "syncnet": sync_windows,
            }
        return base, {
            "engine": engine,
            "verified": True,
            "duration_drift_ms": drift,
            "syncnet": sync_windows,
        }

    if engine.startswith("MuseTalk"):
        repo = os.environ.get(
            "MUSETALK_DIR",
            os.path.join(MODEL_CACHE, "MuseTalk"),
        )
        script = os.environ.get(
            "MUSETALK_INFER",
            os.path.join(repo, "scripts", "inference.py"),
        )
        if not os.path.exists(script):
            log.append(
                "⚠️ MuseTalk 1.5 local checkout not found; visual render blocked."
            )
            return None, {
                "engine": engine,
                "verified": False,
                "reason": "MuseTalk checkout/config missing",
            }
        log.append(
            "ℹ️ MuseTalk 1.5 is environment-specific; configure local model paths before enabling it."
        )
        return None, {
            "engine": engine,
            "verified": False,
            "reason": "explicit local config required",
        }

    return None, {
        "engine": engine,
        "verified": False,
        "reason": "unknown renderer",
    }

# ---------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------

def copy_result(src, project_id, suffix, extension=None):
    project = safe_project_id(project_id)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ext = extension or Path(src).suffix
    out = os.path.join(
        RESULT_ROOT,
        f"{project}_{stamp}_{suffix}{ext}",
    )
    shutil.copy2(src, out)
    return out


def output_json_manifest(project_id):
    return os.path.join(
        memory_dir(project_id),
        "episode_manifest.json",
    )

# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------

def full_pipeline(
    video_file,
    project_id,
    model_size,
    voice_match_threshold,
    reuse_voicebank,
    deep_hindi_polish,
    preserve_bed_mode,
    duck_strength,
    lip_sync,
    hardcode_subtitles,
    min_speakers,
    max_speakers,
    glossary,
    manual_translation,
    tts_passes,
    voice_identity_rerank,
    strict_syncnet,
    progress=gr.Progress(),
):
    global CURRENT_PROJECT_ID, VOICE_ID_SCORING_DEFAULT
    CURRENT_PROJECT_ID = safe_project_id(project_id)
    VOICE_ID_SCORING_DEFAULT = bool(voice_identity_rerank)

    dia_min = safe_int(min_speakers, default=None, minimum=1, maximum=64)
    dia_max = safe_int(max_speakers, default=16, minimum=1, maximum=64)
    if dia_min is not None and dia_min > dia_max:
        dia_min = 1

    quality_passes = safe_int(tts_passes, default=TTS_NATURALNESS_PASSES, minimum=1, maximum=3)
    match_threshold = safe_float(voice_match_threshold, default=0.79, minimum=0.50, maximum=0.98)
    duck_value = safe_float(duck_strength, default=0.35, minimum=0.0, maximum=0.75)

    log = [
        f"🚀 {APP_TITLE}",
        f"🖥️ {DEVICE_LABEL}",
        f"📁 Project: {CURRENT_PROJECT_ID}",
        "🎯 Target: Hindi",
        "🔒 Local inference: YES after model cache is populated",
        "💾 Persistent: /kaggle/working/anime_dub_studio",
        "⚡ Scratch: /kaggle/tmp/anime_dub_studio",
        "📥 Gradio uploads: scratch/tmp/<gradio-folder>",
        "🧪 Pipeline temp: scratch/tmp/pipeline",
        f"🎭 Character bank capacity: {MAX_CHARACTERS_DEFAULT}",
        f"👥 Diarization source-speaker limit: {dia_max}",
        f"🧹 Persistent TTS cache: {'ON' if PERSIST_TTS_CACHE else 'OFF'}",
    ]

    for label, path in [
        ("working", "/kaggle/working"),
        ("scratch", SCRATCH_ROOT),
    ]:
        rep = disk_report(path)
        if rep:
            log.append(
                f"💽 {label}: {rep['free_gb']:.2f} GB free / {rep['total_gb']:.2f} GB"
            )

    video_file = normalize_path(video_file)
    if not video_file:
        return None, None, None, None, None, None, "❌ Video upload करें।"

    video_mp4 = None
    original_audio = None
    vocals = None
    accomp = None
    final_audio = None
    final_video = None

    try:
        # Remove stale Demucs/FFmpeg intermediates from a previous failed run.
        cleanup_intermediates()
        prune_old_results()
        require_ffmpeg()

        progress(0.015, "Staging uploaded source into scratch…")
        staged_input = stage_input_video(video_file)
        log.append(
            f"✅ Source staged safely in scratch • "
            f"{os.path.getsize(staged_input) / 1024**2:.1f} MB"
        )

        progress(0.02, "Preparing source…")
        video_mp4 = prepare_video(staged_input)
        original_audio = extract_audio(video_mp4)
        total_duration = ffprobe_duration(original_audio)
        log.append(f"✅ Source prepared • {total_duration/60:.2f} min")

        if total_duration <= 0.05:
            raise RuntimeError("Source video duration is zero/too short.")

        # 1) Separation
        vocals, accomp = separate_dialogue(original_audio, progress, log)

        # 2) ASR/alignment/cue refinement
        source_lang, segments = transcribe_whisperx(
            vocals,
            model_size,
            progress,
            log,
        )
        if not segments:
            raise RuntimeError("कोई dialogue detect नहीं हुआ।")

        segments = lock_cues_to_aligned_words(segments, log)
        segments = refine_all_cues(vocals, segments, progress, log)
        segments, video_meta = snap_segments_to_frame_grid(
            video_mp4,
            segments,
            log,
        )

        # 3) Diarization
        turns = diarize(
            vocals,
            progress,
            log,
            min_speakers=dia_min,
            max_speakers=dia_max,
        )
        segments = assign_speakers(segments, turns)
        segments = classify_emotions(
            vocals,
            source_lang,
            segments,
            progress,
            log,
            use_model=True,
        )
        unload_sensevoice()

        # 4) Persistent character identity
        effective_threshold = match_threshold if bool(reuse_voicebank) else 1.10
        char_map, match_report = build_voice_bank(
            CURRENT_PROJECT_ID,
            turns,
            vocals,
            effective_threshold,
            progress,
            log,
            segments=segments,
            max_characters=MAX_CHARACTERS_DEFAULT,
        )
        unload_speaker_encoder()

        profiles = {
            p["id"]: p for p in list_profiles(CURRENT_PROJECT_ID)
        }
        for s in segments:
            cid = char_map.get(s.get("speaker"))
            if not cid:
                s["character_id"] = "UNRESOLVED"
                s["character_name"] = "UNRESOLVED"
                s["identity_ok"] = False
                continue
            s["character_id"] = cid
            s["character_name"] = profiles.get(cid, {}).get("name", cid)
            s["style"] = f"{s.get('emotion', 'neutral')} delivery"
            s["identity_ok"] = True

        # 5) Translation
        progress(0.36, f"Translating {source_lang} → Hindi…")
        manual_translation = str(manual_translation or "").strip()
        if manual_translation:
            lines = [
                x.strip()
                for x in manual_translation.splitlines()
                if x.strip()
            ]
            if len(lines) == len(segments):
                for s, line in zip(segments, lines):
                    s["translation"] = line
                log.append("✅ Manual line-by-line Hindi translation used")
            else:
                log.append(
                    "⚠️ Manual translation line count differs; using MADLAD instead"
                )
                nllb_translate(
                    segments,
                    source_lang,
                    "Hindi",
                    progress,
                    log,
                )
        else:
            nllb_translate(
                segments,
                source_lang,
                "Hindi",
                progress,
                log,
            )

        # 6) Local dialogue polish
        if bool(deep_hindi_polish):
            segments = qwen_polish(
                segments,
                CURRENT_PROJECT_ID,
                progress,
                log,
                glossary=str(glossary or ""),
            )

        # 7) TTS
        dub_track = render_dub_track(
            CURRENT_PROJECT_ID,
            segments,
            total_duration,
            char_map,
            profiles,
            vocals,
            progress,
            log,
            quality_passes=quality_passes,
        )
        unload_speaker_encoder()
        unload_chatterbox()

        # 8) Preserve/reconstruct BGM + SFX bed
        final_audio_tmp = preserve_accompaniment_mix(
            dub_track,
            accomp,
            original_audio,
            vocals,
            preserve_bed_mode,
            duck_value,
            segments,
            progress,
            log,
        )

        # 9) Exact audio/video duration lock
        final_audio_exact, sync_report = sync_locked_final_audio(
            final_audio_tmp,
            video_mp4,
            segments,
            log,
        )
        log.append(
            f"🔒 Sync lock: audio/video error={sync_report['duration_error_ms']:.1f}ms "
            f"• bad segments={sync_report['bad_segments']}"
        )
        if not sync_report["segment_timing_ok"]:
            raise RuntimeError(
                "One or more dialogue segments failed the exact timing guard; video export stopped."
            )

        # Result copies are in /kaggle/tmp, not persistent working storage.
        final_audio = copy_result(
            final_audio_exact,
            CURRENT_PROJECT_ID,
            "final_audio",
            ".wav",
        )

        # 10) Subtitles
        names = {
            cid: p.get("name", cid)
            for cid, p in profiles.items()
        }
        srt = write_srt(
            CURRENT_PROJECT_ID,
            segments,
            names,
        )
        ass = write_ass(
            CURRENT_PROJECT_ID,
            segments,
            names,
        )

        lip_report = {
            "engine": str(lip_sync),
            "verified": True,
            "visual_changed": False,
        }

        # 11) Optional visual lipsync using dialogue-only audio
        lip_video = None
        if bool(lip_sync):
            lip_video, lip_report = run_visual_lipsync(
                video_mp4,
                dub_track,
                final_audio_exact,
                lip_sync,
                progress,
                log,
                total_duration,
                segments=segments,
                strict_syncnet=bool(strict_syncnet),
            )
            if str(lip_sync) != "Anime-safe timing lock" and not lip_report.get("verified", False):
                raise RuntimeError(
                    "Visual lip-sync verification failed; final video export stopped."
                )

        # 12) Mux video
        progress(0.94, "Muxing final video…")
        muxed_tmp = mux_final_video(
            video_mp4,
            final_audio_exact,
            subtitle_path=ass if bool(hardcode_subtitles) else None,
            hardcode_subs=bool(hardcode_subtitles),
            lipsync_video=lip_video,
        )
        final_video = copy_result(
            muxed_tmp,
            CURRENT_PROJECT_ID,
            "final_hindi_dub",
            ".mp4",
        )

        # 13) Manifest (persistent small metadata only)
        manifest = {
            "app": APP_TITLE,
            "created": datetime.utcnow().isoformat() + "Z",
            "project": CURRENT_PROJECT_ID,
            "source_language": source_lang,
            "target_language": "hi",
            "duration": total_duration,
            "segments": segments,
            "speaker_to_character": char_map,
            "characters": profiles,
            "voice_match_threshold": float(match_threshold),
            "audio_bed_mode": preserve_bed_mode,
            "lip_sync": str(lip_sync),
            "lip_sync_report": lip_report,
            "sync_report": sync_report,
            "frame_sync_fps": video_meta.get("fps"),
            "frame_sync_policy": (
                "word-aligned + VAD-refined cue boundaries snapped to source frame grid; "
                "dialogue-only audio drives visual renderer; local sliding SyncNet QA gate"
            ),
            "sync_policy": "hard cue lock; failed segments block final export",
            "emotion_engine": "SenseVoiceSmall + acoustic prosody + emotion-matched reference bank",
            "character_memory": os.path.join(
                memory_dir(CURRENT_PROJECT_ID),
                "character_dialogue_memory.json",
            ),
            "max_characters": MAX_CHARACTERS_DEFAULT,
            "max_diarization_speakers": dia_max,
            "storage": {
                "persistent_root": PERSIST_ROOT,
                "scratch_root": SCRATCH_ROOT,
                "model_cache": MODEL_CACHE,
                "persistent_tts_cache": PERSIST_TTS_CACHE,
            },
        }
        manifest_path = output_json_manifest(CURRENT_PROJECT_ID)
        save_json(manifest_path, manifest)

        bank_zip = export_voicebank(CURRENT_PROJECT_ID)

        transcript_text = "\n".join(
            f"[{format_time(s['start'])}–{format_time(s['end'])}] "
            f"[{s.get('character_name', s.get('speaker'))}] {s['text']}"
            for s in segments
        )
        translation_text = "\n".join(
            f"[{format_time(s['start'])}–{format_time(s['end'])}] "
            f"[{s.get('character_name', s.get('speaker'))}] {s.get('translation', '')}"
            for s in segments
        )
        voice_report = "\n".join(match_report)

        log.append(
            f"✅ Characters in episode: {len(set(char_map.values()))}"
        )
        log.append(f"✅ Voice bank: {bank_dir(CURRENT_PROJECT_ID)}")
        log.append(
            "✅ Next episode: same project ID + immutable character voice bank will auto-match speakers"
        )
        log.append(
            "⚠️ Fidelity note: exact actor identity / bit-perfect separated BGM cannot be guaranteed by open-source models."
        )
        log.append(
            f"💾 Final files are in scratch results: {RESULT_ROOT}"
        )
        progress(1.0, "✅ Dubbing complete")
        cleanup_intermediates()

        return (
            transcript_text,
            translation_text,
            final_audio,
            final_video,
            srt,
            bank_zip,
            voice_report + "\n\n" + "\n".join(log),
        )

    except Exception as e:
        unload_sensevoice()
        unload_speaker_encoder()
        unload_qwen()
        unload_chatterbox()
        cleanup_intermediates()
        log.append(f"❌ ERROR: {str(e)}")
        return None, None, None, None, None, None, "\n".join(log)


def format_time(sec):
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


def voicebank_status(project_id):
    project_id = safe_project_id(project_id)
    profiles = list_profiles(project_id)
    if not profiles:
        return f"### 🎭 Voice Bank\nNo saved characters for `{project_id}` yet."
    rows = [
        f"### 🎭 Voice Bank — `{project_id}`",
        "",
    ]
    for p in profiles:
        rows.append(
            f"- **{p.get('id')} → {p.get('name')}** • "
            f"episodes: {p.get('episodes_seen', 0)} • "
            f"refs: {len(p.get('references', []))}"
        )
    return "\n".join(rows)


def storage_status():
    rows = ["### 💾 Storage", ""]
    for label, path in [
        ("Persistent /kaggle/working", "/kaggle/working"),
        ("Scratch /kaggle/tmp", SCRATCH_ROOT),
    ]:
        rep = disk_report(path)
        if rep:
            rows.append(
                f"- **{label}**: {rep['free_gb']:.2f} GB free / {rep['total_gb']:.2f} GB total"
            )
    rows.append("")
    rows.append(f"- Models: `{MODEL_CACHE}`")
    rows.append(f"- Voice Bank: `{VOICEBANK_ROOT}`")
    rows.append(f"- Results: `{RESULT_ROOT}`")
    return "\n".join(rows)

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------

with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🎌 {APP_TITLE}")
    gr.Markdown(
        "**Japanese/English → Hindi • persistent character voice bank • emotion/prosody-aware cloning • frame-grid dialogue lock • dialogue-only lip-sync driver • BGM/SFX preservation • Kaggle GPU • storage-safe scratch architecture**"
    )
    gr.Markdown(
        f"> 🖥️ **{DEVICE_LABEL}**  |  🔐 HF_TOKEN केवल gated pyannote/model access के लिए  |  🚫 gTTS / external TTS API नहीं  |  🔒 exact cue timing guard ON"
    )

    with gr.Row():
        with gr.Column(scale=1):
            project = gr.Textbox(
                label="📁 Anime / Project ID",
                value="my_anime",
                info="Episode 1 और आगे के सभी episodes में वही ID रखें।",
            )
            video_in = gr.Video(
                label="🎬 Anime Episode (MP4 / MKV / AVI / MOV)",
            )
            model_size = gr.Dropdown(
                choices=["small", "medium", "large-v3"],
                value="large-v3" if DEVICE == "cuda" else "small",
                label="🧠 ASR model",
            )
            voice_match = gr.Slider(
                0.70,
                0.92,
                value=0.79,
                step=0.01,
                label="🎭 Character voice match threshold",
                info="Higher = fewer false matches, lower = more aggressive reuse.",
            )
            reuse_bank = gr.Checkbox(
                label="🔒 Reuse persistent Voice Bank across episodes",
                value=True,
            )
            deep_polish = gr.Checkbox(
                label="🧠 Deep Hindi dialogue polish (local Qwen3-4B)",
                value=True,
            )
            bed_mode = gr.Dropdown(
                choices=[
                    "Original residual (maximum fidelity)",
                    "Demucs accompaniment",
                    "Smart duck",
                ],
                value="Original residual (maximum fidelity)",
                label="🎵 BGM / SFX mode",
                info="Original residual attempts to reconstruct the original master minus recovered dialogue; it is not mathematically bit-perfect.",
            )
            duck = gr.Slider(
                0.0,
                0.75,
                value=0.35,
                step=0.05,
                label="🎵 Smart duck strength",
            )
            tts_passes = gr.Slider(
                1,
                3,
                value=TTS_NATURALNESS_PASSES,
                step=1,
                label="🎙️ Natural voice candidate passes",
                info="2–3 local Chatterbox candidates are ranked by prosody + character identity + tempo fit.",
            )
            voice_id_rerank = gr.Checkbox(
                label="🔒 Voice-identity re-rank",
                value=VOICE_ID_SCORING_DEFAULT,
                info="Checks each TTS candidate against the character's immutable speaker embedding.",
            )

        with gr.Column(scale=1):
            min_spk = gr.Number(
                label="👥 Minimum source speakers (optional)",
                value=1,
                precision=0,
                minimum=1,
                maximum=64,
            )
            max_spk = gr.Number(
                label="👥 Maximum source speakers for diarization",
                value=16,
                precision=0,
                minimum=1,
                maximum=64,
                info=f"Diarization limit is separate from the persistent {MAX_CHARACTERS_DEFAULT}+ character Voice Bank.",
            )
            lip = gr.Dropdown(
                choices=[
                    "Wav2Lip — Frame-locked Anime/CGI",
                    "MuseTalk 1.5",
                    "Anime-safe timing lock",
                ],
                value="Anime-safe timing lock",
                label="👄 Lip-sync engine",
                info="Anime-safe timing lock preserves original frames. Wav2Lip requires compatible local checkout/checkpoint and passes strict timing gates.",
            )
            gr.Markdown(
                "### 🎞️ Frame Sync Guard\n**Japanese/English → Hindi में word/phoneme length बदलती है।** इसलिए aligned-word cue → VAD refine → source-frame grid → Hindi delivery budget → controlled fast/slow → dialogue-only visual driver चलता है। Visual output पर optional sliding SyncNet QA चलता है; strict mode में failed QA window export रोकता है."
            )
            strict_sync = gr.Checkbox(
                label="🛡️ Strict local SyncNet QA",
                value=SYNCNET_REQUIRED_DEFAULT,
                info="Local SyncNet weights मौजूद होने पर active dialogue windows verify करता है.",
            )
            hard_sub = gr.Checkbox(
                label="🔤 Burn Hindi subtitles into video",
                value=False,
            )
            glossary = gr.Textbox(
                label="📚 Anime glossary (optional)",
                placeholder="Gojo=गोजो\nDomain Expansion=डोमेन एक्सपैंशन\nSensei=सेन्सेई",
                lines=8,
            )
            manual_translation = gr.Textbox(
                label="✏️ Manual Hindi translation override (optional)",
                placeholder="हर line = एक detected dialogue segment. खाली छोड़ें = MADLAD + local Qwen.",
                lines=8,
            )
            bank_status = gr.Markdown(voicebank_status("my_anime"))
            storage_box = gr.Markdown(storage_status())

    run_btn = gr.Button("🎌 DUB EPISODE", variant="primary", size="lg")
    refresh_btn = gr.Button("🔄 Refresh Voice Bank")
    storage_btn = gr.Button("💾 Refresh Storage Status")

    with gr.Row():
        transcript_out = gr.Textbox(
            label="📝 Source transcript",
            lines=12,
            interactive=False,
        )
        translation_out = gr.Textbox(
            label="🌐 Hindi dubbing script",
            lines=12,
            interactive=False,
        )

    with gr.Row():
        audio_out = gr.Audio(label="🔊 Final mixed audio")
        video_out = gr.Video(label="🎬 Final Hindi dubbed video")

    with gr.Row():
        srt_out = gr.File(label="📜 Hindi SRT")
        bank_out = gr.File(label="🎭 Voice Bank ZIP")

    log_out = gr.Textbox(
        label="📊 Character matching + pipeline log",
        lines=22,
        interactive=False,
    )

    run_btn.click(
        fn=full_pipeline,
        inputs=[
            video_in,
            project,
            model_size,
            voice_match,
            reuse_bank,
            deep_polish,
            bed_mode,
            duck,
            lip,
            hard_sub,
            min_spk,
            max_spk,
            glossary,
            manual_translation,
            tts_passes,
            voice_id_rerank,
            strict_sync,
        ],
        outputs=[
            transcript_out,
            translation_out,
            audio_out,
            video_out,
            srt_out,
            bank_out,
            log_out,
        ],
    )

    refresh_btn.click(
        fn=voicebank_status,
        inputs=[project],
        outputs=[bank_status],
    )
    storage_btn.click(
        fn=storage_status,
        inputs=[],
        outputs=[storage_box],
    )

    gr.Markdown("---")
    gr.Markdown(
        "### 🧠 Local MAX pipeline\n"
        "`FFmpeg → Demucs → WhisperX → aligned-word/VAD cue refine → source frame-grid lock → Pyannote Community-1 → persistent Voice Bank → MADLAD → Qwen3 → multi-candidate Chatterbox V3 → frame-locked dialogue track → original residual BGM/SFX → optional Wav2Lip → optional local SyncNet QA → ASS/SRT`"
    )
    gr.Markdown(
        "### 💾 Storage-safe rule\n"
        "`/kaggle/working/anime_dub_studio` में सिर्फ persistent Voice Bank + project memory/manifest रहता है। Heavy models, Demucs stems, temporary WAV/MP4 और session results `/kaggle/tmp/anime_dub_studio` में रहते हैं. इससे persistent Kaggle working quota model cache से नहीं भरता."
    )
    gr.Markdown(
        "**Cross-episode rule:** same `Anime / Project ID` रखें. Voice Bank folder को Kaggle Dataset में save करके अगली notebook/session में mount करें; इससे character references और canonical embeddings वापस मिलेंगे."
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
