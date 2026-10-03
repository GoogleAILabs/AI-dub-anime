"""
Anime Dub Studio Ultra 2
Kaggle LOCAL model/cache preparation — TEMP STORAGE build

- Heavy models/cache -> /kaggle/tmp (fallback /kaggle/temp, then /tmp)
- /kaggle/working -> small manifest only
- Resumable Hugging Face downloads
- Gated pyannote requires HF_TOKEN
- Chatterbox entry is Multilingual V3
"""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
import tempfile


# ============================================================
# STORAGE
# ============================================================

SCRATCH_PARENT = None

for candidate in [Path("/kaggle/tmp"), Path("/kaggle/temp"), Path("/tmp")]:
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".anime_dub_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        SCRATCH_PARENT = candidate
        break
    except Exception:
        continue

if SCRATCH_PARENT is None:
    SCRATCH_PARENT = Path(tempfile.gettempdir())  # pragma: no cover

SCRATCH_ROOT = Path(
    os.environ.get(
        "ANIME_DUB_SCRATCH",
        str(SCRATCH_PARENT / "anime_dub_studio"),
    )
)

MODEL_ROOT = Path(
    os.environ.get(
        "MODEL_CACHE",
        str(SCRATCH_ROOT / "models"),
    )
)

PERSIST_ROOT = Path(
    os.environ.get(
        "MODEL_MANIFEST_DIR",
        "/kaggle/working/anime_dub_studio",
    )
)

SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
PERSIST_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CACHE ENVIRONMENT
# ============================================================

HF_HOME = MODEL_ROOT / "hf"
HF_HUB_CACHE = MODEL_ROOT / "hub"
TRANSFORMERS_CACHE = MODEL_ROOT / "transformers"
HF_DATASETS_CACHE = MODEL_ROOT / "datasets"
TORCH_HOME = MODEL_ROOT / "torch"
XDG_CACHE_HOME = MODEL_ROOT / "xdg"
TMP_ROOT = SCRATCH_ROOT / "tmp"

for p in [
    HF_HOME,
    HF_HUB_CACHE,
    TRANSFORMERS_CACHE,
    HF_DATASETS_CACHE,
    TORCH_HOME,
    XDG_CACHE_HOME,
    TMP_ROOT,
]:
    p.mkdir(parents=True, exist_ok=True)

for key, value in {
    "HF_HOME": HF_HOME,
    "HF_HUB_CACHE": HF_HUB_CACHE,
    "HUGGINGFACE_HUB_CACHE": HF_HUB_CACHE,
    "TRANSFORMERS_CACHE": TRANSFORMERS_CACHE,
    "HF_DATASETS_CACHE": HF_DATASETS_CACHE,
    "TORCH_HOME": TORCH_HOME,
    "XDG_CACHE_HOME": XDG_CACHE_HOME,
    "TMPDIR": TMP_ROOT,
    "TEMP": TMP_ROOT,
    "TMP": TMP_ROOT,
    "GRADIO_TEMP_DIR": TMP_ROOT,
}.items():
    os.environ[key] = str(value)

try:
    import tempfile
    tempfile.tempdir = str(TMP_ROOT)
except Exception:
    pass

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "translation": {
        "repo": "google/madlad400-3b-mt",
        "patterns": [
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
        "required_space_gb": 13.0,
        "optional": True,
        "description": "MADLAD-400 3B translation",
    },

    "dialogue_director": {
        "repo": "Qwen/Qwen3-4B",
        "patterns": [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
            "chat_template.json",
            "special_tokens_map.json",
        ],
        "required_space_gb": 9.0,
        "optional": True,
        "description": "Qwen3 4B dialogue director",
    },

    "speaker_encoder": {
        "repo": "speechbrain/spkrec-ecapa-voxceleb",
        "patterns": [
            "*.ckpt",
            "*.yaml",
            "*.txt",
            "*.json",
            "*.md",
        ],
        "required_space_gb": 0.25,
        "optional": False,
        "description": "ECAPA speaker encoder",
    },

    "pyannote": {
        "repo": "pyannote/speaker-diarization-community-1",
        "patterns": [
            "*.yaml",
            "*.yml",
            "*.json",
            "*.bin",
            "*.safetensors",
            "*.ckpt",
            "*.txt",
        ],
        "required_space_gb": 1.0,
        "optional": False,
        "gated": True,
        "description": "pyannote Community-1 diarization",
    },

    "sensevoice": {
        "repo": "FunAudioLLM/SenseVoiceSmall",
        "patterns": [
            "model.pt",
            "config.yaml",
            "configuration.json",
            "chn_jpn_yue_eng_ko_spectok.bpe.model",
            "am.mvn",
            "README.md",
        ],
        "required_space_gb": 1.0,
        "optional": True,
        "description": "SenseVoiceSmall emotion/event model",
    },

    # Current Chatterbox Multilingual V3 T3 checkpoint.
    "chatterbox": {
        "repo": "ResembleAI/chatterbox",
        "patterns": [
            "ve.safetensors",
            "t3_mtl23ls_v3.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ],
        "required_space_gb": 4.5,
        "optional": True,
        "description": "Chatterbox Multilingual V3",
    },
}


# ============================================================
# UTILITIES
# ============================================================

def run(cmd, cwd=None, quiet=False):
    print("$", " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=None,
    )


def disk_info(path):
    u = shutil.disk_usage(str(path))
    return {
        "total_gb": u.total / (1024 ** 3),
        "used_gb": u.used / (1024 ** 3),
        "free_gb": u.free / (1024 ** 3),
    }


def free_gb(path=None):
    target = path or SCRATCH_ROOT
    try:
        return disk_info(target)["free_gb"]
    except Exception:
        for p in [Path("/kaggle/tmp"), Path("/kaggle/temp"), Path("/tmp")]:
            try:
                return disk_info(p)["free_gb"]
            except Exception:
                pass
    return 0.0


def folder_size_gb(path):
    path = Path(path)
    if not path.exists():
        return 0.0

    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except Exception:
                    pass
    except Exception:
        pass

    return total / (1024 ** 3)


def has_files(path):
    try:
        return Path(path).exists() and any(Path(path).iterdir())
    except Exception:
        return False


def print_storage():
    print()
    print("=" * 72)
    print("KAGGLE STORAGE")
    print("=" * 72)

    for label, path in [
        ("SCRATCH", SCRATCH_ROOT),
        ("MODELS", MODEL_ROOT),
        ("WORKING", Path("/kaggle/working")),
    ]:
        try:
            info = disk_info(path)
            print(
                f"{label:9s} "
                f"{str(path):48s} "
                f"free={info['free_gb']:.2f} GB "
                f"used={info['used_gb']:.2f} GB"
            )
        except Exception as e:
            print(f"{label:9s} {e}")

    print("=" * 72)


def show_model_sizes():
    print()
    print("=" * 72)
    print("MODEL FOLDER SIZES")
    print("=" * 72)

    if not MODEL_ROOT.exists():
        print("No model cache yet.")
        return

    for p in sorted(MODEL_ROOT.iterdir()):
        if p.is_dir():
            print(f"{p.name:32s} {folder_size_gb(p):8.2f} GB")


# ============================================================
# DEPENDENCIES
# ============================================================

def ensure_huggingface_hub():
    try:
        import huggingface_hub
        print("huggingface_hub:", huggingface_hub.__version__)
        return True
    except Exception:
        print("huggingface_hub missing; installing...")
        try:
            run([
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "huggingface_hub>=0.30",
            ])
            return True
        except Exception as e:
            print("huggingface_hub installation failed:", e)
            return False


def ensure_gdown():
    try:
        import gdown  # noqa
        return True
    except Exception:
        print("gdown missing; installing...")
        try:
            run([
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "gdown",
            ])
            return True
        except Exception as e:
            print("gdown installation failed:", e)
            return False


# ============================================================
# HUGGING FACE
# ============================================================

def hf_download_model(name, cfg):
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print(f"[{name}] huggingface_hub unavailable.")
        return False

    target = MODEL_ROOT / name
    target.mkdir(parents=True, exist_ok=True)

    required = float(cfg.get("required_space_gb", 0))
    available = free_gb(SCRATCH_ROOT)
    safety_margin = 0.50

    print()
    print("=" * 72)
    print(f"[{name}] {cfg.get('description', '')}")
    print("=" * 72)
    print("Repo         :", cfg["repo"])
    print("Target       :", target)
    print(f"Need approx  : {required:.2f} GB")
    print(f"Scratch free : {available:.2f} GB")

    if cfg.get("gated") and not HF_TOKEN:
        print("SKIP: HF_TOKEN is not configured.")
        return False

    if available < required + safety_margin:
        print("SKIP: insufficient scratch storage.")
        print(
            f"Need with safety margin: "
            f"{required + safety_margin:.2f} GB"
        )
        return False

    if has_files(target):
        print(
            f"Existing target size: "
            f"{folder_size_gb(target):.2f} GB"
        )
        print("Hugging Face will reuse/resume existing files.")

    kwargs = {
        "repo_id": cfg["repo"],
        "local_dir": str(target),
        "allow_patterns": cfg["patterns"],
    }

    if cfg.get("gated"):
        kwargs["token"] = HF_TOKEN

    try:
        started = time.time()

        snapshot_download(**kwargs)

        elapsed = time.time() - started
        final_size = folder_size_gb(target)

        print()
        print(f"[{name}] DOWNLOAD OK")
        print(f"Final size : {final_size:.2f} GB")
        print(f"Elapsed    : {elapsed / 60:.1f} min")
        return True

    except Exception as e:
        print(f"[{name}] DOWNLOAD FAILED")
        print(str(e)[:3000])
        return False


# ============================================================
# CHATTERBOX VERIFY
# ============================================================

def verify_chatterbox():
    root = MODEL_ROOT / "chatterbox"

    required = [
        "ve.safetensors",
        "t3_mtl23ls_v3.safetensors",
        "s3gen.safetensors",
        "tokenizer.json",
        "conds.pt",
    ]

    print()
    print("=" * 72)
    print("CHATTERBOX MULTILINGUAL V3 CHECK")
    print("=" * 72)

    ok = True

    for filename in required:
        p = root / filename
        if p.exists():
            size_mb = p.stat().st_size / (1024 ** 2)
            print(f"OK   {filename:40s} {size_mb:,.1f} MB")
        else:
            print(f"MISS {filename}")
            ok = False

    return ok


# ============================================================
# WAV2LIP
# ============================================================

def install_wav2lip():
    repo_dir = MODEL_ROOT / "Wav2Lip"
    ckpt_dir = repo_dir / "checkpoints"
    gan = ckpt_dir / "wav2lip_gan.pth"

    print()
    print("=" * 72)
    print("WAV2LIP")
    print("=" * 72)

    if gan.exists() and gan.stat().st_size > 100_000_000:
        print("Wav2Lip checkpoint already exists.")
        return True

    if free_gb(SCRATCH_ROOT) < 2.0:
        print("SKIP: not enough scratch storage.")
        return False

    if not repo_dir.exists():
        try:
            run([
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/Rudrabha/Wav2Lip.git",
                str(repo_dir),
            ])
        except Exception as e:
            print("Wav2Lip clone failed:", e)
            return False

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not ensure_gdown():
        return False

    file_id = "15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk"

    try:
        run([
            sys.executable,
            "-m",
            "gdown",
            "--id",
            file_id,
            "-O",
            str(gan),
        ])

        if gan.exists() and gan.stat().st_size > 100_000_000:
            print("Wav2Lip checkpoint OK.")
            return True

        print("Wav2Lip checkpoint appears incomplete.")
        return False

    except Exception as e:
        print("Wav2Lip checkpoint download failed:", e)
        return False


# ============================================================
# SYNCNET
# ============================================================

def install_syncnet():
    repo_dir = MODEL_ROOT / "syncnet-python"
    stable_dir = MODEL_ROOT / "syncnet"
    stable = stable_dir / "syncnet_v2.model"

    print()
    print("=" * 72)
    print("SYNCNET")
    print("=" * 72)

    stable_dir.mkdir(parents=True, exist_ok=True)

    if stable.exists() and stable.stat().st_size > 10_000_000:
        print("SyncNet weights already exist.")
        print(stable)
        return True

    if free_gb(SCRATCH_ROOT) < 1.0:
        print("SKIP: not enough scratch storage.")
        return False

    if not repo_dir.exists():
        try:
            run([
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/colossyan/syncnet-python.git",
                str(repo_dir),
            ])
        except Exception as e:
            print("SyncNet clone failed:", e)
            return False

    script = repo_dir / "download_model.sh"

    if script.exists():
        try:
            run(
                ["bash", str(script)],
                cwd=repo_dir,
            )
        except Exception as e:
            print("SyncNet download script failed:", e)

    candidates = []
    try:
        candidates.extend(repo_dir.rglob("*.model"))
        candidates.extend(repo_dir.rglob("*.pth"))
    except Exception:
        pass

    candidates = sorted(
        candidates,
        key=lambda p: p.stat().st_size if p.exists() else 0,
        reverse=True,
    )

    for src in candidates:
        try:
            if src.exists() and src.stat().st_size > 10_000_000:
                shutil.copy2(src, stable)
                print("SyncNet weights copied:")
                print(stable)
                return True
        except Exception as e:
            print("SyncNet copy failed:", e)

    print("SyncNet model weights not found.")
    return False


# ============================================================
# MANIFEST
# ============================================================

def write_manifest(results):
    scratch = disk_info(SCRATCH_ROOT)

    manifest = {
        "app": "Anime Dub Studio Ultra 2",
        "created_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "scratch_root": str(SCRATCH_ROOT),
        "model_root": str(MODEL_ROOT),
        "persistent_manifest_dir": str(PERSIST_ROOT),
        "scratch_storage": {
            "total_gb": round(scratch["total_gb"], 2),
            "used_gb": round(scratch["used_gb"], 2),
            "free_gb": round(scratch["free_gb"], 2),
        },
        "models": results,
    }

    payload = json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )

    temp_manifest = MODEL_ROOT / "model_manifest.json"
    persistent_manifest = PERSIST_ROOT / "model_manifest.json"

    temp_manifest.write_text(payload, encoding="utf-8")
    persistent_manifest.write_text(payload, encoding="utf-8")

    print()
    print("Manifest saved:")
    print("Temporary :", temp_manifest)
    print("Persistent:", persistent_manifest)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 72)
    print("ANIME DUB STUDIO ULTRA 2")
    print("KAGGLE LOCAL MODEL PREPARATION")
    print("TEMP STORAGE BUILD")
    print("=" * 72)

    print()
    print("Scratch root:", SCRATCH_ROOT)
    print("Model root  :", MODEL_ROOT)
    print("Persistent  :", PERSIST_ROOT)
    print()
    print("Heavy model/cache storage -> scratch")
    print("Persistent output        -> /kaggle/working")
    print()

    print_storage()
    show_model_sizes()

    if not ensure_huggingface_hub():
        return 1

    results = {}

    # HF models
    for name, cfg in MODELS.items():
        results[name] = bool(
            hf_download_model(name, cfg)
        )
        print_storage()

    # Verify Chatterbox
    if (MODEL_ROOT / "chatterbox").exists():
        results["chatterbox_verify"] = bool(
            verify_chatterbox()
        )

    # Wav2Lip
    results["wav2lip"] = bool(
        install_wav2lip()
    )
    print_storage()

    # SyncNet
    results["syncnet"] = bool(
        install_syncnet()
    )
    print_storage()

    # Final report
    show_model_sizes()
    write_manifest(results)

    print()
    print("=" * 72)
    print("PREPARATION FINISHED")
    print("=" * 72)

    for name, status in results.items():
        print(
            f"  {'OK' if status else 'SKIP/FAIL':10s} {name}"
        )

    print()
    print("Model cache:")
    print(MODEL_ROOT)
    print()
    print(
        "NOTE: scratch storage is not guaranteed to survive "
        "a new Kaggle session."
    )
    print(
        "For permanent reuse, package model folders as Kaggle "
        "Dataset(s) and mount them under /kaggle/input/."
    )

    print_storage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
