```python
"""
Anime Dub Studio Ultra 2
Kaggle LOCAL model/cache preparation

KAGGLE TEMP STORAGE BUILD

IMPORTANT:
- Run with Kaggle Internet enabled.
- ALL heavy model downloads go to /kaggle/tmp by default.
- /kaggle/working is NOT used for model storage.
- Does NOT download every quantization/checkpoint in HF repositories.
- Downloads only allowlisted model files.
- Checks available scratch storage before each large download.
- HF_TOKEN is required for gated pyannote models.

STORAGE:

Temporary / current session:
    /kaggle/tmp/anime_dub_studio/models/

Persistent output area:
    /kaggle/working/
    Only a small manifest is optionally written there.

IMPORTANT:
    /kaggle/tmp can be cleared when the Kaggle session ends.
    For permanent reuse, package model folders as Kaggle Datasets
    and mount them under /kaggle/input/.
"""

from pathlib import Path
import os
import sys
import subprocess
import shutil
import json
import time


# ============================================================
# STORAGE CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# DEFAULT: EVERYTHING HEAVY GOES TO KAGGLE TEMP
# ------------------------------------------------------------

DEFAULT_TMP_ROOT = "/kaggle/tmp/anime_dub_studio"

SCRATCH_ROOT = Path(
    os.environ.get(
        "ANIME_DUB_SCRATCH",
        DEFAULT_TMP_ROOT
    )
)

# Model root inside temporary storage
ROOT = Path(
    os.environ.get(
        "MODEL_CACHE",
        str(SCRATCH_ROOT / "models")
    )
)

# Optional tiny persistent manifest location
PERSIST_MANIFEST_DIR = Path(
    os.environ.get(
        "MODEL_MANIFEST_DIR",
        "/kaggle/working/anime_dub_studio"
    )
)

SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
ROOT.mkdir(parents=True, exist_ok=True)
PERSIST_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HUGGING FACE CACHE
# ============================================================

HF_HOME = ROOT / "hf"
HF_HUB_CACHE = ROOT / "hub"
TRANSFORMERS_CACHE = ROOT / "transformers"
HF_DATASETS_CACHE = ROOT / "datasets"

HF_HOME.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
TRANSFORMERS_CACHE.mkdir(parents=True, exist_ok=True)
HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)


# Force HF/Transformers caches into /kaggle/tmp
for key, value in {
    "HF_HOME": HF_HOME,
    "HF_HUB_CACHE": HF_HUB_CACHE,
    "HUGGINGFACE_HUB_CACHE": HF_HUB_CACHE,
    "TRANSFORMERS_CACHE": TRANSFORMERS_CACHE,
    "HF_DATASETS_CACHE": HF_DATASETS_CACHE,
    "TORCH_HOME": ROOT / "torch",
    "XDG_CACHE_HOME": ROOT / "xdg",
}.items():
    os.environ[key] = str(value)

# Prevent Hugging Face from writing surprise caches elsewhere.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Kaggle temp for Python temporary files
TMP_DIR = SCRATCH_ROOT / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TMPDIR"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)

# Make tempfile use Kaggle scratch too.
try:
    import tempfile
    tempfile.tempdir = str(TMP_DIR)
except Exception:
    pass


# ============================================================
# HF TOKEN
# ============================================================

HF_TOKEN = (
    os.environ.get("HF_TOKEN", "").strip()
    or None
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODELS = {

    # --------------------------------------------------------
    # Translation
    # --------------------------------------------------------
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
    },

    # --------------------------------------------------------
    # Dialogue Director
    # --------------------------------------------------------
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
    },

    # --------------------------------------------------------
    # Speaker Encoder
    # --------------------------------------------------------
    "speaker_encoder": {
        "repo": "speechbrain/spkrec-ecapa-voxceleb",

        "patterns": [
            "*.ckpt",
            "*.yaml",
            "*.txt",
            "*.json",
            "*.md",
        ],

        "required_space_gb": 0.15,
        "optional": False,
    },

    # --------------------------------------------------------
    # PyAnnote Community-1
    # --------------------------------------------------------
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

        "required_space_gb": 0.8,
        "optional": False,
        "gated": True,
    },

    # --------------------------------------------------------
    # SenseVoice
    # --------------------------------------------------------
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
    },

    # --------------------------------------------------------
    # Chatterbox
    # --------------------------------------------------------
    "chatterbox": {
        "repo": "ResembleAI/chatterbox",

        "patterns": [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ],

        "required_space_gb": 3.5,
        "optional": True,
    },
}


# ============================================================
# UTILITY
# ============================================================

def run(cmd, cwd=None, quiet=False):
    """Run shell command with useful logging."""

    print(
        "$ " +
        " ".join(
            str(x) for x in cmd
        )
    )

    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=None,
    )


def disk_usage(path):
    return shutil.disk_usage(str(path))


def free_gb(path=None):
    """
    Calculate free storage for the actual scratch filesystem.

    By default checks SCRATCH_ROOT rather than /kaggle/working.
    """

    path = path or SCRATCH_ROOT

    try:
        usage = disk_usage(path)
        return usage.free / (1024 ** 3)
    except Exception:

        # Fallback
        usage = disk_usage("/kaggle/tmp")
        return usage.free / (1024 ** 3)


def used_gb(path=None):
    path = path or SCRATCH_ROOT

    try:
        usage = disk_usage(path)

        return {
            "total": usage.total / (1024 ** 3),
            "used": usage.used / (1024 ** 3),
            "free": usage.free / (1024 ** 3),
        }

    except Exception:

        usage = disk_usage("/kaggle/tmp")

        return {
            "total": usage.total / (1024 ** 3),
            "used": usage.used / (1024 ** 3),
            "free": usage.free / (1024 ** 3),
        }


def folder_size_gb(path):
    """
    Calculate actual folder size recursively.
    """

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


def print_disk():

    scratch = used_gb(SCRATCH_ROOT)
    working = used_gb("/kaggle/working")

    print()
    print("=" * 70)
    print("KAGGLE STORAGE")
    print("=" * 70)

    print()
    print("TEMP / SCRATCH")
    print(f"Path  : {SCRATCH_ROOT}")
    print(f"Total : {scratch['total']:.2f} GB")
    print(f"Used  : {scratch['used']:.2f} GB")
    print(f"Free  : {scratch['free']:.2f} GB")

    print()
    print("WORKING")
    print("Path  : /kaggle/working")
    print(f"Total : {working['total']:.2f} GB")
    print(f"Used  : {working['used']:.2f} GB")
    print(f"Free  : {working['free']:.2f} GB")

    print()
    print("=" * 70)
    print()


def safe_model_dir(name):

    path = ROOT / name

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    return path


def path_exists_nonempty(path):

    path = Path(path)

    if not path.exists():
        return False

    try:
        return any(path.iterdir())
    except Exception:
        return False


# ============================================================
# REQUIREMENTS
# ============================================================

def ensure_huggingface_hub():

    try:

        import huggingface_hub

        print(
            "huggingface_hub:",
            huggingface_hub.__version__
        )

        return True

    except Exception:

        print(
            "huggingface_hub not found."
        )

        print(
            "Installing compatible version..."
        )

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

            print(
                "huggingface_hub installation failed:"
            )

            print(e)

            return False


# ============================================================
# HUGGING FACE DOWNLOAD
# ============================================================

def hf_download_model(name, cfg):

    try:

        from huggingface_hub import snapshot_download

    except Exception:

        print(
            f"[{name}] huggingface_hub unavailable."
        )

        return False


    repo = cfg["repo"]
    patterns = cfg["patterns"]

    target = safe_model_dir(name)

    required = float(
        cfg.get(
            "required_space_gb",
            0
        )
    )

    available = free_gb()

    print()
    print("=" * 70)
    print(f"[{name}]")
    print("=" * 70)

    print(f"Repo           : {repo}")
    print(f"Target         : {target}")
    print(f"Required ~     : {required:.2f} GB")
    print(f"Scratch free   : {available:.2f} GB")
    print(f"Persistent     : /kaggle/working")
    print(f"Heavy storage  : /kaggle/tmp")
    print()

    # --------------------------------------------------------
    # TOKEN CHECK FOR GATED MODEL
    # --------------------------------------------------------

    if cfg.get("gated"):

        if not HF_TOKEN:

            print(
                "SKIP: HF_TOKEN is not configured."
            )

            print(
                "Accept the gated pyannote model terms "
                "and set HF_TOKEN before running again."
            )

            return False


    # --------------------------------------------------------
    # DISK CHECK
    # --------------------------------------------------------

    safety_margin = 0.50

    if available < required + safety_margin:

        print(
            "SKIP: insufficient temporary storage."
        )

        print(
            f"Need approximately "
            f"{required + safety_margin:.2f} GB "
            f"including safety margin."
        )

        print(
            f"Available: {available:.2f} GB"
        )

        return False


    # --------------------------------------------------------
    # SHOW EXISTING FILES
    # --------------------------------------------------------

    if path_exists_nonempty(target):

        print(
            "Existing target directory detected."
        )

        size = folder_size_gb(target)

        print(
            f"Existing size: {size:.2f} GB"
        )

        print(
            "HF download will reuse/resume existing files."
        )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    kwargs = {
        "repo_id": repo,
        "local_dir": str(target),
        "allow_patterns": patterns,
    }


    if HF_TOKEN and cfg.get("gated"):

        kwargs["token"] = HF_TOKEN


    try:

        start = time.time()

        snapshot_download(
            **kwargs
        )

        elapsed = time.time() - start

        size = folder_size_gb(target)

        print()
        print(
            f"[{name}] DOWNLOAD OK"
        )

        print(
            f"Final folder size: {size:.2f} GB"
        )

        print(
            f"Time: {elapsed / 60:.1f} min"
        )

        return True

    except Exception as e:

        print()
        print(
            f"[{name}] DOWNLOAD FAILED"
        )

        print(
            str(e)[:2000]
        )

        return False


# ============================================================
# CHATTERBOX VERIFY
# ============================================================

def verify_chatterbox():

    path = ROOT / "chatterbox"

    required = [
        "ve.safetensors",
        "t3_cfg.safetensors",
        "s3gen.safetensors",
        "tokenizer.json",
        "conds.pt",
    ]

    print()
    print("=" * 70)
    print("CHATTERBOX FILE CHECK")
    print("=" * 70)

    ok = True

    for filename in required:

        p = path / filename

        if p.exists():

            size_mb = (
                p.stat().st_size /
                (1024 ** 2)
            )

            print(
                f"OK    {filename:35s} "
                f"{size_mb:,.1f} MB"
            )

        else:

            print(
                f"MISS  {filename}"
            )

            ok = False

    print()

    return ok


# ============================================================
# WAV2LIP
# ============================================================

def wav2lip():

    repo_dir = ROOT / "Wav2Lip"
    ckpt_dir = repo_dir / "checkpoints"

    gan = ckpt_dir / "wav2lip_gan.pth"

    print()
    print("=" * 70)
    print("WAV2LIP")
    print("=" * 70)

    print(
        "Repository:",
        repo_dir
    )

    if gan.exists():

        print(
            "Wav2Lip checkpoint already exists."
        )

        return True


    required = 1.5

    available = free_gb()

    if available < required + 0.25:

        print(
            "SKIP: not enough scratch storage for Wav2Lip."
        )

        return False


    # --------------------------------------------------------
    # Clone repo into /kaggle/tmp
    # --------------------------------------------------------

    if not repo_dir.exists():

        print(
            "Cloning Wav2Lip into temporary storage..."
        )

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

            print(
                "Wav2Lip clone failed:"
            )

            print(e)

            return False


    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Download checkpoint
    # --------------------------------------------------------

    file_id = (
        "15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk"
    )

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

            print(
                "Wav2Lip checkpoint OK"
            )

            return True

        print(
            "Wav2Lip checkpoint file looks invalid."
        )

        return False

    except Exception as e:

        print(
            "Wav2Lip checkpoint download failed:"
        )

        print(e)

        return False


# ============================================================
# SYNCNET
# ============================================================

def syncnet():

    repo_dir = ROOT / "syncnet-python"

    print()
    print("=" * 70)
    print("SYNCNET")
    print("=" * 70)

    print(
        "Repository:",
        repo_dir
    )


    # --------------------------------------------------------
    # Stable model location
    # --------------------------------------------------------

    stable_dir = ROOT / "syncnet"
    stable = stable_dir / "syncnet_v2.model"

    stable_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    if stable.exists():

        size_mb = (
            stable.stat().st_size /
            (1024 ** 2)
        )

        if stable.stat().st_size > 10_000_000:

            print(
                f"SyncNet stable weights already exist "
                f"({size_mb:.1f} MB)."
            )

            return True


    # --------------------------------------------------------
    # Clone repository
    # --------------------------------------------------------

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

            print(
                "SyncNet clone failed:"
            )

            print(e)

            return False


    # --------------------------------------------------------
    # Download model
    # --------------------------------------------------------

    script = repo_dir / "download_model.sh"

    if script.exists():

        try:

            run(
                [
                    "bash",
                    str(script)
                ],
                cwd=str(repo_dir)
            )

        except Exception as e:

            print(
                "SyncNet download script failed:"
            )

            print(e)


    # --------------------------------------------------------
    # Locate downloaded model
    # --------------------------------------------------------

    candidates = []

    try:

        candidates += list(
            repo_dir.rglob("*.model")
        )

        candidates += list(
            repo_dir.rglob("*.pth")
        )

    except Exception:
        pass


    candidates = sorted(
        candidates,
        key=lambda p: p.stat().st_size
        if p.exists()
        else 0,
        reverse=True,
    )


    for src in candidates:

        try:

            if (
                src.exists()
                and src.stat().st_size > 10_000_000
                and (
                    "syncnet"
                    in src.name.lower()
                    or src.suffix.lower()
                    in {".model", ".pth"}
                )
            ):

                shutil.copy2(
                    src,
                    stable
                )

                print()
                print(
                    "SyncNet stable weights:"
                )

                print(
                    stable
                )

                print(
                    f"Size: "
                    f"{stable.stat().st_size / (1024**2):.1f} MB"
                )

                return True

        except Exception as e:

            print(
                "SyncNet copy failed:"
            )

            print(e)


    print(
        "SyncNet model weights not found."
    )

    return False


# ============================================================
# INCOMPLETE / TEMP CACHE CLEANUP
# ============================================================

def show_cache_sizes():

    print()
    print("=" * 70)
    print("SCRATCH CACHE SIZES")
    print("=" * 70)

    try:

        for child in sorted(
            ROOT.iterdir()
        ):

            if child.is_dir():

                size = folder_size_gb(
                    child
                )

                print(
                    f"{child.name:35s} "
                    f"{size:8.2f} GB"
                )

    except Exception as e:

        print(
            "Cache size scan failed:",
            e
        )


def clean_failed_partial_directory(name):

    """
    Optional manual helper.

    It does NOT automatically delete anything because
    partially downloaded files can sometimes be resumed.
    """

    path = ROOT / name

    if not path.exists():

        print(
            f"No cache found for {name}"
        )

        return


    print()
    print(
        f"{name} cache exists at:"
    )

    print(
        path
    )

    print(
        "No automatic deletion performed."
    )

    print(
        "This allows Hugging Face resume behavior."
    )


# ============================================================
# MANIFEST
# ============================================================

def write_manifest(results):

    data = used_gb(
        SCRATCH_ROOT
    )

    manifest = {

        "app":
            "Anime Dub Studio Ultra 2",

        "created":
            time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime()
            ),

        "scratch_root":
            str(SCRATCH_ROOT),

        "model_root":
            str(ROOT),

        "scratch_storage": {
            "total_gb":
                round(data["total"], 2),

            "used_gb":
                round(data["used"], 2),

            "free_gb":
                round(data["free"], 2),
        },

        "models":
            results,
    }


    # --------------------------------------------------------
    # Persistent copy of only the small manifest
    # --------------------------------------------------------

    persistent_path = (
        PERSIST_MANIFEST_DIR /
        "model_manifest.json"
    )


    try:

        persistent_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print()
        print(
            "Persistent manifest:"
        )

        print(
            persistent_path
        )

    except Exception as e:

        print(
            "Persistent manifest write failed:"
        )

        print(e)


    # Also keep a copy next to the model cache
    try:

        temp_manifest = (
            ROOT /
            "model_manifest.json"
        )

        temp_manifest.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    except Exception:
        pass


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ANIME DUB STUDIO ULTRA 2")
    print("KAGGLE LOCAL MODEL PREPARATION")
    print("TEMP STORAGE BUILD")
    print("=" * 70)

    print()
    print(
        "Scratch root:"
    )

    print(
        SCRATCH_ROOT
    )

    print()
    print(
        "Model root:"
    )

    print(
        ROOT
    )

    print()
    print(
        "Heavy model/cache storage:"
    )

    print(
        "✅ /kaggle/tmp"
    )

    print()
    print(
        "Persistent output:"
    )

    print(
        "✅ /kaggle/working "
        "(manifest only)"
    )


    # --------------------------------------------------------
    # Ensure directories
    # --------------------------------------------------------

    SCRATCH_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Initial disk report
    # --------------------------------------------------------

    print_disk()

    show_cache_sizes()


    # --------------------------------------------------------
    # Hugging Face dependency
    # --------------------------------------------------------

    if not ensure_huggingface_hub():

        print(
            "Cannot continue without huggingface_hub."
        )

        return


    results = {}


    # --------------------------------------------------------
    # Download HF models
    # --------------------------------------------------------

    for name, cfg in MODELS.items():

        print()

        status = hf_download_model(
            name,
            cfg
        )

        results[name] = bool(
            status
        )

        print_disk()


    # --------------------------------------------------------
    # Chatterbox verification
    # --------------------------------------------------------

    chatterbox_dir = (
        ROOT /
        "chatterbox"
    )

    if chatterbox_dir.exists():

        results[
            "chatterbox_verify"
        ] = bool(
            verify_chatterbox()
        )


    # --------------------------------------------------------
    # Wav2Lip
    # --------------------------------------------------------

    results[
        "wav2lip"
    ] = bool(
        wav2lip()
    )

    print_disk()


    # --------------------------------------------------------
    # SyncNet
    # --------------------------------------------------------

    results[
        "syncnet"
    ] = bool(
        syncnet()
    )

    print_disk()


    # --------------------------------------------------------
    # Final cache report
    # --------------------------------------------------------

    show_cache_sizes()


    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    write_manifest(
        results
    )


    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREPARATION FINISHED")
    print("=" * 70)

    print()
    print(
        "Results:"
    )

    for name, status in results.items():

        label = (
            "OK"
            if status
            else
            "SKIP/FAIL"
        )

        print(
            f"  {label:10s} {name}"
        )


    print()

    print(
        "Temporary model root:"
    )

    print(
        ROOT
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Large models are intentionally stored under "
        "/kaggle/tmp instead of /kaggle/working."
    )

    print()

    print(
        "Because /kaggle/tmp is session scratch storage, "
        "models may disappear after the Kaggle session ends."
    )

    print()

    print(
        "For persistent reuse, package the completed model "
        "directory as a Kaggle Dataset and mount it under "
        "/kaggle/input/."
    )

    print()

    print_disk()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
```
