"""
Anime Dub Studio Ultra 2
Kaggle LOCAL model/cache preparation

IMPORTANT:
- Run with Kaggle Internet enabled.
- Downloads ONLY required model files.
- Does NOT download every quantization/checkpoint in a HF repository.
- Uses local folders suitable for later Kaggle Dataset packaging.
- HF_TOKEN is required for gated pyannote models.

Storage reality:
MADLAD 3B + Qwen3-4B + Chatterbox + other models can exceed
Kaggle's available writable disk. This script therefore checks
free space before each large download.
"""

from pathlib import Path
import os
import sys
import subprocess
import shutil
import json


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    os.environ.get(
        "MODEL_CACHE",
        "/kaggle/working/anime_dub_studio/models"
    )
)

ROOT.mkdir(parents=True, exist_ok=True)

HF_HOME = ROOT / "hf"
HF_HUB_CACHE = ROOT / "hub"
TRANSFORMERS_CACHE = ROOT / "transformers"
HF_DATASETS_CACHE = ROOT / "datasets"

for key, value in {
    "HF_HOME": HF_HOME,
    "HF_HUB_CACHE": HF_HUB_CACHE,
    "TRANSFORMERS_CACHE": TRANSFORMERS_CACHE,
    "HF_DATASETS_CACHE": HF_DATASETS_CACHE,
}.items():
    os.environ[key] = str(value)


HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODELS = {
    # --------------------------------------------------------
    # Translation
    # --------------------------------------------------------
    "translation": {
        "repo": "google/madlad400-3b-mt",

        # ONLY files required by Transformers.
        #
        # IMPORTANT:
        # This downloads the 11.8 GB full safetensors model.
        # It intentionally does NOT download Q2/Q3/Q4 GGUF.
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
    # Dialogue / Script Director
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
    # Chatterbox English
    # --------------------------------------------------------
    #
    # Official ChatterboxTTS.from_pretrained() needs:
    # ve.safetensors
    # t3_cfg.safetensors
    # s3gen.safetensors
    # tokenizer.json
    # conds.pt
    #
    # This avoids downloading all multilingual / legacy files.
    #
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

def run(cmd, cwd=None):
    print("$", " ".join(map(str, cmd)))
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True
    )


def free_gb(path="/kaggle/working"):
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def print_disk():
    usage = shutil.disk_usage("/kaggle/working")

    total = usage.total / (1024 ** 3)
    used = usage.used / (1024 ** 3)
    free = usage.free / (1024 ** 3)

    print()
    print("=" * 70)
    print("KAGGLE DISK")
    print("=" * 70)
    print(f"Total : {total:.2f} GB")
    print(f"Used  : {used:.2f} GB")
    print(f"Free  : {free:.2f} GB")
    print("=" * 70)
    print()


def safe_model_dir(name):
    path = ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# HUGGING FACE DOWNLOAD
# ============================================================

def hf_download_model(name, cfg):

    from huggingface_hub import snapshot_download

    repo = cfg["repo"]
    patterns = cfg["patterns"]

    target = safe_model_dir(name)

    required = cfg.get("required_space_gb", 0)
    available = free_gb()

    print()
    print("=" * 70)
    print(f"[{name}]")
    print(f"Repo      : {repo}")
    print(f"Target    : {target}")
    print(f"Need ~    : {required:.2f} GB")
    print(f"Free      : {available:.2f} GB")
    print("=" * 70)

    if available < required:
        print(
            f"SKIP: insufficient disk space. "
            f"Need ~{required:.2f} GB, have {available:.2f} GB."
        )
        return False

    if cfg.get("gated"):
        if not HF_TOKEN:
            print(
                "SKIP: HF_TOKEN not configured. "
                "Accept the pyannote model terms and set HF_TOKEN."
            )
            return False

    kwargs = {
        "repo_id": repo,
        "cache_dir": str(HF_HUB_CACHE),
        "local_dir": str(target),
        "allow_patterns": patterns,
        "resume_download": True,
    }

    if cfg.get("gated"):
        kwargs["token"] = HF_TOKEN

    try:
        snapshot_download(**kwargs)

        print(f"[{name}] DOWNLOAD OK")
        return True

    except Exception as e:
        print(f"[{name}] DOWNLOAD FAILED")
        print(str(e)[:1000])
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

    for file in required:
        p = path / file

        if p.exists():
            size_mb = p.stat().st_size / (1024 ** 2)
            print(f"OK    {file:35s} {size_mb:,.1f} MB")
        else:
            print(f"MISS  {file}")
            ok = False

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

    if gan.exists():
        print("Wav2Lip checkpoint already exists.")
        return True

    required = 1.5

    if free_gb() < required:
        print("SKIP: not enough disk space for Wav2Lip.")
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
            print("Wav2Lip clone failed:")
            print(e)
            return False

    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

        print("Wav2Lip checkpoint OK")
        return True

    except Exception as e:

        print("Wav2Lip checkpoint download failed:")
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
            print("SyncNet clone failed:")
            print(e)
            return False

    script = repo_dir / "download_model.sh"

    if script.exists():

        try:
            run(
                ["bash", str(script)],
                cwd=str(repo_dir)
            )

        except Exception as e:

            print("SyncNet download script failed:")
            print(e)

    stable = ROOT / "syncnet" / "syncnet_v2.model"
    stable.parent.mkdir(parents=True, exist_ok=True)

    candidates = (
        list(repo_dir.rglob("*.model"))
        + list(repo_dir.rglob("*.pth"))
    )

    for src in candidates:

        try:

            if (
                src.name.lower().startswith("syncnet")
                and src.stat().st_size > 10_000_000
            ):

                shutil.copy2(src, stable)

                print(
                    "SyncNet stable weights:",
                    stable
                )

                return True

        except Exception as e:

            print("SyncNet copy failed:")
            print(e)

    print("SyncNet model weights not found.")
    return False


# ============================================================
# CLEAN INCOMPLETE HF CACHE
# ============================================================

def clean_incomplete_qwen():

    """
    The previous failed Qwen download may have left partial
    blobs occupying several GB.

    This function ONLY removes the Qwen cache so it can be
    downloaded cleanly later.
    """

    qwen_cache = (
        HF_HUB_CACHE
        / "models--Qwen--Qwen3-4B"
    )

    if not qwen_cache.exists():
        return

    print()
    print("Found existing Qwen HF cache:")
    print(qwen_cache)

    print(
        "Leaving existing cache untouched by default "
        "so completed files can be reused."
    )


# ============================================================
# MANIFEST
# ============================================================

def write_manifest(results):

    manifest = {
        "root": str(ROOT),
        "python": sys.version,
        "free_disk_gb_after": round(free_gb(), 2),
        "models": results,
    }

    path = ROOT / "model_manifest.json"

    path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print("Manifest:")
    print(path)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ANIME DUB STUDIO ULTRA 2")
    print("KAGGLE LOCAL MODEL PREPARATION")
    print("=" * 70)

    print("Model root:", ROOT)

    print_disk()

    try:

        import huggingface_hub

        print(
            "huggingface_hub:",
            huggingface_hub.__version__
        )

    except Exception:

        print("Installing huggingface_hub...")

        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "huggingface_hub>=0.30"
        ])

    results = {}

    # --------------------------------------------------------
    # Download HF models
    # --------------------------------------------------------

    for name, cfg in MODELS.items():

        results[name] = hf_download_model(
            name,
            cfg
        )

        print_disk()

    # --------------------------------------------------------
    # Verify Chatterbox
    # --------------------------------------------------------

    if (ROOT / "chatterbox").exists():
        results["chatterbox_verify"] = verify_chatterbox()

    # --------------------------------------------------------
    # Wav2Lip
    # --------------------------------------------------------

    results["wav2lip"] = wav2lip()

    print_disk()

    # --------------------------------------------------------
    # SyncNet
    # --------------------------------------------------------

    results["syncnet"] = syncnet()

    print_disk()

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    write_manifest(results)

    print()
    print("=" * 70)
    print("PREPARATION FINISHED")
    print("=" * 70)

    print()
    print("Results:")

    for name, status in results.items():
        print(
            f"  {'OK' if status else 'SKIP/FAIL':10s} {name}"
        )

    print()
    print(
        "IMPORTANT: A complete MADLAD + Qwen + Chatterbox stack "
        "may not fit into one Kaggle writable disk."
    )

    print(
        "For a fully offline setup, package the large model folders "
        "into separate Kaggle Datasets and mount them under /kaggle/input."
    )

    print()


if __name__ == "__main__":
    main()
