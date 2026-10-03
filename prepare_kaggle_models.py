"""Kaggle one-time LOCAL model/cache preparation.

Run while Kaggle Internet is enabled. After caches/checkpoints are present, the
inference app itself does not call any external TTS/translation/lipsync API.
HF_TOKEN is only used for gated Hugging Face models such as pyannote.
"""
from pathlib import Path
import os, subprocess, sys, shutil

ROOT = Path(os.environ.get("MODEL_CACHE", "/kaggle/working/anime_dub_studio/models"))
ROOT.mkdir(parents=True, exist_ok=True)
for key, value in {
    "HF_HOME": ROOT / "hf",
    "TRANSFORMERS_CACHE": ROOT / "transformers",
    "HF_DATASETS_CACHE": ROOT / "datasets",
}.items():
    os.environ[key] = str(value)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None

MODELS = {
    "translation": "google/madlad400-3b-mt",
    "dialogue_director": "Qwen/Qwen3-4B",
    "speaker_encoder": "speechbrain/spkrec-ecapa-voxceleb",
    "pyannote": "pyannote/speaker-diarization-community-1",
    "sensevoice": "iic/SenseVoiceSmall",
    "chatterbox": "ResembleAI/chatterbox",
}


def run(cmd, cwd=None):
    print("$", " ".join(map(str, cmd)))
    return subprocess.run(cmd, cwd=cwd, check=True)


def hf_cache():
    from huggingface_hub import snapshot_download
    for name, repo in MODELS.items():
        print(f"\n[{name}] {repo}")
        kwargs = {"repo_id": repo, "cache_dir": str(ROOT / "hub")}
        if repo.startswith("pyannote/"):
            kwargs["token"] = HF_TOKEN
        try:
            snapshot_download(**kwargs)
            print("  OK")
        except Exception as e:
            print("  SKIP/ERROR:", str(e)[:250])


def wav2lip():
    repo_dir = ROOT / "Wav2Lip"
    if not repo_dir.exists():
        try:
            run(["git", "clone", "--depth", "1", "https://github.com/Rudrabha/Wav2Lip.git", str(repo_dir)])
        except Exception as e:
            print("Wav2Lip repo clone failed:", e)
    ckpt_dir = repo_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gan = ckpt_dir / "wav2lip_gan.pth"
    if not gan.exists():
        # Official repo's README currently points the GAN checkpoint to this public Drive file.
        file_id = "15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk"
        try:
            subprocess.run([sys.executable, "-m", "gdown", "--id", file_id, "-O", str(gan)], check=True)
            print("Wav2Lip GAN checkpoint OK")
        except Exception as e:
            print("Wav2Lip GAN download failed:", e)
            print("Manual source: official Wav2Lip README checkpoint link.")


def syncnet():
    repo_dir = ROOT / "syncnet-python"
    if not repo_dir.exists():
        try:
            run(["git", "clone", "--depth", "1", "https://github.com/colossyan/syncnet-python.git", str(repo_dir)])
        except Exception as e:
            print("SyncNet clone failed:", e)
            return
    script = repo_dir / "download_model.sh"
    if script.exists():
        try:
            run(["bash", str(script)], cwd=str(repo_dir))
        except Exception as e:
            print("SyncNet model script failed:", e)
    # Look for a downloaded .model/.pth and copy it to the stable path used by the app.
    stable = ROOT / "syncnet" / "syncnet_v2.model"
    stable.parent.mkdir(parents=True, exist_ok=True)
    candidates = list(repo_dir.rglob("*.model")) + list(repo_dir.rglob("*.pth"))
    for src in candidates:
        if src.name.lower().startswith("syncnet") and src.stat().st_size > 10_000_000:
            try:
                shutil.copy2(src, stable)
                print("SyncNet stable weights:", stable)
                break
            except Exception as e:
                print("Copy SyncNet weights failed:", e)


if __name__ == "__main__":
    print("=== Anime Dub Studio Ultra 2: Kaggle local cache preparation ===")
    print("Model root:", ROOT)
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. Public models can cache; gated pyannote may fail.")
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print("Installing huggingface_hub...")
        run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.30"])
    hf_cache()
    wav2lip()
    syncnet()
    print("\nDONE. Put /kaggle/working/anime_dub_studio into a Kaggle Dataset if you want the Voice Bank + model cache to survive later sessions.")
    print("For fully offline inference after preparation, set: HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
