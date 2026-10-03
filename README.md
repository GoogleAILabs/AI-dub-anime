# 🎌 Anime Dub Studio Pro — Kaggle Ultra FrameSync 2

Japanese / English Anime → Hindi local dubbing pipeline designed for Kaggle GPU sessions.

## Target

- 30 GB class GPU VRAM / 30 GB RAM friendly through sequential model loading.
- 50–160+ character voice identities per project.
- Persistent character voice bank across episodes.
- Natural Hindi speech with local Chatterbox Multilingual V3 voice cloning.
- Word-aligned source timing + source-frame cue locking.
- Adaptive fast/slow TTS fitting instead of whole-episode stretching.
- Dialogue-only audio drives visual lip-sync; BGM/SFX never drives the mouth model.
- Optional strict sliding SyncNet QA; failed QA can block export.
- Original residual BGM/SFX mode preserves the source bed at original gain/timebase as closely as source separation permits.
- Resume/cache for dialogue audio and character memory.
- SRT + ASS + JSON manifest + Voice Bank ZIP.
- No gTTS, no external translation API, no cloud TTS API.

## Important lip-sync design

The pipeline follows this order:

1. WhisperX transcription.
2. Prefer aligned word boundaries over coarse sentence boundaries.
3. VAD/energy refinement.
4. Snap cue boundaries to the source video frame grid.
5. Create Hindi delivery budget using word/syllable load + local eSpeak-NG phoneme estimate when available.
6. Local Hindi dialogue polishing with Qwen3-4B.
7. Generate multiple Chatterbox candidates.
8. Rerank by prosody + immutable character voice embedding.
9. Fit only inside the exact cue using adaptive `atempo` (fast/slow). Do not hard-truncate words.
10. Build a dialogue-only, frame-locked speech track.
11. Run visual lip-sync.
12. Run full-duration validation and optional sliding SyncNet QA windows.
13. Block final export if strict lip-sync QA fails.
14. Add the final mixed audio only after the visual pass.

This avoids the common failure where Japanese and Hindi have different spoken lengths and the dubbed line drifts into the next mouth movement.

## Character memory

Keep the same `Anime / Project ID` for Episode 1, Episode 2, Episode 3, ...

Each project stores:

- immutable canonical speaker embedding
- historical prototype embeddings
- up to ~12 clean reference clips per character
- reference emotion labels
- source speaker labels observed in each episode
- character dialogue memory
- translation memory

A diarizer label such as `SPEAKER_00` is not treated as the permanent character identity. The voice bank tries to re-identify the speaker by embedding similarity and requires a score/margin before reusing an old character. If identity is not trustworthy, the segment is left unresolved rather than silently assigning the wrong voice.

## Kaggle setup

### 1) Install OS packages

```bash
!apt-get update -qq
!apt-get install -y ffmpeg git espeak-ng -qq
```

### 2) Install Python requirements

```bash
!pip install -q -r /kaggle/working/requirements.txt
```

> Keep Kaggle's existing Torch/CUDA build when possible. Reinstalling Torch can break the notebook's CUDA stack.

### 3) Set the Hugging Face token

```python
import os
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"
```

The token is for gated Hugging Face model access (for example pyannote). It is not a TTS/translation API key.

### 4) Pre-cache models and local lip-sync weights while Internet is enabled

```bash
!python /kaggle/working/prepare_kaggle_models.py
```

This caches:

- MADLAD-400 3B translation
- Qwen3-4B dialogue director
- pyannote Community-1
- SpeechBrain ECAPA
- SenseVoiceSmall
- Chatterbox Multilingual
- Wav2Lip repository + GAN checkpoint when download succeeds
- local SyncNet repository/model when its download script succeeds

### 5) Start the app

```bash
!python /kaggle/working/app.py
```

## Persisting characters for future episodes

After Episode 1, save the whole directory below as a Kaggle Dataset or otherwise persist it:

```text
/kaggle/working/anime_dub_studio/voicebank/
/kaggle/working/anime_dub_studio/projects/
```

For the next episode, restore that data and keep the same project ID. The app will reuse the character bank.

## Local/offline mode

After all model/checkpoint caches are already present:

```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

The app itself has no Google TTS call and no external runtime TTS/translation endpoint.

## Recommended quality settings

- ASR: `large-v3` on GPU.
- Natural voice candidate passes: `2` or `3`.
- Voice-identity re-rank: ON.
- Strict local SyncNet QA: ON when local SyncNet weights are available.
- Lip-sync engine: `Wav2Lip — Frame-locked Anime/CGI`.
- BGM/SFX mode: `Original residual (maximum fidelity)`.
- Same Project ID across all episodes.

## Why Wav2Lip is not claimed to be perfect

Stock Wav2Lip's official inference code detects one face per frame and writes its prediction into that single detected face region. It supports CGI faces and has a `--nosmooth` option for problematic mouth placement, but arbitrary multi-face anime scenes remain a difficult routing problem.

Therefore this build uses strict guards rather than pretending that every arbitrary anime shot is guaranteed perfect:

- frame-locked cue timing
- dialogue-only audio driver
- `--nosmooth`
- exact video/audio duration validation
- optional SyncNet verification
- strict export blocking on failed QA

For shots where the visual renderer cannot safely verify the active speaker, the safer behavior is to preserve original frames and keep the audio timing locked rather than modifying the wrong character.

## Sources / upstream references

- WhisperX: https://github.com/m-bain/whisperX
- pyannote speaker diarization: https://huggingface.co/pyannote/speaker-diarization-community-1
- Chatterbox: https://github.com/resemble-ai/chatterbox
- Wav2Lip: https://github.com/Rudrabha/Wav2Lip
- SyncNet Python fork: https://github.com/colossyan/syncnet-python
- SenseVoice: https://github.com/FunAudioLLM/SenseVoice
- Qwen3: https://github.com/QwenLM/Qwen3
- MADLAD-400: https://huggingface.co/google/madlad400-3b-mt

## Reality check

No open-source zero-shot pipeline can guarantee literal actor-identical voice reproduction or mathematically perfect phoneme-to-mouth correspondence on every arbitrary anime frame. This application is engineered around that limitation: improve identity/timing, validate locally, and block questionable visual renders instead of silently shipping an obviously mistimed dub.
