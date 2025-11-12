# Python Backend (XTTS Voice Cloning Microservice)# Backend (FastAPI)



> **Note**: This is experimental voice cloning service. The main production backend is the Rust implementation in `backend-low-latency/`.## Setup (venv)

```bash

## Overviewpython -m venv .venv

# Windows PowerShell

FastAPI microservice providing XTTS voice cloning for TTS synthesis. Can be called by the Rust backend or used standalone.. .venv/Scripts/Activate.ps1

# macOS/Linux

## Featuressource .venv/bin/activate



- **Voice Cloning**: Uses Coqui XTTS v2 model to clone speaker voice from audio samplepip install --upgrade pip

- **Multi-language**: Supports English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Koreanpip install -r requirements.txt

- **Fast Inference**: Cached model loading, CPU-optimized PyTorch```



## PrerequisitesInstall ffmpeg for Whisper:

- macOS: `brew install ffmpeg`

1. **Python 3.11+**: XTTS requires recent Python- Windows: `choco install ffmpeg`

2. **PyTorch 2.9+**: Installed automatically with TTS package- Linux: use your package manager

3. **FFmpeg**: For audio processing (optional with librosa fallback)

4. **~2GB disk space**: For XTTS model downloadSet Anthropic key:

```bash

## Setup# macOS/Linux

export ANTHROPIC_API_KEY="sk-ant-..."

```powershell# Windows PowerShell

# Create virtual environment$env:ANTHROPIC_API_KEY="sk-ant-..."

python -m venv .venv```



# Activate (Windows PowerShell)Run:

.\.venv\Scripts\Activate.ps1```bash

# Or (macOS/Linux)uvicorn app:app --reload --port 8000

# source .venv/bin/activate```



# Upgrade pipPOST /process (multipart):

pip install --upgrade pip- file, degrade_percent, whisper_model, repair_model, synth_all_text



# Install dependenciesReturns JSON with ASR/repaired text and base64 WAVs.

pip install -r requirements.txt
```

### Install FFmpeg (Optional but Recommended)

- **Windows**: `winget install FFmpeg` or download from https://ffmpeg.org/download.html
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian)

> **Note**: If FFmpeg is not available, the service will use librosa for audio processing (slower but works).

## Run Service

```powershell
# Start XTTS microservice on port 8001
python tts_service.py
```

Service starts on `http://127.0.0.1:8001`

First run will download the XTTS model (~2GB) from Hugging Face.

## API Endpoints

### Synthesize Speech (Voice Cloning)

```bash
POST http://127.0.0.1:8001/synthesize
Content-Type: multipart/form-data

Fields:
- text: Text to synthesize (required)
- speaker_audio: Audio file of speaker's voice (required, WAV/MP3/etc.)
- language: Language code (default: "en")
```

**Response**: 16kHz mono WAV audio bytes

**Example** (curl):
```bash
curl -X POST http://127.0.0.1:8001/synthesize \
  -F "text=Hello, this is a voice cloning test" \
  -F "speaker_audio=@speaker_sample.wav" \
  -F "language=en" \
  --output cloned_voice.wav
```

**Supported Languages**: `en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, `tr`, `ru`, `nl`, `cs`, `ar`, `zh-cn`, `ja`, `hu`, `ko`

## Integration with Rust Backend

The Rust backend in `backend-low-latency/` can be modified to call this service:

1. Start this service: `python tts_service.py`
2. In Rust `tts.rs`, POST to `http://127.0.0.1:8001/synthesize` with multipart form
3. Parse WAV response and convert to `Vec<f32>`

See `backend-low-latency/src/tts.rs` comments for integration example.

## Troubleshooting

### PyTorch Pickle Loading Error
```
UnpicklingError: Weights only load failed
```
**Fix**: Update `tts_service.py` with `weights_only=False` workaround (already implemented).

### torchcodec Windows DLL Errors
```
RuntimeError: Could not load libtorchcodec_core*.dll
```
**Fix**: Service uses librosa for audio loading (no torchcodec needed). If torchcodec is installed, uninstall it:
```powershell
pip uninstall torchcodec -y
```

### XTTS Model Download Fails
```
ConnectionError during model download
```
**Fix**: 
- Check internet connection
- Model downloads from Hugging Face (~2GB)
- On slow connections, download may timeout - restart service to resume

### Low Quality Voice Cloning
- Provide 3-10 seconds of clean speaker audio (no background noise)
- Use WAV format for speaker sample (better quality than MP3)
- Ensure speaker audio is mono or stereo (not multi-channel)
- Try different speaker samples if quality is poor

## Performance

- **First synthesis**: ~10-30s (model loading + inference)
- **Subsequent calls**: ~3-10s (model cached in memory)
- **CPU vs GPU**: CPU-only inference (faster with CUDA not implemented yet)

## Known Issues

1. **Voice cloning quality**: May not perfectly match original speaker
2. **Windows compatibility**: torchcodec issues resolved by using librosa
3. **Long text**: Synthesis time increases linearly with text length

## Development

```powershell
# Run with auto-reload
uvicorn tts_service:app --reload --port 8001

# Test endpoint
curl http://127.0.0.1:8001/docs
# Opens interactive API docs
```

## Future Work

- [ ] Fix librosa audio loading edge cases
- [ ] Add GPU support for faster inference
- [ ] Implement voice presets (cache speaker embeddings)
- [ ] Add streaming synthesis for long text
- [ ] Quality improvements (speaker embedding tuning)
