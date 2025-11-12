# LLMao – Voice Packet Loss Repair POC# LLMao – Voice Packet Loss Hackathon POC



AI-powered audio repair tool that fixes degraded voice audio caused by packet loss using Whisper ASR, AWS Bedrock text repair, and voice synthesis.## Overview



## OverviewLLMao is an AI-powered audio repair tool that fixes degraded voice audio caused by packet loss. It uses a three-stage pipeline:

1. **Whisper**: Transcribe degraded audio

LLMao uses a multi-stage pipeline to repair audio degraded by simulated packet loss:2. **FLAN-T5**: Repair corrupted transcriptions

3. **Coqui XTTS**: Synthesize clean audio with voice cloning

1. **Audio Degradation**: Simulate packet loss by zeroing out random 40ms frames

2. **Whisper ASR**: Transcribe degraded audio (produces text with gaps/errors)## Components

3. **AWS Bedrock**: Repair corrupted transcriptions using Amazon Nova Micro LLM

4. **TTS Synthesis**: Generate clean audio from repaired text (AWS Polly or XTTS voice cloning)### 1. Standalone Demo (Web UI)

- **Frontend**: React (Vite) - Upload audio and simulate packet loss

## Repository Structure- **Backend**: FastAPI (Python) - Process audio through repair pipeline



```### 2. Phone Call Integration (FMO Copilot)

LLMao-voice-packet-loss-poc/- Real-time copilot that monitors phone calls

├── backend-low-latency/     # 🚀 Production Rust backend (recommended)- Trigger-based repair: Say "fix the audio" to activate

│   ├── src/- Integrates with Alianza's FMO (Fabric Media Orchestrator)

│   │   ├── main.rs         # Axum HTTP server

│   │   ├── asr.rs          # Whisper ASR (whisper.cpp bindings)## Stack

│   │   ├── repair.rs       # AWS Bedrock text repair- Frontend: React (Vite)

│   │   ├── tts.rs          # AWS Polly TTS- Backend: FastAPI (Python)

│   │   ├── audio.rs        # Audio loading & degradation- ASR: Whisper

│   │   └── degradation.rs  # Packet loss simulation- LLM: Local FLAN-T5 (transformers)

│   ├── models/             # Whisper model (ggml-base.bin)- TTS: Coqui XTTS v2

│   ├── Cargo.toml- Phone Integration: FMO ICF Media SDK

│   └── README.md           # ⭐ Setup instructions

│## Backend

├── backend/                # Python FastAPI (XTTS voice cloning - experimental)Make sure you're using python 3.11 or greater (I found with python3.9 I got unresolvable dependency conflicts)

│   ├── tts_service.py      # XTTS microservice (port 8001)

│   ├── requirements.txt```bash

│   └── README.mdcd backend

│python -m venv .venv

├── frontend/               # React UI for audio upload & playback# Windows PowerShell: . .venv/Scripts/Activate.ps1

│   ├── src/# macOS/Linux: source .venv/bin/activate

│   │   ├── App.jsx         # Main UI componentpip install --upgrade pip

│   │   └── main.jsxpip install -r requirements.txt

│   ├── package.json# ffmpeg required for Whisper

│   └── README.md# macOS: brew install ffmpeg | Windows: choco install ffmpeg

│uvicorn app:app --reload --port 8000

├── icf_app/                # Phone call integration (FMO Copilot)```

│   ├── main.py

│   └── README.md## Frontend

│```bash

└── testAudio/              # Sample audio files for testingcd ../frontend

```npm install

npm run dev

## Quick Start```

Open http://localhost:5173

### Production Setup (Rust Backend)

If backend is not on localhost: create `frontend/.env` with:

**Best for**: Performance, production use, AWS integration```

VITE_API_URL=http://localhost:8000

1. **Setup AWS credentials** (see [backend-low-latency/README.md](backend-low-latency/README.md))```

   ```powershell

   # Install AWS CLI## ICF App (Phone Call Integration)

   winget install Amazon.AWSCLI

   The ICF app integrates with FMO to repair audio in real-time phone calls.

   # Configure SSO

   aws configure sso### Quick Start

   ```bash

   # Set environmentcd icf_app

   $env:AWS_REGION="us-east-1"./run.sh

   ``````



2. **Build Rust backend**Or manually:

   ```powershell```bash

   cd backend-low-latencycd icf_app

   poetry install

   # Download Whisper modelpoetry run python main.py

   mkdir models -ErrorAction SilentlyContinue```

   Invoke-WebRequest -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin" -OutFile "models\ggml-base.bin"

   The app will listen on `ws://localhost:8084`.

   # Build & run

   cargo run --release### Register with FMO

   ```

   1. Start the FMO demo server (see `https://github.com/alianza-dev/fmo-hackathon-demo`)

   Server runs on `http://127.0.0.1:8000`2. Open http://localhost:3001

3. Register the app:

3. **Start frontend**   - Name: LLMao Audio Repair

   ```powershell   - URL: ws://localhost:8084

   cd ..\frontend   - Type: Copilot

   npm install   - Receive Mode: Text and Audio

   npm run dev   - Send Mode: Audio

   ```

   ### Usage

   Open `http://localhost:5173`

1. Make a call through FMO with the LLMao app selected

### Experimental Voice Cloning (Python XTTS)2. Have a conversation

3. Say **"fix the audio"** to trigger repair

**Best for**: Voice cloning research, matching original speaker4. The app will analyze the last 5 seconds and repair any degraded audio



See [backend/README.md](backend/README.md) for XTTS microservice setup (port 8001).See [icf_app/README.md](icf_app/README.md) for detailed documentation.



> **Note**: Voice cloning integration with Rust backend is work-in-progress due to Windows audio library compatibility issues.## VS Code

Open workspace root and use the provided launch configurations under `.vscode/launch.json`.

## Architecture

### Production Stack (Rust)
- **Backend**: Axum (Rust async web framework)
- **ASR**: whisper.cpp via whisper-rs bindings
- **Text Repair**: AWS Bedrock (Amazon Nova Micro)
- **TTS**: AWS Polly (Neural, Joanna voice)
- **Audio**: symphonia, hound (native Rust audio libraries)

### Experimental Stack (Python)
- **TTS**: Coqui XTTS v2 (voice cloning)
- **Audio**: librosa, soundfile, PyTorch
- **Server**: FastAPI, uvicorn

## API Endpoints

### Main Processing Endpoint
```http
POST http://127.0.0.1:8000/process
Content-Type: multipart/form-data

Fields:
- file: Audio file (WAV, MP3, FLAC, etc.)
- degrade_percent: Packet loss percentage (0-100, default: 30)
```

**Response** (JSON):
```json
{
  "asr_text": "to be skilled at finding food shelter and protection from threats",
  "repaired_text": "To be skilled at finding food, shelter, and protection from threats.",
  "degraded_wav_b64": "UklGRiQAAABXQVZF...",
  "repaired_wav_b64": "UklGRhwCAABXQVZF..."
}
```

## AWS Requirements

### Services Used
- **AWS Bedrock**: Text repair using Amazon Nova Micro (`us.amazon.nova-micro-v1:0`)
- **AWS Polly**: Neural TTS synthesis

### Credentials Setup

**Option 1: SSO (AWS Academy)**
```powershell
aws configure sso
aws sso login --profile default
$env:AWS_REGION="us-east-1"
```

**Option 2: IAM User**
```powershell
aws configure
# Enter Access Key ID, Secret Access Key, region (us-east-1)
$env:AWS_REGION="us-east-1"
```

### Required Permissions
- `bedrock:InvokeModel` for Amazon Nova Micro
- `polly:SynthesizeSpeech` for TTS

### Costs (Approximate)
- **Bedrock Nova Micro**: $0.00035 per 1K input tokens, $0.0014 per 1K output tokens
- **Polly Neural**: $16 per 1M characters
- **Typical request**: <$0.01 per audio file

## Development

### VS Code Setup
```powershell
# Open workspace
code .

# Use launch configurations in .vscode/launch.json
# - "Rust: Launch Server" (F5)
# - "Frontend: Launch Dev Server"
```

### Testing
```powershell
# Test with sample audio
curl -X POST http://127.0.0.1:8000/process \
  -F "file=@testAudio/sample.wav" \
  -F "degrade_percent=30"
```

### Logging
```powershell
# Debug mode
$env:RUST_LOG="debug"
cargo run
```

## Phone Call Integration (ICF App)

Real-time copilot for FMO (Fabric Media Orchestrator) phone calls. See [icf_app/README.md](icf_app/README.md).

**Features**:
- Monitors phone calls in real-time
- Trigger repair by saying "fix the audio"
- Repairs last 5 seconds of degraded audio
- WebSocket integration with FMO

**Quick Start**:
```bash
cd icf_app
poetry install
poetry run python main.py
# Register at http://localhost:3001 (FMO demo UI)
```

## Troubleshooting

### "No AWS credentials found"
- Run `aws configure` or `aws sso login`
- Verify: `aws sts get-caller-identity`
- Set `$env:AWS_REGION="us-east-1"`

### "Whisper model not found"
- Download to `backend-low-latency/models/ggml-base.bin`
- URL: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin

### "Bedrock access denied"
- Ensure you're in `us-east-1` region
- Amazon Nova Micro requires no approval
- Check IAM permissions for `bedrock:InvokeModel`

### CORS errors (frontend)
- Ensure backend is running on port 8000
- Check `VITE_API_URL` in frontend/.env

## Performance

**Rust Backend** (release build):
- Audio degradation: ~10ms
- Whisper ASR (base model, CPU): ~2-5s for 5s audio
- Bedrock text repair: ~0.5-1.5s
- Polly TTS: ~0.5-2s
- **Total**: ~3-10s end-to-end

**Python XTTS** (experimental):
- Voice cloning inference: ~3-10s (CPU)
- First synthesis: ~10-30s (model loading)

## Roadmap

- [x] Rust backend with AWS Bedrock + Polly
- [x] React frontend for testing
- [x] Whisper ASR integration
- [x] Packet loss simulation
- [ ] XTTS voice cloning integration (in progress)
- [ ] Crossfade audio stitching
- [ ] GPU acceleration for Whisper
- [ ] Streaming audio support
- [ ] Real-time phone call repair

## License

MIT License - see LICENSE file

## Credits

- **Whisper**: OpenAI (https://github.com/openai/whisper)
- **XTTS**: Coqui AI (https://github.com/coqui-ai/TTS)
- **AWS Bedrock**: Amazon Web Services
- **FMO**: Alianza (https://github.com/alianza-dev/fmo-hackathon-demo)

## Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Test with sample audio
5. Submit pull request

For issues or questions, open a GitHub issue.
