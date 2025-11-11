# LLMao – Voice Packet Loss Hackathon POC

## Overview

LLMao is an AI-powered audio repair tool that fixes degraded voice audio caused by packet loss. It uses a three-stage pipeline:
1. **Whisper**: Transcribe degraded audio
2. **FLAN-T5**: Repair corrupted transcriptions
3. **Coqui XTTS**: Synthesize clean audio with voice cloning

## Components

### 1. Standalone Demo (Web UI)
- **Frontend**: React (Vite) - Upload audio and simulate packet loss
- **Backend**: FastAPI (Python) - Process audio through repair pipeline

### 2. Phone Call Integration (FMO Copilot)
- Real-time copilot that monitors phone calls
- Trigger-based repair: Say "fix the audio" to activate
- Integrates with Alianza's FMO (Fabric Media Orchestrator)

## Stack
- Frontend: React (Vite)
- Backend: FastAPI (Python)
- ASR: Whisper
- LLM: Local FLAN-T5 (transformers)
- TTS: Coqui XTTS v2
- Phone Integration: FMO ICF Media SDK

## Backend
```bash
cd backend
python -m venv .venv
# Windows PowerShell: . .venv/Scripts/Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# ffmpeg required for Whisper
# macOS: brew install ffmpeg | Windows: choco install ffmpeg
uvicorn app:app --reload --port 8000
```

## Frontend
```bash
cd ../frontend
npm install
npm run dev
```
Open http://localhost:5173

If backend is not on localhost: create `frontend/.env` with:
```
VITE_API_URL=http://localhost:8000
```

## ICF App (Phone Call Integration)

The ICF app integrates with FMO to repair audio in real-time phone calls.

### Quick Start
```bash
cd icf_app
./run.sh
```

Or manually:
```bash
cd icf_app
poetry install
poetry run python main.py
```

The app will listen on `ws://localhost:8084`.

### Register with FMO

1. Start the FMO demo server (see `../fmo-hackathon-demo`)
2. Open http://localhost:3001
3. Register the app:
   - Name: LLMao Audio Repair
   - URL: ws://localhost:8084
   - Type: Copilot
   - Receive Mode: Text (with partial transcription)
   - Send Mode: Text

### Usage

1. Make a call through FMO with the LLMao app selected
2. Have a conversation
3. Say **"fix the audio"** to trigger repair
4. The app will analyze the last 5 seconds and repair any degraded audio

See [icf_app/README.md](icf_app/README.md) for detailed documentation.

## VS Code
Open workspace root and use the provided launch configurations under `.vscode/launch.json`.
