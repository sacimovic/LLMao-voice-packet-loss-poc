# LLMao – Voice Packet Loss Hackathon POC

## Stack
- Frontend: React (Vite)
- Backend: FastAPI (Python)
- ASR: Whisper
- LLM: Local FLAN-T5 (transformers)
- TTS: Coqui XTTS v2

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
# Set Anthropic API key:
#  macOS/Linux: export ANTHROPIC_API_KEY="sk-ant-..."
#  Windows PS:  $env:ANTHROPIC_API_KEY="sk-ant-..."
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

## VS Code
Open workspace root and use the provided launch configurations under `.vscode/launch.json`.
