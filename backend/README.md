# Backend (FastAPI)

## Setup (venv)
```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Install ffmpeg for Whisper:
- macOS: `brew install ffmpeg`
- Windows: `choco install ffmpeg`
- Linux: use your package manager

Set Anthropic key:
```bash
# macOS/Linux
export ANTHROPIC_API_KEY="sk-ant-..."
# Windows PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

Run:
```bash
uvicorn app:app --reload --port 8000
```

POST /process (multipart):
- file, degrade_percent, whisper_model, repair_model, synth_all_text

Returns JSON with ASR/repaired text and base64 WAVs.
