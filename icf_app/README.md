# LLMao Audio Repair ICF App

A real-time audio repair app for phone calls that integrates with the FMO (Fabric Media Orchestrator) demo server. This app listens to ongoing phone conversations and repairs audio degradation when triggered by specific phrases.

## How It Works

1. **Passive Monitoring**: The app listens to both parties in a call without interfering
2. **Trigger Detection**: When someone says "fix the audio" or similar phrases, it activates
3. **Audio Analysis**: Analyzes the last 5 seconds of audio for packet loss/degradation
4. **Repair Pipeline**: 
   - Transcribes degraded audio with Whisper
   - Repairs the transcription with FLAN-T5
   - Synthesizes clean audio with Coqui XTTS
5. **Audio Injection**: Plays the repaired audio back into the call

## Features

- **On-Demand Repair**: Only processes audio when explicitly requested (solves latency issues)
- **Trigger Phrases**: Responds to:
  - "fix the audio"
  - "repair audio"
  - "audio is broken"
- **Bidirectional**: Automatically switches to bidirectional mode when needed
- **Smart Buffering**: Keeps a rolling 5-second buffer of recent audio
- **Packet Loss Detection**: Identifies silence gaps that indicate degradation
- **Debug Recordings**: Saves repaired audio samples for analysis

## Setup

### Prerequisites

1. The FMO hackathon demo server running (see `../../fmo-hackathon-demo`)
2. Python 3.9 or higher
3. Poetry for dependency management

### Installation

```bash
cd icf_app

# Install dependencies
poetry install

# Note: First run will download models (Whisper, FLAN-T5, XTTS)
# This may take several minutes and requires ~2-3GB disk space
```

### Configuration

Edit `main.py` to adjust:

- `TRIGGER_PATTERNS`: Customize trigger phrases
- `BUFFER_DURATION_MS`: Change how much audio history to keep (default: 5000ms)
- `COPILOT_PORT`: Change WebSocket port (default: 8084)
- Whisper model size in `AudioRepairer` (default: "tiny" for speed)

## Usage

### 1. Start the FMO Demo Server

```bash
cd ../../fmo-hackathon-demo
./start-all-dev.sh
```

The server should be accessible at http://localhost:3001

### 2. Start the ICF App

```bash
cd ../LLMao-voice-packet-loss-poc/icf_app
poetry run python main.py
```

You should see:
```
======================================================================
🎙️  LLMao Audio Repair Copilot
======================================================================
📡 Listening on ws://localhost:8084
🎯 Trigger phrases: fix the audio, repair audio, audio is broken
💾 Debug recordings: /home/rnd/hackathon/LLMao-voice-packet-loss-poc/icf_app/recordings
```

### 3. Register the App in FMO

1. Open http://localhost:3001 in your browser
2. Click "Register App"
3. Fill in:
   - **Name**: LLMao Audio Repair
   - **URL**: ws://localhost:8084
   - **Type**: Copilot
   - **Receive Mode**: Text (with partial transcription updates checked)
   - **Send Mode**: Text
   - **Default Start Mode**: Listen only
   - **Default Mix Mode**: Duck

4. Click "Register"

### 4. Test the App

1. In the FMO web interface, select "LLMao Audio Repair" from the app dropdown
2. Click "Make Call"
3. Have another browser/device pick up the call
4. Have a conversation
5. Say one of the trigger phrases: **"Can you fix the audio?"**
6. The app will:
   - Acknowledge the request
   - Analyze recent audio
   - Repair any degraded segments
   - Inject repaired audio into the call

## Architecture

```
Phone Call (FMO)
      ↓
   Copilot (main.py)
      ↓
   ┌─────────────────────┐
   │ AudioBuffer         │  ← Stores last 5s of audio
   │ PacketLossDetector  │  ← Detects degradation
   │ AudioRepairer       │  ← Repair pipeline
   └─────────────────────┘
            ↓
   ┌─────────────────────┐
   │ Backend (../backend)│
   │ - Whisper (ASR)     │
   │ - FLAN-T5 (Repair)  │
   │ - XTTS (TTS)        │
   └─────────────────────┘
```

## Performance Considerations

- **Latency**: Full repair takes 3-10 seconds depending on hardware
- **Model Sizes**: Currently uses:
  - Whisper: "tiny" (fastest, ~39M params)
  - FLAN-T5: "small" (~80M params)
  - XTTS: Large model for quality
- **Memory**: Requires ~2-3GB RAM during processing
- **GPU**: Will use GPU if available (recommended for better performance)

## Troubleshooting

### Models not downloading
```bash
# Manually download models first
poetry run python -c "import whisper; whisper.load_model('tiny')"
poetry run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/flan-t5-small')"
```

### Port already in use
Change `COPILOT_PORT` in `main.py` and update the registration in FMO.

### No audio detected
Check the logs - ensure audio is being buffered:
```
📊 Analyzing audio - Caller: 5000ms, Callee: 5000ms
```

### TTS fails
Ensure you have a valid speaker reference. The copilot uses the buffered audio itself as reference.

## Development

### Project Structure

```
icf_app/
├── main.py              # Main ICF application
├── audio_processor.py   # Audio processing utilities
├── pyproject.toml       # Dependencies
├── README.md           # This file
└── recordings/         # Debug recordings (created at runtime)
```

### Adding New Trigger Phrases

Edit `TRIGGER_PATTERNS` in `main.py`:

```python
TRIGGER_PATTERNS = [
    r'\bfix\s+(?:the\s+)?audio\b',
    r'\brepair\s+(?:the\s+)?audio\b',
    r'\baudio\s+(?:is\s+)?broken\b',
    r'\bhelp\s+with\s+sound\b',  # Add your own
]
```

### Adjusting Buffer Duration

Longer buffers capture more context but use more memory:

```python
BUFFER_DURATION_MS = 10000  # 10 seconds
```

## Future Enhancements

- [ ] Automatic degradation detection (without trigger phrase)
- [ ] Real-time streaming repair (lower latency)
- [ ] Configurable quality vs. speed tradeoffs
- [ ] Multi-language support
- [ ] Custom voice cloning profiles
- [ ] Integration with Azure Speech Services as alternative

## License

Part of the RenAIssance Hackathon project.
