# Low-Latency Rust Backend

High-performance voice processing pipeline with AWS Bedrock text repair and Polly TTS.

## Architecture

```
Audio Upload → ASR (Whisper) → Packet Loss Simulation → Text Repair (Bedrock) → TTS (Polly) → WAV Response
```

## Features

- **ASR**: Whisper.cpp bindings for fast speech-to-text (ggml-base.bin model)
- **Text Repair**: AWS Bedrock with Amazon Nova Micro for fixing transcription errors
- **TTS**: AWS Polly Neural engine (Joanna voice)
- **Audio Processing**: Multi-format support via Symphonia (WAV, MP3, FLAC, AAC, Vorbis)
- **Performance**: Release builds with LTO optimization

## Prerequisites

1. **Rust** (1.70+):
   ```powershell
   # Install via rustup
   winget install Rustlang.Rustup
   ```

2. **AWS CLI** (configured with credentials):
   ```powershell
   # Install AWS CLI v2
   msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi

   # Configure credentials
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, and region (us-east-1)
   ```

3. **Whisper Model** (ggml-base.bin, ~147MB):
   ```powershell
   # Download to models/ directory
   cd models
   curl -L -o ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
   ```

## AWS Setup

### Required Permissions

Your AWS IAM user/role needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "polly:SynthesizeSpeech"
      ],
      "Resource": "*"
    }
  ]
}
```

### AWS Credentials

Set environment variables (or use `aws configure`):

```powershell
$env:AWS_ACCESS_KEY_ID="AKIA..."
$env:AWS_SECRET_ACCESS_KEY="..."
$env:AWS_DEFAULT_REGION="us-east-1"
```

## Build & Run

```powershell
# Build release binary
cargo build --release

# Run server (listens on http://localhost:8000)
cargo run --release
```

First build takes ~5-10 minutes (compiles 341+ dependencies).

## API Endpoints

### Health Check

```bash
GET /health
```

**Response**: `200 OK`

### Process Audio

```bash
POST /process
Content-Type: multipart/form-data

# Form fields:
- file: Audio file (WAV, MP3, FLAC, AAC, etc.)
- packet_loss_percent: Optional, default 0.0 (range: 0-100)
```

**Response**:
```json
{
  "asr_text": "original transcription",
  "repaired_text": "corrected by Bedrock",
  "audio_base64": "UklGRi4EAABXQVZF..." 
}
```

The `audio_base64` field contains a base64-encoded WAV file (16kHz mono PCM).

## Example Usage

**PowerShell**:
```powershell
curl.exe -X POST http://localhost:8000/process `
  -F "file=@recording.wav" `
  -F "packet_loss_percent=20"
```

**Python**:
```python
import requests

with open('recording.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process',
        files={'file': f},
        data={'packet_loss_percent': 20}
    )

result = response.json()
print(f"Original: {result['asr_text']}")
print(f"Repaired: {result['repaired_text']}")

# Decode audio
import base64
wav_bytes = base64.b64decode(result['audio_base64'])
with open('output.wav', 'wb') as out:
    out.write(wav_bytes)
```

## Troubleshooting

### AWS Credentials Not Found

**Error**: `failed to load AWS config`

**Fix**: Run `aws configure` or set environment variables:
```powershell
$env:AWS_ACCESS_KEY_ID="..."
$env:AWS_SECRET_ACCESS_KEY="..."
```

### Bedrock Access Denied

**Error**: `AccessDeniedException`

**Fix**: Amazon Nova Micro is available by default in us-east-1. Ensure your IAM user has `bedrock:InvokeModel` permission.

### Whisper Model Not Found

**Error**: `Failed to load Whisper model`

**Fix**: Download ggml-base.bin to `models/` directory:
```powershell
cd models
curl -L -o ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
```

### Port Already in Use

**Error**: `Address already in use`

**Fix**: Change port in `src/main.rs` or kill process on port 8000:
```powershell
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process
```

## Performance

- **ASR**: ~100-200ms (Whisper base model, CPU)
- **Bedrock**: ~500-1000ms (Amazon Nova Micro)
- **Polly**: ~300-500ms (Neural voice)
- **Total latency**: ~1-2s for 5-second audio clips

## Directory Structure

```
backend-low-latency/
├── Cargo.toml           # Rust dependencies
├── Cargo.lock           # Locked versions
├── .gitignore           # Excludes target/, models/*.bin
├── README.md            # This file
├── models/
│   └── ggml-base.bin    # Whisper model (download separately)
└── src/
    ├── main.rs          # Axum HTTP server
    ├── asr.rs           # Whisper ASR wrapper
    ├── audio.rs         # Audio loading/resampling
    ├── degradation.rs   # Packet loss simulation
    ├── repair.rs        # AWS Bedrock text repair
    └── tts.rs           # AWS Polly TTS
```

## License

MIT
