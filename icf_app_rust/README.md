# LLMao Audio Repair Copilot (Rust)

A Rust-based FMO copilot application that monitors phone calls and repairs degraded audio using the LLMao backend.

## Features

- **Real-time monitoring**: Listens to phone calls via FMO
- **Trigger-based repair**: Activates when user says "fix the audio" or "repair audio"
- **Audio buffering**: Keeps last 5 seconds of audio from both parties
- **Backend integration**: Uses the Rust backend for Whisper ASR, AWS Bedrock text repair, and AWS Polly TTS

## Prerequisites

1. **FMO Demo Server** running (see `fmo-hackathon-demo`)
2. **LLMao Rust Backend** running on port 8000 with AWS credentials configured
3. Rust toolchain installed

## Building

```bash
cd icf_app_rust
cargo build --release
```

## Running

```bash
# Set backend URL (optional, defaults to http://localhost:8000)
export BACKEND_URL=http://localhost:8000

# Run the copilot
cargo run --release
```

Or use the binary directly:

```bash
./target/release/llmao_audio_repair_copilot
```

## Usage

1. Start the FMO demo server (port 3001)
2. Start the LLMao Rust backend (port 8000) with AWS credentials
3. Start this copilot (port 8084)
4. Register the copilot in FMO web UI:
   - Name: LLMao Audio Repair
   - URL: ws://localhost:8084
   - Type: Copilot
   - Receive Mode: Text and Audio
   - Send Mode: Audio
5. Make a call through FMO with the copilot selected
6. During the call, say **"fix the audio"** to trigger repair
7. The copilot will analyze the last 5 seconds and inject repaired audio

## Configuration

- `BACKEND_URL`: URL of the LLMao backend (default: `http://localhost:8000`)
- `RUST_LOG`: Log level (default: `info`, options: `debug`, `trace`)
- Port: Fixed at `8084`

## Architecture

```
Phone Call (FMO)
    ↓
Copilot (Port 8084)
    ↓ (on trigger)
Rust Backend (Port 8000)
    ↓
AWS Bedrock + Polly
    ↓
Repaired Audio → Injected to Call
```

## Trigger Phrases

- "fix the audio"
- "repair the audio"
- "audio is broken"

## Logging

Set `RUST_LOG` for detailed logs:

```bash
RUST_LOG=debug cargo run --release
```

## Files

- `src/main.rs` - Main copilot logic and FMO integration
- `src/audio_buffer.rs` - Rolling audio buffer (5 second window)
- `src/audio_repair.rs` - Backend API client and audio processing
- `Cargo.toml` - Dependencies and project metadata
