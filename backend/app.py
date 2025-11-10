#!/usr/bin/env python3
import io
import base64
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from audio_utils import (
    degrade_audio_simulated_loss,
    degrade_audio_zero_window,
    transcribe_whisper,
    repair_text_with_local_model,
    synthesize_xtts,
    stitch_simple_crossfade,
    load_audio_mono_16k,
    write_wav_bytes,
)

app = FastAPI(title="Voice Packet-Loss Hackathon API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessResponse(BaseModel):
    asr_text: str
    repaired_text: str
    degraded_wav_b64: str
    tts_wav_b64: str
    combined_wav_b64: str

@app.post("/process", response_model=ProcessResponse)
async def process_audio(
    file: UploadFile = File(...),
    degrade_percent: int = Form(30), # 0..100
    degrade_mode: str = Form("percentage"),
    window_ms: int = Form(40),
    window_start_ms: int = Form(0),
    whisper_model: str = Form("base"),
    repair_model: str = Form("google/flan-t5-small"),
    synth_all_text: bool = Form(True)
):
    try:
        raw = await file.read()
        sr, y_orig = load_audio_mono_16k(io.BytesIO(raw))
        duration_ms = int(len(y_orig) / sr * 1000) if len(y_orig) and sr else 0
        mode = (degrade_mode or "percentage").strip().lower()

        if mode == "window":
            max_window = max(40, duration_ms // 3) if duration_ms else max(40, window_ms)
            window_ms = max(40, int(window_ms))
            if max_window:
                window_ms = min(window_ms, max_window)
            start_ms = max(0, int(window_start_ms))
            if duration_ms:
                start_ms = min(start_ms, max(duration_ms - window_ms, 0))
            y_degraded = degrade_audio_zero_window(y_orig, sr, start_ms=start_ms, window_ms=window_ms)
        else:
            degrade_ratio = max(0, min(100, int(degrade_percent))) / 100.0
            y_degraded = degrade_audio_simulated_loss(y_orig, sr, loss_ratio=degrade_ratio, chunk_ms=40)

        asr_text = transcribe_whisper(y_degraded, sr, model_size=whisper_model)
        repaired_text = repair_text_with_local_model(asr_text, model_name=repair_model)

        tts_wav = synthesize_xtts(repaired_text, speaker_wav_bytes=raw)

        if synth_all_text:
            combined_wav = tts_wav
        else:
            combined_wav = stitch_simple_crossfade(y_degraded, sr, tts_wav, crossfade_ms=25)

        degraded_b = write_wav_bytes(y_degraded, sr)
        combined_b = combined_wav if isinstance(combined_wav, (bytes, bytearray)) else write_wav_bytes(combined_wav, sr)
        tts_b = tts_wav if isinstance(tts_wav, (bytes, bytearray)) else write_wav_bytes(tts_wav, sr)

        return ProcessResponse(
            asr_text=asr_text,
            repaired_text=repaired_text,
            degraded_wav_b64=base64.b64encode(degraded_b).decode("ascii"),
            tts_wav_b64=base64.b64encode(tts_b).decode("ascii"),
            combined_wav_b64=base64.b64encode(combined_b).decode("ascii"),
        )
    except Exception as e:
        traceback.print_exc()
        return ProcessResponse(
            asr_text=f"ERROR: {e}",
            repaired_text="",
            degraded_wav_b64="",
            tts_wav_b64="",
            combined_wav_b64="",
        )

@app.get("/health")
def health():
    return {"status": "ok"}
