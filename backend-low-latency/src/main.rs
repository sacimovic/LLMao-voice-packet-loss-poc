use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::Serialize;
use std::{env, path::Path, sync::Arc};
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

mod asr;
mod audio;
mod degradation;
mod repair;
mod tts;

use asr::WhisperModel;
use degradation::{degrade_audio, degrade_audio_window};
use audio::{compress_silence, load_audio_mono_16k};
use repair::BedrockRepair;
use tts::TTSEngine;

const SAMPLE_RATE: u32 = 16_000;
const MIN_ASR_SAMPLES: usize = (SAMPLE_RATE as usize) / 5;


#[derive(Clone)]
struct AppState {
    whisper: Arc<WhisperModel>,
    repair: Arc<BedrockRepair>,
    tts: Arc<TTSEngine>,
}

#[derive(Serialize)]
struct ProcessResponse {
    asr_text: String,
    repaired_text: String,
    degraded_wav_b64: String,
    tts_wav_b64: String,
    combined_wav_b64: String,
    original_file_path: String,
    degraded_file_path: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env::set_var("RUST_LOG", "info");
    tracing_subscriber::fmt::init();

    info!("Loading models...");

    let candidate_models = [
        ("models/ggml-tiny.en-q5_1.bin", "tiny.en-q5_1"),
        ("models/ggml-tiny.bin", "tiny"),
        ("models/ggml-base.bin", "base"),
    ];

    let (model_path, model_label) = match env::var("WHISPER_MODEL_PATH") {
        Ok(path) => {
            info!("Whisper model path provided via WHISPER_MODEL_PATH={}", path);
            (path, "custom")
        }
        Err(_) => {
            let found = candidate_models
                .iter()
                .find(|(path, _)| Path::new(path).exists())
                .map(|(path, label)| ((*path).to_string(), *label));

            match found {
                Some((path, label)) => {
                    if label != "tiny" && label != "tiny.en-q5_1" {
                        warn!(
                            "Falling back to {} Whisper model because no tiny models were found.",
                            label
                        );
                    }
                    (path, label)
                }
                None => {
                    warn!(
                        "No Whisper models found in ./models. Please download ggml-tiny.en-q5_1.bin or ggml-tiny.bin."
                    );
                    ("models/ggml-base.bin".to_string(), "base")
                }
            }
        }
    };

    info!("Loading Whisper {:?} model from {}", model_label, model_path);

    let whisper = Arc::new(WhisperModel::new(&model_path)?);
    info!("Whisper model loaded successfully");
    
    let repair = Arc::new(BedrockRepair::new().await?);
    let tts = Arc::new(TTSEngine::new().await?);

    let state = AppState { whisper, repair, tts };

    info!("Models loaded. Starting server...");

    let app = Router::new()
        .route("/health", get(health))
        .route("/process", post(process_audio))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8000").await?;
    info!("Server running on http://127.0.0.1:8000");
    
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> &'static str {
    "OK"
}

async fn process_audio(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<ProcessResponse>, StatusCode> {
    let mut audio_bytes = None;
    let mut degrade_percent = 30;
    let mut degrade_mode = String::from("percentage");
    let mut window_ms = 40u32;
    let mut window_start_ms = 0u32;

    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap_or("").to_string();
        
        match name.as_str() {
            "file" => {
                audio_bytes = Some(field.bytes().await.unwrap().to_vec());
            }
            "degrade_percent" => {
                if let Ok(text) = field.text().await {
                    degrade_percent = text.parse().unwrap_or(30);
                }
            }
            "degrade_mode" => {
                if let Ok(text) = field.text().await {
                    degrade_mode = text;
                }
            }
            "window_ms" => {
                if let Ok(text) = field.text().await {
                    window_ms = text.parse().unwrap_or(40);
                    window_ms = text.parse().unwrap_or(0);
                }
            }
            "window_start_ms" => {
                if let Ok(text) = field.text().await {
                    window_start_ms = text.parse().unwrap_or(0);
                }
            }
            _ => {}
        }
    }

    let audio_bytes = audio_bytes.ok_or(StatusCode::BAD_REQUEST)?;

    let original_audio = load_audio_mono_16k(&audio_bytes)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let degraded = if degrade_mode == "window" {
        println!("Using window degradation: start={}ms, length={}ms", window_start_ms, window_ms);
        degradation::degrade_audio_window(&original_audio, window_start_ms, window_ms)
    } else {
        println!("Using random degradation: {}%", degrade_percent);
        degradation::degrade_audio_random(&original_audio, degrade_percent as f32)
    };

    let trimmed = compress_silence(&degraded, SAMPLE_RATE);
    let (asr_audio, trimmed_used) = if let Some(ref trimmed_audio) = trimmed {
        if trimmed_audio.len() >= MIN_ASR_SAMPLES {
            (trimmed_audio.as_slice(), true)
        } else {
            (&degraded[..], false)
        }
    } else {
        (&degraded[..], false)
    };

    let degraded_seconds = degraded.len() as f32 / SAMPLE_RATE as f32;
    let asr_seconds = asr_audio.len() as f32 / SAMPLE_RATE as f32;
    if trimmed_used {
        info!(
            "Trimmed silence for ASR: {:.2}s -> {:.2}s",
            degraded_seconds, asr_seconds
        );
    } else {
        info!("Using full audio for ASR: {:.2}s", asr_seconds);
    }

    info!("Running Whisper ASR...");
    let asr_text_result = state.whisper.transcribe(asr_audio);
    let asr_text = match asr_text_result {
        Ok(text) => text,
        Err(err) if trimmed_used => {
            warn!("Trimmed ASR failed ({}). Retrying with full audio", err);
            state.whisper.transcribe(&degraded).map_err(|fallback_err| {
                eprintln!("ASR error after fallback: {}", fallback_err);
                StatusCode::INTERNAL_SERVER_ERROR
            })?
        }
        Err(err) => {
            eprintln!("ASR error: {}", err);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };
    info!("ASR result: {}", asr_text);

    info!("Calling Bedrock API for text repair...");
    info!("Input text length: {} chars", asr_text.len());
    let repaired_text = state.repair.repair(&asr_text).await
        .map_err(|e| {
            eprintln!("Bedrock repair failed: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    info!("Bedrock repair complete. Output: {}", repaired_text);

    info!("Calling AWS Polly for TTS...");
    info!("Text to synthesize: {}", repaired_text);
    let (tts_audio, tts_sample_rate) = state.tts.synthesize(&repaired_text, Some(&audio_bytes)).await
        .map_err(|e| {
            eprintln!("TTS failed: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    info!("TTS complete. Generated {} samples at {}Hz", tts_audio.len(), tts_sample_rate);

    let degraded_wav = encode_wav_base64(&degraded, SAMPLE_RATE);
    let repaired_wav = encode_wav_base64(&tts_audio, SAMPLE_RATE);
    let combined_wav = repaired_wav.clone();

    Ok(Json(ProcessResponse {
        asr_text,
        repaired_text,
        degraded_wav_b64: degraded_wav,
        tts_wav_b64: repaired_wav,
        combined_wav_b64: combined_wav,
        original_file_path: String::from(""),
        degraded_file_path: String::from(""),
    }))
}

fn encode_wav_base64(samples: &[f32], sample_rate: u32) -> String {
    use hound::{WavSpec, WavWriter};
    use std::io::Cursor;

    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
        for &sample in samples {
            let sample_i16 = (sample * i16::MAX as f32) as i16;
            writer.write_sample(sample_i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    use base64::{Engine, engine::general_purpose};
    general_purpose::STANDARD.encode(cursor.into_inner())
}
