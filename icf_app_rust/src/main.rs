use std::sync::{Arc, Mutex};
use std::path::PathBuf;

use async_trait::async_trait;
use icf_media_sdk::{
    create_copilot_app, AppOptions, CallCleanup, CopilotCallHandle, CopilotMode,
    MediaHandler, SdkResult, SpeakingParty, CopilotSpeechTarget, MixMode,
};
use log::{error, info, warn};
use regex::Regex;
use tokio::{signal, fs};

mod audio_buffer;
mod audio_repair;
mod audio_degradation;
mod background_repair;

use audio_buffer::AudioBuffer;
use audio_repair::AudioRepairer;

use background_repair::StreamType;

const COPILOT_PORT: u16 = 8084;
const BUFFER_DURATION_MS: u64 = 10000; // Keep last 10 seconds of audio

// Trigger phrases to activate audio repair
static TRIGGER_PATTERNS: &[&str] = &[
    r"\bfix\s+(?:the\s+)?audio\b",
    r"\brepair\s+(?:the\s+)?audio\b",
    r"\baudio\s+(?:is\s+)?broken\b",
];

struct LLMaoRepairCopilot {
    backend_url: String,
    recordings_dir: PathBuf,
}

impl LLMaoRepairCopilot {
    fn new() -> Self {
        let backend_url = std::env::var("BACKEND_URL")
            .unwrap_or_else(|_| "http://localhost:8000".to_string());
        let recordings_dir = PathBuf::from("recordings");
        
        Self {
            backend_url,
            recordings_dir,
        }
    }
}

#[async_trait]
impl MediaHandler for LLMaoRepairCopilot {
    type Call = CopilotCallHandle;

    // Use default health check implementation from SDK
    // This is equivalent to Python's behavior (no custom handler)

    async fn on_call(&self, call: Self::Call) -> SdkResult<Option<CallCleanup>> {
        info!(
            "📞 Call started: {} → {}",
            call.calling_number(),
            call.called_number()
        );

        // Create recordings directory if it doesn't exist
        if let Err(e) = fs::create_dir_all(&self.recordings_dir).await {
            error!("Failed to create recordings directory: {}", e);
        }

        let sample_rate = call.audio_sampling_rate_khz() * 1000;
        
        // Audio buffers for both parties
        let caller_buffer = Arc::new(Mutex::new(AudioBuffer::new(sample_rate as usize, BUFFER_DURATION_MS)));
        let callee_buffer = Arc::new(Mutex::new(AudioBuffer::new(sample_rate as usize, BUFFER_DURATION_MS)));

        let audio_repairer = match AudioRepairer::new(self.backend_url.clone()) {
            Ok(repairer) => Arc::new(repairer),
            Err(e) => {
                error!("Failed to create audio repairer: {}", e);
                return Err(icf_media_sdk::SdkError::Call(icf_media_sdk::MediaCallError {
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    error_code: -1,
                    error_text: "Failed to create audio repairer".to_string(),
                    error_details: Some(e.to_string()),
                }));
            }
        };
        
        let is_processing = Arc::new(Mutex::new(false));
        let repair_count = Arc::new(Mutex::new(0));

        // Compile trigger patterns
        let trigger_regexes: Vec<Regex> = TRIGGER_PATTERNS
            .iter()
            .filter_map(|pattern| Regex::new(pattern).ok())
            .collect();

        // Handle audio data from both parties separately
        call.on_caller_audio(Some({
            let caller_buf = Arc::clone(&caller_buffer);
            
            Arc::new(move |audio_bytes| {
                // Convert bytes to i16 samples (little-endian PCM)
                let samples: Vec<i16> = audio_bytes
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                
                // Debug: Log every 100 chunks
                static CALLER_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let count = CALLER_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if count % 100 == 0 {
                    info!("🎤 Received CALLER audio chunk {} ({} samples)", count, samples.len());
                }
                
                if let Ok(mut buf) = caller_buf.lock() {
                    buf.add_samples(&samples);
                }
            })
        }));

        call.on_callee_audio(Some({
            let callee_buf = Arc::clone(&callee_buffer);
            
            Arc::new(move |audio_bytes| {
                // Convert bytes to i16 samples (little-endian PCM)
                let samples: Vec<i16> = audio_bytes
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                
                // Debug: Log every 100 chunks
                static CALLEE_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let count = CALLEE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if count % 100 == 0 {
                    info!("🎤 Received CALLEE audio chunk {} ({} samples)", count, samples.len());
                }
                
                if let Ok(mut buf) = callee_buf.lock() {
                    buf.add_samples(&samples);
                }
            })
        }));

        // Handle partial utterances to detect trigger phrase
        call.on_partial_utterance(Some({
            let call_ref = Arc::clone(&call);
            let caller_buf = Arc::clone(&caller_buffer);
            let callee_buf = Arc::clone(&callee_buffer);
            let repairer = Arc::clone(&audio_repairer);
            let processing_flag = Arc::clone(&is_processing);
            let count = Arc::clone(&repair_count);
            
            Arc::new(move |partial, speaking_party| {
                let text = &partial.text;
                info!("💬 {}: \"{}\"", speaking_party, text);

                // Check for trigger phrase
                let text_lower = text.to_lowercase();
                let triggered = trigger_regexes
                    .iter()
                    .any(|re| re.is_match(&text_lower));

                if triggered {
                    info!("🎯 Trigger detected from {}: \"{}\"", speaking_party, text);
                    
                    // Check if already processing
                    {
                        let mut processing = processing_flag.lock().unwrap();
                        if *processing {
                            warn!("⚠️ Already processing, skipping...");
                            return;
                        }
                        *processing = true;
                    }

                    // Determine which stream to repair (opposite of speaker)
                    // and who should hear the fixed audio (the speaker who requested it)
                    let target_stream = match speaking_party {
                        SpeakingParty::Caller => StreamType::Callee,
                        SpeakingParty::Callee => StreamType::Caller,
                    };
                    let send_to = match speaking_party {
                        SpeakingParty::Caller => CopilotSpeechTarget::Caller,
                        SpeakingParty::Callee => CopilotSpeechTarget::Callee,
                    };

                    // Get the appropriate buffer
                    let buffer = match target_stream {
                        StreamType::Caller => Arc::clone(&caller_buf),
                        StreamType::Callee => Arc::clone(&callee_buf),
                    };

                    // Spawn repair task
                    let call_clone = Arc::clone(&call_ref);
                    let repairer_clone = Arc::clone(&repairer);
                    let processing = Arc::clone(&processing_flag);
                    let repair_count = Arc::clone(&count);

                    tokio::spawn(async move {
                        match handle_repair_request(
                            Arc::clone(&call_clone),
                            buffer,
                            repairer_clone,
                            target_stream,
                            send_to,
                        ).await {
                            Ok(_) => {
                                let mut count = repair_count.lock().unwrap();
                                *count += 1;
                                info!("✅ Repair complete (total: {})", *count);
                            }
                            Err(e) => {
                                error!("❌ Repair failed: {}", e);
                            }
                        }
                        
                        // Reset processing flag
                        *processing.lock().unwrap() = false;
                    });
                }
            })
        }));

        let call_id = call.call_id().to_string();
        let cleanup = async move {
            info!("🤖 LLMao copilot finished for call {}", call_id);
        };

        Ok(Some(Box::pin(cleanup)))
    }
}

async fn handle_repair_request(
    call: CopilotCallHandle,
    buffer: Arc<Mutex<AudioBuffer>>,
    repairer: Arc<AudioRepairer>,
    target_stream: StreamType,
    send_to: CopilotSpeechTarget,
) -> anyhow::Result<()> {
    info!("🔧 Starting repair for {} stream (sending to {:?})", target_stream, send_to);

    // Switch to bidirectional mode if needed
    if call.call_mode() == CopilotMode::ListenOnly {
        info!("🔄 Switching to bidirectional mode...");
        call.set_bidirectional().await?;
        info!("✅ Now bidirectional");
    }

    // Get the last 10 seconds of audio from buffer
    let audio_samples = {
        let buf = buffer.lock().unwrap();
        buf.get_samples()
    };

    if audio_samples.is_empty() {
        info!("❌ No audio in buffer for {}", target_stream);
        if let Err(e) = call.say(
            "No audio available to repair.",
            send_to,
            Some(MixMode::Override)
        ) {
            warn!("⚠️ Failed to announce no audio: {}", e);
        }
        return Ok(());
    }

    let duration_s = audio_samples.len() as f64 / 16000.0;
    info!("📦 Retrieved {:.1}s of audio from buffer", duration_s);

    // Announce repair starting
    match call.say(
        "Analyzing and repairing audio...",
        send_to,
        Some(MixMode::Override)
    ) {
        Ok(speech_handle) => {
            if let Err(e) = speech_handle.await_completion().await {
                warn!("⚠️ Failed to wait for announcement: {}", e);
            }
        }
        Err(e) => {
            warn!("⚠️ Failed to announce: {}", e);
        }
    }

    // Repair the audio
    info!("🔄 Sending {:.1}s of audio for repair...", duration_s);
    let repaired_samples = repairer.repair_audio(&audio_samples).await?;
    
    let repaired_duration = repaired_samples.len() as f64 / 16000.0;
    info!("✅ Received {:.1}s of repaired audio", repaired_duration);

    // Announce playback
    match call.say(
        "Playing repaired audio.",
        send_to,
        Some(MixMode::Override)
    ) {
        Ok(speech_handle) => {
            if let Err(e) = speech_handle.await_completion().await {
                warn!("⚠️ Failed to wait for announcement: {}", e);
            }
        }
        Err(e) => {
            warn!("⚠️ Failed to announce playback: {}", e);
        }
    }

    // Play back the repaired audio
    send_repaired_audio(&call, &repaired_samples, send_to).await?;

    info!("✅ Repaired audio sent successfully");
    Ok(())
}

async fn send_repaired_audio(
    call: &CopilotCallHandle,
    samples: &[i16],
    target: CopilotSpeechTarget,
) -> anyhow::Result<()> {
    const CHUNK_SIZE_MS: usize = 32;  // Match browser audio packet size
    const SAMPLE_RATE: usize = 16000;
    let chunk_samples = (SAMPLE_RATE * CHUNK_SIZE_MS) / 1000;

    let duration_s = samples.len() as f64 / SAMPLE_RATE as f64;
    info!("🔊 Sending {:.1}s of audio in {}ms chunks", duration_s, CHUNK_SIZE_MS);

    let chunks: Vec<_> = samples.chunks(chunk_samples).collect();
    info!("📦 Total chunks to send: {}", chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        let audio_bytes = samples_to_bytes(chunk);
        
        match target {
            CopilotSpeechTarget::Caller => {
                call.send_audio_to_caller(&audio_bytes, Some(MixMode::Override))?;
            }
            CopilotSpeechTarget::Callee => {
                call.send_audio_to_callee(&audio_bytes, Some(MixMode::Override))?;
            }
            _ => {
                warn!("⚠️ Broadcast target not supported for audio playback");
            }
        }

        if (i + 1) % 50 == 0 {
            info!("   Sent chunk {} / {}", i + 1, chunks.len());
        }
        
        // Sleep for the actual duration of the chunk to match real-time playback
        // Use slightly less than the chunk duration to avoid gaps
        tokio::time::sleep(tokio::time::Duration::from_millis((CHUNK_SIZE_MS as u64) - 2)).await;
    }

    info!("✅ All audio chunks sent");
    Ok(())
}

// Convert i16 samples to u8 bytes (little-endian PCM)
fn samples_to_bytes(samples: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

#[tokio::main]
async fn main() -> SdkResult<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    info!("🚀 LLMao Audio Repair Copilot starting...");
    
    let handler = LLMaoRepairCopilot::new();
    let backend_url = handler.backend_url.clone();
    let app = create_copilot_app(handler, AppOptions::default());

    info!("🤖 LLMao Copilot ready at ws://localhost:{}", COPILOT_PORT);
    info!("🔧 Backend URL: {}", backend_url);
    info!("🎯 Say 'fix the audio' or 'repair audio' to trigger repair");
    info!("📦 Buffering last {}ms of audio", BUFFER_DURATION_MS);
    info!("Press Ctrl+C to stop");

    if let Err(err) = app.start(COPILOT_PORT).await {
        error!("Failed to start copilot app: {:?}", err);
        return Err(err);
    }

    signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");

    info!("\n🛑 Shutting down copilot app...");
    app.stop().await?;
    info!("👋 Copilot app stopped");
    Ok(())
}
