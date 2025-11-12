use std::sync::{Arc, Mutex};
use std::path::PathBuf;

use async_trait::async_trait;
use icf_media_sdk::{
    create_copilot_app, AppOptions, CallCleanup, CopilotCallHandle, CopilotMode,
    MediaHandler, SdkResult, SpeakingParty,
};
use log::{error, info, warn};
use regex::Regex;
use tokio::{signal, fs};

mod audio_buffer;
mod audio_repair;

use audio_buffer::AudioBuffer;
use audio_repair::AudioRepairer;

const COPILOT_PORT: u16 = 8084;
const BUFFER_DURATION_MS: u64 = 5000; // Keep last 5 seconds of audio

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

    async fn on_health_check(
        &self,
        request: icf_media_sdk::HealthCheckRequest,
    ) -> SdkResult<Option<icf_media_sdk::HealthCheckResponse>> {
        info!("💓 Health check received from region: {}", request.source_region);
        
        // Return a response with current timestamp to indicate we're healthy
        Ok(Some(icf_media_sdk::HealthCheckResponse {
            response_timestamp: chrono::Utc::now().to_rfc3339(),
        }))
    }

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
        
        let backend_url = self.backend_url.clone();
        let is_processing = Arc::new(Mutex::new(false));
        let repair_count = Arc::new(Mutex::new(0));

        // Compile trigger patterns
        let trigger_regexes: Vec<Regex> = TRIGGER_PATTERNS
            .iter()
            .filter_map(|pattern| Regex::new(pattern).ok())
            .collect();

        // Handle audio data
        call.on_audio(Some({
            let caller_buf = Arc::clone(&caller_buffer);
            let callee_buf = Arc::clone(&callee_buffer);
            
            Arc::new(move |audio_bytes, speaking_party| {
                // Convert bytes to i16 samples (little-endian PCM)
                let samples: Vec<i16> = audio_bytes
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                
                // Add audio to appropriate buffer based on party
                match speaking_party {
                    icf_media_sdk::SpeakingParty::Caller => {
                        if let Ok(mut buf) = caller_buf.lock() {
                            buf.add_samples(&samples);
                        }
                    }
                    icf_media_sdk::SpeakingParty::Callee => {
                        if let Ok(mut buf) = callee_buf.lock() {
                            buf.add_samples(&samples);
                        }
                    }
                }
            })
        }));

        // Handle partial utterances to detect trigger phrase
        call.on_partial_utterance(Some({
            let call_ref = Arc::clone(&call);
            let backend_url_clone = backend_url.clone();
            let processing_flag = Arc::clone(&is_processing);
            let count = Arc::clone(&repair_count);
            let caller_buf = Arc::clone(&caller_buffer);
            let callee_buf = Arc::clone(&callee_buffer);
            
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

                    // Spawn repair task
                    let call_clone = Arc::clone(&call_ref);
                    let backend = backend_url_clone.clone();
                    let processing = Arc::clone(&processing_flag);
                    let repair_count = Arc::clone(&count);
                    let buffer_to_repair = match speaking_party {
                        SpeakingParty::Caller => Arc::clone(&callee_buf),
                        SpeakingParty::Callee => Arc::clone(&caller_buf),
                    };
                    let target_party = match speaking_party {
                        SpeakingParty::Caller => "caller",
                        SpeakingParty::Callee => "callee",
                    }.to_string();

                    tokio::spawn(async move {
                        match process_and_repair_audio(
                            call_clone,
                            backend,
                            buffer_to_repair,
                            target_party,
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

async fn process_and_repair_audio(
    call: Arc<dyn icf_media_sdk::CopilotCall>,
    backend_url: String,
    audio_buffer: Arc<Mutex<AudioBuffer>>,
    target_party: String,
) -> anyhow::Result<()> {
    info!("🔧 Starting audio repair process...");

    // Switch to bidirectional mode if needed
    if call.call_mode() == CopilotMode::ListenOnly {
        info!("🔄 Switching to bidirectional mode...");
        call.set_bidirectional().await?;
        info!("✅ Now bidirectional");
    }

    // Get audio samples from buffer
    let samples = {
        let buf = audio_buffer.lock().unwrap();
        buf.get_samples()
    };

    if samples.is_empty() {
        warn!("⚠️ No audio in buffer to repair");
        return Ok(());
    }

    info!("📊 Processing {} samples", samples.len());

    // Create audio repairer and process
    let repairer = AudioRepairer::new(backend_url)?;
    let repaired_audio = repairer.repair_audio(&samples).await?;

    info!("🎵 Repaired audio ready, sending to {}", target_party);

    // Determine target and send audio
    let audio_bytes = samples_to_bytes(&repaired_audio);
    
    if target_party.as_str() == "caller" {
        call.send_audio_to_caller(&audio_bytes, None)?;
    } else {
        call.send_audio_to_callee(&audio_bytes, None)?;
    }
    
    info!("🔊 Repaired audio sent");

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
