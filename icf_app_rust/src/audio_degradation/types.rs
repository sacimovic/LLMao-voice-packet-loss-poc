use std::time::Instant;

// Detection constants
#[allow(dead_code)]
pub const MIN_GAP_CHUNKS: usize = 5;           // 160ms minimum
#[allow(dead_code)]
pub const CONFIDENT_GAP_CHUNKS: usize = 16;    // 512ms+ (very confident)
#[allow(dead_code)]
pub const ENERGY_THRESHOLD_DB: f32 = -40.0;    // dB below recent peak
#[allow(dead_code)]
pub const DISCONTINUITY_THRESHOLD: f32 = 0.15; // Normalized amplitude jump
#[allow(dead_code)]
pub const HISTORY_BUFFER_CHUNKS: usize = 64;   // ~2 seconds

// Pattern detection
#[allow(dead_code)]
pub const PATTERN_WINDOW_MS: f64 = 10000.0;  // Look back 10 seconds instead of 5
#[allow(dead_code)]
pub const MIN_GAPS_FOR_PATTERN: usize = 2;    // Only need 2 gaps instead of 3
#[allow(dead_code)]
pub const MAX_GAP_FOR_PATTERN_MS: f64 = 5000.0; // Allow gaps up to 5 seconds (was 500ms)
#[allow(dead_code)]
pub const MIN_SPEECH_ENERGY: f32 = 0.001;     // Lower threshold (was 0.01)
#[allow(dead_code)]
pub const DEFAULT_ALERT_INTERVAL_MS: f64 = 5000.0;

// Re-export SpeakingParty from the SDK (defined in icf_media_sdk::enums)
// We don't define our own to avoid conflicts
pub use icf_media_sdk::SpeakingParty;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum StreamState {
    Speaking,
    Silent,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GapEvent {
    pub party: SpeakingParty,
    pub duration_ms: f64,
    pub confidence: f32,
    #[allow(dead_code)]
    pub had_discontinuity: bool,
    pub time: Instant,  // When gap ended
}

#[allow(dead_code)]
pub type RepairCallback = Box<dyn Fn(SpeakingParty, Instant) + Send + Sync>;
