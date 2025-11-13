use std::time::Instant;

// Detection constants
pub const MIN_GAP_CHUNKS: usize = 5;           // 160ms minimum
pub const CONFIDENT_GAP_CHUNKS: usize = 16;    // 512ms+ (very confident)
pub const ENERGY_THRESHOLD_DB: f32 = -40.0;    // dB below recent peak
pub const DISCONTINUITY_THRESHOLD: f32 = 0.15; // Normalized amplitude jump
pub const HISTORY_BUFFER_CHUNKS: usize = 64;   // ~2 seconds

// Pattern detection
pub const PATTERN_WINDOW_MS: f64 = 10000.0;  // Look back 10 seconds instead of 5
pub const MIN_GAPS_FOR_PATTERN: usize = 2;    // Only need 2 gaps instead of 3
pub const MAX_GAP_FOR_PATTERN_MS: f64 = 5000.0; // Allow gaps up to 5 seconds (was 500ms)
pub const MIN_SPEECH_ENERGY: f32 = 0.001;     // Lower threshold (was 0.01)
pub const DEFAULT_ALERT_INTERVAL_MS: f64 = 5000.0;

// Re-export SpeakingParty from the SDK (defined in icf_media_sdk::enums)
// We don't define our own to avoid conflicts
pub use icf_media_sdk::SpeakingParty;

#[derive(Debug, Clone, Copy)]
pub enum StreamState {
    Speaking,
    Silent,
}

#[derive(Debug, Clone)]
pub struct GapEvent {
    pub party: SpeakingParty,
    pub duration_ms: f64,
    pub confidence: f32,
    #[allow(dead_code)]
    pub had_discontinuity: bool,
    pub time: Instant,  // When gap ended
}

pub type RepairCallback = Box<dyn Fn(SpeakingParty, Instant) + Send + Sync>;
