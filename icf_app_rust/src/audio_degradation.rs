mod gap_validator;
mod stream_analyzer;
mod types;

pub use types::RepairCallback;
// SpeakingParty is re-exported from types, which gets it from icf_media_sdk
pub use types::SpeakingParty;

use std::sync::Mutex;
use gap_validator::GapValidator;
use stream_analyzer::StreamAnalyzer;
use types::*;

#[allow(dead_code)]
pub struct AudioDegradationDetector {
    caller_analyzer: Mutex<StreamAnalyzer>,
    callee_analyzer: Mutex<StreamAnalyzer>,
    validator: Mutex<GapValidator>,
}

#[allow(dead_code)]
impl AudioDegradationDetector {
    pub fn new(
        on_repair_requested: RepairCallback,
        alert_interval_ms: Option<f64>,
    ) -> Self {
        let interval = alert_interval_ms.unwrap_or(DEFAULT_ALERT_INTERVAL_MS);
        
        Self {
            caller_analyzer: Mutex::new(StreamAnalyzer::new(SpeakingParty::Caller)),
            callee_analyzer: Mutex::new(StreamAnalyzer::new(SpeakingParty::Callee)),
            validator: Mutex::new(GapValidator::new(on_repair_requested, interval)),
        }
    }
    
    /// Process audio chunk from a specific party
    pub fn process_chunk(&self, chunk_bytes: &[u8], party: SpeakingParty) {
        let analyzer = match party {
            SpeakingParty::Caller => &self.caller_analyzer,
            SpeakingParty::Callee => &self.callee_analyzer,
        };
        
        let gap_event = {
            let mut analyzer = analyzer.lock().unwrap();
            analyzer.process_chunk(chunk_bytes)
        };
        
        if let Some(gap) = gap_event {
            let avg_energy = {
                let analyzer = analyzer.lock().unwrap();
                analyzer.get_recent_speaking_energy()
            };
            
            let mut validator = self.validator.lock().unwrap();
            validator.validate_gap(&gap, avg_energy);
        }
    }
}
