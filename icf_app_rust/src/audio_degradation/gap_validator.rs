use std::collections::VecDeque;
use std::time::{Duration, Instant};
use super::types::*;

#[derive(Debug, Clone)]
struct GapRecord {
    time: Instant,
    duration_ms: f64,
    confidence: f32,
}

pub struct GapValidator {
    callback: RepairCallback,
    recent_gaps_caller: VecDeque<GapRecord>,
    recent_gaps_callee: VecDeque<GapRecord>,
    last_alert_time_caller: Instant,
    last_alert_time_callee: Instant,
    alert_interval: Duration,
    pattern_window: Duration,
}

impl GapValidator {
    pub fn new(callback: RepairCallback, alert_interval_ms: f64) -> Self {
        let now = Instant::now();
        Self {
            callback,
            recent_gaps_caller: VecDeque::with_capacity(20),
            recent_gaps_callee: VecDeque::with_capacity(20),
            last_alert_time_caller: now - Duration::from_secs(3600), // Far past
            last_alert_time_callee: now - Duration::from_secs(3600),
            alert_interval: Duration::from_millis(alert_interval_ms as u64),
            pattern_window: Duration::from_millis(PATTERN_WINDOW_MS as u64),
        }
    }
    
    pub fn validate_gap(&mut self, gap: &GapEvent, avg_speaking_energy: f32) -> bool {
        let gap_record = GapRecord {
            time: gap.time,
            duration_ms: gap.duration_ms,
            confidence: gap.confidence,
        };
        
        let (recent_gaps, last_alert_time) = match gap.party {
            SpeakingParty::Caller => (&mut self.recent_gaps_caller, self.last_alert_time_caller),
            SpeakingParty::Callee => (&mut self.recent_gaps_callee, self.last_alert_time_callee),
        };
        
        // Store gap
        if recent_gaps.len() >= 20 {
            recent_gaps.pop_front();
        }
        recent_gaps.push_back(gap_record);
        
        // Check for degradation pattern (pass pattern_window to avoid borrowing self)
        if has_degradation_pattern(recent_gaps, gap.time, avg_speaking_energy, self.pattern_window) {
            // Check alert cooldown
            if gap.time.duration_since(last_alert_time) >= self.alert_interval {
                log::warn!(
                    "🚨 [{}] DEGRADATION PATTERN DETECTED at t={:.2}s",
                    gap.party,
                    gap.time.elapsed().as_secs_f64()
                );
                self.trigger_alert(gap.party, gap.time);
                return true;
            }
        }
        false
    }
    
    fn trigger_alert(&mut self, party: SpeakingParty, time: Instant) {
        match party {
            SpeakingParty::Caller => self.last_alert_time_caller = time,
            SpeakingParty::Callee => self.last_alert_time_callee = time,
        }
        (self.callback)(party, time);
    }
}

// Make this a free function to avoid borrow checker issues
fn has_degradation_pattern(
    recent_gaps: &VecDeque<GapRecord>,
    current_time: Instant,
    avg_speaking_energy: f32,
    pattern_window: Duration,
) -> bool {
    // Get gaps within time window
    let relevant_gaps: Vec<_> = recent_gaps
        .iter()
        .filter(|g| {
            current_time.duration_since(g.time) < pattern_window
                && g.duration_ms < MAX_GAP_FOR_PATTERN_MS
        })
        .collect();
    
    // Need minimum number of gaps
    if relevant_gaps.len() < MIN_GAPS_FOR_PATTERN {
        return false;
    }
    
    // Check if speaking energy indicates real speech
    if avg_speaking_energy < MIN_SPEECH_ENERGY {
        log::debug!(
            "   Pattern detected but energy too low ({:.6}) - likely background noise",
            avg_speaking_energy
        );
        return false;
    }
    
    log::info!(
        "   Pattern: {} gaps in last {:.1}s, avg energy: {:.6}",
        relevant_gaps.len(),
        PATTERN_WINDOW_MS / 1000.0,
        avg_speaking_energy
    );
    true
}
