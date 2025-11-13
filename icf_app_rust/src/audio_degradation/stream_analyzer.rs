use std::collections::VecDeque;
use std::time::Instant;
use super::types::*;

pub struct StreamAnalyzer {
    party: SpeakingParty,
    state: StreamState,
    
    // Buffers
    chunk_buffer: VecDeque<Vec<f32>>,
    energy_history: VecDeque<f32>,
    recent_speaking_energies: VecDeque<f32>,
    
    // Tracking
    silent_chunk_count: usize,
    peak_energy: f32,
    last_chunk_samples: Option<Vec<f32>>,
    chunk_counter: usize,
}

impl StreamAnalyzer {
    pub fn new(party: SpeakingParty) -> Self {
        Self {
            party,
            state: StreamState::Silent,
            chunk_buffer: VecDeque::with_capacity(HISTORY_BUFFER_CHUNKS),
            energy_history: VecDeque::with_capacity(HISTORY_BUFFER_CHUNKS),
            recent_speaking_energies: VecDeque::with_capacity(100),
            silent_chunk_count: 0,
            peak_energy: 0.001, // Minimum floor
            last_chunk_samples: None,
            chunk_counter: 0,
        }
    }
    
    /// Process a single audio chunk (32ms of 16kHz mono PCM)
    pub fn process_chunk(&mut self, chunk_bytes: &[u8]) -> Option<GapEvent> {
        let samples = bytes_to_samples(chunk_bytes);
        let energy = calculate_rms(&samples);
        
        // Update buffers
        if self.chunk_buffer.len() >= HISTORY_BUFFER_CHUNKS {
            self.chunk_buffer.pop_front();
        }
        self.chunk_buffer.push_back(samples.clone());
        
        if self.energy_history.len() >= HISTORY_BUFFER_CHUNKS {
            self.energy_history.pop_front();
        }
        self.energy_history.push_back(energy);
        
        // Track peak with decay
        self.peak_energy = (self.peak_energy * 0.999).max(energy).max(0.001);
        
        // Check for discontinuity
        let discontinuity = self.check_discontinuity(&samples);
        
        // Determine if chunk is silent
        let threshold = self.peak_energy * 10f32.powf(ENERGY_THRESHOLD_DB / 20.0);
        let is_silent = energy < threshold;
        
        // Debug logging every 10 chunks
        self.chunk_counter += 1;
        if self.chunk_counter % 10 == 0 {
            log::debug!(
                "[{}] Energy: {:.6}, Peak: {:.6}, Threshold: {:.6}, Silent: {}, Count: {}",
                self.party, energy, self.peak_energy, threshold, is_silent, self.silent_chunk_count
            );
        }
        
        let gap_event = if is_silent {
            self.silent_chunk_count += 1;
            self.state = StreamState::Silent;
            None
        } else {
            // Non-silent chunk - check if we just ended a gap
            let event = if self.silent_chunk_count >= MIN_GAP_CHUNKS {
                let gap_duration_ms = (self.silent_chunk_count * 32) as f64;
                log::warn!(
                    "[{}] GAP DETECTED: {:.0}ms (confidence: {:.2})",
                    self.party, gap_duration_ms, self.calculate_confidence()
                );
                Some(GapEvent {
                    party: self.party,
                    duration_ms: gap_duration_ms,
                    confidence: self.calculate_confidence(),
                    had_discontinuity: discontinuity,
                    time: Instant::now(),
                })
            } else {
                None
            };
            
            // Track speaking energy
            if self.recent_speaking_energies.len() >= 100 {
                self.recent_speaking_energies.pop_front();
            }
            self.recent_speaking_energies.push_back(energy);
            
            // Reset gap counter
            self.silent_chunk_count = 0;
            self.state = StreamState::Speaking;
            
            event
        };
        
        self.last_chunk_samples = Some(samples);
        gap_event
    }
    
    pub fn get_recent_speaking_energy(&self) -> f32 {
        if self.recent_speaking_energies.is_empty() {
            return 0.0;
        }
        // Average of last 20 speaking chunks
        let count = self.recent_speaking_energies.len().min(20);
        let sum: f32 = self.recent_speaking_energies
            .iter()
            .rev()
            .take(count)
            .sum();
        sum / count as f32
    }
    
    fn check_discontinuity(&self, samples: &[f32]) -> bool {
        if let Some(ref last_samples) = self.last_chunk_samples {
            if let (Some(&last), Some(&first)) = (last_samples.last(), samples.first()) {
                return (last - first).abs() > DISCONTINUITY_THRESHOLD;
            }
        }
        false
    }
    
    fn calculate_confidence(&self) -> f32 {
        if self.silent_chunk_count >= CONFIDENT_GAP_CHUNKS {
            return 1.0;
        }
        // Linear scale: MIN_GAP_CHUNKS = 0.75, CONFIDENT_GAP_CHUNKS+ = 1.0
        0.75 + (0.25 * (self.silent_chunk_count - MIN_GAP_CHUNKS) as f32 / 
                (CONFIDENT_GAP_CHUNKS - MIN_GAP_CHUNKS) as f32)
    }
}

/// Convert little-endian PCM bytes to normalized f32 samples
fn bytes_to_samples(chunk_bytes: &[u8]) -> Vec<f32> {
    chunk_bytes
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect()
}

/// Calculate RMS energy
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}
