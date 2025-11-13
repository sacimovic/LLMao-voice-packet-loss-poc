use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tokio::sync::mpsc;
use log::{info, warn, error, debug};

use crate::audio_buffer::AudioBuffer;
use crate::audio_repair::AudioRepairer;

// Configuration
#[allow(dead_code)]
const TARGET_REPAIR_DURATION_S: f64 = 10.0;  // Always use 10 second buffers
#[allow(dead_code)]
const MIN_SPEECH_ENERGY: f32 = 0.01;
#[allow(dead_code)]
const SILENCE_TIMEOUT_S: f64 = 2.0;  // How long to wait after silence before finalizing
#[allow(dead_code)]
const SEGMENT_GAP_THRESHOLD_S: f64 = 2.0;
#[allow(dead_code)]
const REPAIRED_SEGMENT_TTL_S: f64 = 30.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamType {
    Caller,
    Callee,
}

impl std::fmt::Display for StreamType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamType::Caller => write!(f, "caller"),
            StreamType::Callee => write!(f, "callee"),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct RepairedSegment {
    pub audio_pcm: Vec<i16>,
    pub start_time: Instant,
    pub end_time: Instant,
    pub stream: StreamType,
}

#[allow(dead_code)]
impl RepairedSegment {
    pub fn duration_s(&self) -> f64 {
        self.end_time.duration_since(self.start_time).as_secs_f64()
    }
    
    pub fn age_s(&self, now: Instant) -> f64 {
        now.duration_since(self.end_time).as_secs_f64()
    }
}

#[allow(dead_code)]
pub struct BackgroundRepairState {
    is_active_caller: AtomicBool,
    is_active_callee: AtomicBool,
    segments: Arc<Mutex<Vec<RepairedSegment>>>,
    caller_task: Mutex<Option<JoinHandle<()>>>,
    callee_task: Mutex<Option<JoinHandle<()>>>,
    caller_audio_tx: Mutex<Option<mpsc::UnboundedSender<Vec<i16>>>>,
    callee_audio_tx: Mutex<Option<mpsc::UnboundedSender<Vec<i16>>>>,
}

#[allow(dead_code)]
impl BackgroundRepairState {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            is_active_caller: AtomicBool::new(false),
            is_active_callee: AtomicBool::new(false),
            segments: Arc::new(Mutex::new(Vec::new())),
            caller_task: Mutex::new(None),
            callee_task: Mutex::new(None),
            caller_audio_tx: Mutex::new(None),
            callee_audio_tx: Mutex::new(None),
        })
    }
    
    pub fn is_active(&self, stream: StreamType) -> bool {
        match stream {
            StreamType::Caller => self.is_active_caller.load(Ordering::Relaxed),
            StreamType::Callee => self.is_active_callee.load(Ordering::Relaxed),
        }
    }
    
    fn set_active(&self, stream: StreamType, active: bool) {
        match stream {
            StreamType::Caller => self.is_active_caller.store(active, Ordering::Relaxed),
            StreamType::Callee => self.is_active_callee.store(active, Ordering::Relaxed),
        }
    }
    
    pub fn send_audio_chunk(&self, stream: StreamType, samples: Vec<i16>) {
        let tx = match stream {
            StreamType::Caller => self.caller_audio_tx.lock().unwrap(),
            StreamType::Callee => self.callee_audio_tx.lock().unwrap(),
        };
        
        if let Some(tx) = tx.as_ref() {
            let _ = tx.send(samples); // Ignore errors (channel might be closed)
        }
    }
    
    pub fn add_segment(&self, segment: RepairedSegment) {
        let mut segments = self.segments.lock().unwrap();
        segments.push(segment);
    }
    
    pub fn get_recent_segment(&self, stream: StreamType) -> Option<RepairedSegment> {
        let segments = self.segments.lock().unwrap();
        
        // Filter for this stream
        let mut stream_segments: Vec<_> = segments.iter()
            .filter(|s| s.stream == stream)
            .collect();
        
        if stream_segments.is_empty() {
            return None;
        }
        
        // Sort by end time (most recent last)
        stream_segments.sort_by_key(|s| s.end_time);
        
        // Start with most recent
        let recent = stream_segments.last().unwrap();
        
        // Find continuous group (gap < threshold)
        let mut continuous_group = vec![(*recent).clone()];
        
        for i in (0..stream_segments.len() - 1).rev() {
            let current = stream_segments[i];
            let next = &continuous_group[0];
            
            let gap = next.start_time.duration_since(current.end_time).as_secs_f64();
            
            if gap < SEGMENT_GAP_THRESHOLD_S {
                continuous_group.insert(0, current.clone());
            } else {
                break;
            }
        }
        
        // Merge if multiple segments
        if continuous_group.len() == 1 {
            Some(continuous_group[0].clone())
        } else {
            // Merge PCM data
            let merged_pcm: Vec<i16> = continuous_group
                .iter()
                .flat_map(|s| s.audio_pcm.iter().copied())
                .collect();
            
            Some(RepairedSegment {
                audio_pcm: merged_pcm,
                start_time: continuous_group[0].start_time,
                end_time: continuous_group.last().unwrap().end_time,
                stream,
            })
        }
    }
    
    pub fn cleanup_old_segments(&self) {
        let mut segments = self.segments.lock().unwrap();
        let now = Instant::now();
        let initial_count = segments.len();
        
        segments.retain(|s| s.age_s(now) < REPAIRED_SEGMENT_TTL_S);
        
        let removed = initial_count - segments.len();
        if removed > 0 {
            info!("🗑️ Cleaned up {} expired segment(s)", removed);
        }
    }
    
    pub fn remove_segment(&self, segment: &RepairedSegment) {
        let mut segments = self.segments.lock().unwrap();
        segments.retain(|s| {
            !(s.stream == segment.stream 
              && s.start_time == segment.start_time 
              && s.end_time == segment.end_time)
        });
    }
    
    pub fn start_repair_if_needed(
        self: &Arc<Self>,
        stream: StreamType,
        _buffer: Arc<Mutex<AudioBuffer>>,
        repairer: Arc<AudioRepairer>,
    ) {
        if self.is_active(stream) {
            // Already running
            return;
        }
        
        self.set_active(stream, true);
        
        // Create channel for sending audio chunks to the task
        let (tx, rx) = mpsc::unbounded_channel();
        
        // Store the sender
        match stream {
            StreamType::Caller => {
                *self.caller_audio_tx.lock().unwrap() = Some(tx);
            }
            StreamType::Callee => {
                *self.callee_audio_tx.lock().unwrap() = Some(tx);
            }
        }
        
        let task = tokio::spawn(background_repair_task(
            stream,
            rx,
            repairer,
            Arc::clone(self),
        ));
        
        match stream {
            StreamType::Caller => {
                *self.caller_task.lock().unwrap() = Some(task);
            }
            StreamType::Callee => {
                *self.callee_task.lock().unwrap() = Some(task);
            }
        }
    }
    
    pub async fn stop_and_wait(
        &self,
        stream: StreamType,
        timeout_secs: u64,
    ) -> Option<RepairedSegment> {
        if !self.is_active(stream) {
            return None;
        }
        
        info!("⏸️ Stopping background repair for {}", stream);
        
        // Signal to stop
        self.set_active(stream, false);
        
        // Get the task handle
        let task = match stream {
            StreamType::Caller => self.caller_task.lock().unwrap().take(),
            StreamType::Callee => self.callee_task.lock().unwrap().take(),
        };
        
        if let Some(task) = task {
            info!("⏳ Waiting for current repair operation to complete...");
            
            // Wait with timeout
            match tokio::time::timeout(Duration::from_secs(timeout_secs), task).await {
                Ok(_) => {
                    info!("✅ Repair task completed");
                }
                Err(_) => {
                    warn!("⏱️ Repair task timed out");
                }
            }
        }
        
        // Return most recent segment
        self.get_recent_segment(stream)
    }
}

#[allow(dead_code)]
async fn background_repair_task(
    stream: StreamType,
    mut audio_rx: mpsc::UnboundedReceiver<Vec<i16>>,
    repairer: Arc<AudioRepairer>,
    state: Arc<BackgroundRepairState>,
) {
    info!("🔧 [{}] Starting background repair task", stream);
    
    let start_time = Instant::now();
    let mut accumulated_audio: Vec<i16> = Vec::new();
    let mut last_speech_time: Option<Instant> = None;
    
    loop {
        // Check if we've accumulated 10 seconds of audio
        let current_duration_s = accumulated_audio.len() as f64 / 16000.0;
        if current_duration_s >= TARGET_REPAIR_DURATION_S {
            info!("✅ [{}] Reached target duration of {:.1}s", stream, TARGET_REPAIR_DURATION_S);
            break;
        }
        
        // Wait for audio chunks or timeout
        match tokio::time::timeout(Duration::from_millis(100), audio_rx.recv()).await {
            Ok(Some(chunk)) => {
                // Got audio chunk
                // Convert to f32 to check energy
                let chunk_f32: Vec<f32> = chunk.iter()
                    .map(|&s| s as f32 / 32768.0)
                    .collect();
                
                let rms = calculate_rms(&chunk_f32);
                
                if rms > MIN_SPEECH_ENERGY {
                    // Speech detected
                    accumulated_audio.extend_from_slice(&chunk);
                    last_speech_time = Some(Instant::now());
                    debug!("[{}] Accumulated chunk: {} samples, RMS: {:.4}", 
                           stream, chunk.len(), rms);
                } else {
                    debug!("[{}] Skipping quiet chunk (RMS: {:.6})", stream, rms);
                }
            }
            Ok(None) => {
                // Channel closed
                info!("🛑 [{}] Audio channel closed", stream);
                break;
            }
            Err(_) => {
                // Timeout - check if we should finalize
                if !state.is_active(stream) {
                    info!("🛑 [{}] Received stop signal", stream);
                    break;
                }
                
                // Check if speech has ended (silence for > SILENCE_TIMEOUT_S)
                if let Some(last_speech) = last_speech_time {
                    let silence_duration = last_speech.elapsed().as_secs_f64();
                    if silence_duration > SILENCE_TIMEOUT_S {
                        info!("✅ [{}] Speech ended ({:.1}s of silence detected)", stream, silence_duration);
                        break;
                    }
                }
            }
        }
    }
    
    // Process the accumulated audio
    let duration_s = accumulated_audio.len() as f64 / 16000.0;
    
    if accumulated_audio.is_empty() {
        info!("ℹ️ [{}] No audio accumulated", stream);
        state.set_active(stream, false);
        info!("🏁 [{}] Background repair task finished", stream);
        return;
    }
    
    info!("🔄 [{}] Sending {:.1}s of accumulated audio for repair", stream, duration_s);
    
    match repairer.repair_audio(&accumulated_audio).await {
        Ok(repaired_i16) => {
            let repaired_duration = repaired_i16.len() as f64 / 16000.0;
            info!("✅ [{}] Repair complete, received {:.1}s of repaired audio", 
                  stream, repaired_duration);
            
            let segment = RepairedSegment {
                audio_pcm: repaired_i16,
                start_time,
                end_time: Instant::now(),
                stream,
            };
            
            state.add_segment(segment);
            info!("💾 [{}] Stored {:.1}s segment", stream, repaired_duration);
        }
        Err(e) => {
            error!("❌ [{}] Repair failed: {}", stream, e);
        }
    }
    
    state.set_active(stream, false);
    info!("🏁 [{}] Background repair task finished", stream);
}

#[allow(dead_code)]
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    
    let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}
