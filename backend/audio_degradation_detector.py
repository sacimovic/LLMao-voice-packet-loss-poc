from enum import Enum
from collections import deque
from typing import Callable, Optional, AsyncIterator
from librosa import stream
import numpy as np
from scipy import signal
from dataclasses import dataclass
import asyncio

# Detection thresholds
MIN_GAP_CHUNKS = 5           # # 160ms minimum 
CONFIDENT_GAP_CHUNKS = 16    # 512ms+ (very confident)
ENERGY_THRESHOLD_DB = -40    # dB below recent peak
DISCONTINUITY_THRESHOLD = 0.15  # Normalized amplitude jump

# Buffering
HISTORY_BUFFER_CHUNKS = 64   # ~2 seconds (for peak tracking)

# Cross-stream validation
SUSPICIOUS_GAP_COUNT = 3     # gaps in...
SUSPICIOUS_GAP_WINDOW_S = 5.0  # ...5 seconds

import logging

logger = logging.getLogger(__name__)

class SpeakingParty(Enum):
    CALLER = "caller"
    CALLEE = "callee"

class StreamState(Enum):
    SPEAKING = "speaking"
    SILENT = "silent"
    GAP_SUSPECTED = "gap_suspected"

@dataclass
class GapEvent:
    party: SpeakingParty
    duration_ms: float
    confidence: float
    had_discontinuity: bool
    time: float  # Timestamp when gap ended (in seconds from start)

class StreamAnalyzer:
    """Analyzes a single audio stream for gaps and discontinuities."""
    
    def __init__(self, party: SpeakingParty):
        self.party = party
        self.state = StreamState.SILENT
        self.chunk_buffer = deque(maxlen=HISTORY_BUFFER_CHUNKS)
        self.energy_history = deque(maxlen=HISTORY_BUFFER_CHUNKS)
        self.silent_chunk_count = 0
        self.recent_gap_times = deque(maxlen=10)
        self.peak_energy = 0.0
        self.last_chunk_samples = None
        self.last_update_time = 0.0 
        self.recent_speaking_energies = deque(maxlen=100)
        
    def process_chunk(self, chunk_bytes: bytes) -> Optional[GapEvent]:
        """Process audio chunk and return GapEvent if gap detected."""
        self.last_update_time = asyncio.get_event_loop().time()
        samples = self._bytes_to_samples(chunk_bytes)
        energy = self._calculate_rms(samples)
        
        # Update buffers
        self.chunk_buffer.append(samples)
        self.energy_history.append(energy)
        
        # Track peak for relative threshold with minimum floor
        MIN_ENERGY_FLOOR = 0.001  # Prevent threshold from being too low
        self.peak_energy = max(self.peak_energy * 0.999, energy, MIN_ENERGY_FLOOR)
        
        # Check for discontinuity (click/glitch)
        discontinuity = self._check_discontinuity(samples)
        
        # Check if chunk is silent
        threshold = self.peak_energy * (10 ** (ENERGY_THRESHOLD_DB / 20))
        is_silent = energy < threshold
        if is_silent:
            self.silent_chunk_count += 1
            self.state = StreamState.SILENT
        else:
            # Track energy when speaking (not silent)
            self.recent_speaking_energies.append(energy)
        
        # Log every 10th chunk for debugging
        if hasattr(self, '_chunk_counter'):
            self._chunk_counter += 1
        else:
            self._chunk_counter = 0
        
        if self._chunk_counter % 10 == 0:
            logger.debug(f"[{self.party.value}] Energy: {energy:.6f}, Peak: {self.peak_energy:.6f}, "
                        f"Threshold: {threshold:.6f}, Silent: {is_silent}, Silent count: {self.silent_chunk_count}")
        
        if is_silent:
            self.silent_chunk_count += 1
            self.state = StreamState.SILENT  # Update state immediately
        else:
            # Non-silent chunk detected
            gap_event = None
            if self.silent_chunk_count >= MIN_GAP_CHUNKS:
                # We had a gap, now it's over
                gap_duration_ms = self.silent_chunk_count * 32
                logger.warning(f"[{self.party.value}] GAP DETECTED: {gap_duration_ms}ms "
                            f"(confidence: {self._calculate_confidence():.2f})")
                gap_event = GapEvent(
                    party=self.party,
                    duration_ms=gap_duration_ms,
                    confidence=self._calculate_confidence(),
                    had_discontinuity=discontinuity,
                    time=self.last_update_time
                )
            
            # Reset and update state
            self.silent_chunk_count = 0
            self.state = StreamState.SPEAKING
            
            # Return gap event if one was detected
            if gap_event:
                return gap_event
        
        self.last_chunk_samples = samples
        return None
    
    def get_recent_speaking_energy(self) -> float:
        """Get average energy from recent speaking chunks."""
        if len(self.recent_speaking_energies) == 0:
            return 0.0
        # Average of last 20 speaking chunks
        recent = list(self.recent_speaking_energies)[-20:]
        return float(np.mean(recent))
    
    def _bytes_to_samples(self, chunk_bytes: bytes) -> np.ndarray:
        """Convert bytes to normalized float samples."""
        return np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    def _calculate_rms(self, samples: np.ndarray) -> float:
        """Calculate RMS energy."""
        return np.sqrt(np.mean(samples ** 2))
    
    def _check_discontinuity(self, samples: np.ndarray) -> bool:
        """Check for abrupt amplitude jump between chunks."""
        if self.last_chunk_samples is None:
            return False
        delta = abs(self.last_chunk_samples[-1] - samples[0])
        return delta > DISCONTINUITY_THRESHOLD
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence score for detected gap."""
        if self.silent_chunk_count >= CONFIDENT_GAP_CHUNKS:
            return 1.0
        # Linear scale: 3 chunks = 0.75, 16+ chunks = 1.0
        return 0.75 + (0.25 * (self.silent_chunk_count - MIN_GAP_CHUNKS) / 
                       (CONFIDENT_GAP_CHUNKS - MIN_GAP_CHUNKS))

class GapValidator:
    """Pattern-based gap validator with energy filtering."""
    
    def __init__(self, callback, alertIntvMsec=5000.0):
        self.callback = callback
        self.recent_gaps_caller = deque(maxlen=20)
        self.recent_gaps_callee = deque(maxlen=20)
        self.recent_energies_caller = deque(maxlen=100)  # Track recent speaking energy
        self.recent_energies_callee = deque(maxlen=100)
        self.alert_interval_s = alertIntvMsec / 1000.0
        self.last_alert_time_caller = 0.0
        self.last_alert_time_callee = 0.0
        
        # Pattern detection parameters
        self.pattern_window_s = 5.0
        self.min_gaps_for_pattern = 3
        self.max_gap_for_pattern_ms = 500
        
        # Energy threshold for "real speech" vs background noise
        self.min_speech_energy = 0.01  # Require 0.01 RMS for speech detection
        
    def update_energy(self, party, energy, is_speaking):
        """Track energy levels for speech detection."""
        if is_speaking:  # Only track energy when not silent
            if party == SpeakingParty.CALLER:
                self.recent_energies_caller.append(energy)
            else:
                self.recent_energies_callee.append(energy)
    
    def validate_gap(self, gap : GapEvent, avg_speaking_energy: float):
        """Validate if gap should trigger alert based on pattern detection."""
        
        # Store gap with timestamp
        gap_record = {
            'time': gap.time,
            'duration_ms': gap.duration_ms,
            'confidence': gap.confidence
        }
        
        if gap.party == SpeakingParty.CALLER:
            self.recent_gaps_caller.append(gap_record)
            recent_gaps = self.recent_gaps_caller
            recent_energies = self.recent_energies_caller
            last_alert_time = self.last_alert_time_caller
        else:
            self.recent_gaps_callee.append(gap_record)
            recent_gaps = self.recent_gaps_callee
            recent_energies = self.recent_energies_callee
            last_alert_time = self.last_alert_time_callee
        
        # Check for degradation pattern
        if self._has_degradation_pattern(recent_gaps, recent_energies, gap.time):
            # Only alert if enough time has passed since last alert
            if (gap.time - last_alert_time) >= self.alert_interval_s:
                print(f"🚨 [{gap.party.value}] DEGRADATION PATTERN DETECTED at t={gap.time:.2f}s")
                self._trigger_alert(gap)
                return True
        return False
    
    def _has_degradation_pattern(self, recent_gaps, recent_energies, current_time):
        """Check if recent gaps indicate a degradation pattern (choppy audio)."""
        
        # Get gaps within the time window
        relevant_gaps = [g for g in recent_gaps 
                        if (current_time - g['time']) < self.pattern_window_s
                        and g['duration_ms'] < self.max_gap_for_pattern_ms]
        
        # Need minimum number of short gaps in the window
        if len(relevant_gaps) < self.min_gaps_for_pattern:
            return False
        
        # Check if recent speaking chunks have sufficient energy for "real speech"
        recent_avg_energy = 0.0
        if len(recent_energies) > 0:
            recent_avg_energy = np.mean(list(recent_energies)[-20:])  # Last 20 speaking chunks
            
            if recent_avg_energy < self.min_speech_energy:
                print(f"   Pattern detected but energy too low ({recent_avg_energy:.6f}) - likely background noise")
                return False
        
        print(f"   Pattern: {len(relevant_gaps)} gaps in last {self.pattern_window_s}s, "
              f"avg energy: {recent_avg_energy:.6f}")
        return True
    
    def _trigger_alert(self, gap):
        """Trigger repair callback."""
        if gap.party == SpeakingParty.CALLER:
            self.last_alert_time_caller = gap.time
        else:
            self.last_alert_time_callee = gap.time
            
        self.callback(gap.party, gap.time)

class AudioDegradationDetector:
    """Main detector coordinating both streams."""
    
    def __init__(self, on_repair_requested: Callable[[SpeakingParty], None], alertIntvMsec: float = 5000.0):
        self.caller_analyzer = StreamAnalyzer(SpeakingParty.CALLER)
        self.callee_analyzer = StreamAnalyzer(SpeakingParty.CALLEE)
        self.validator = GapValidator(on_repair_requested, alertIntvMsec)
    
    async def start(self, caller_stream: AsyncIterator[bytes], callee_stream: AsyncIterator[bytes]):
        """Start monitoring both audio streams."""
        await asyncio.gather(
            self._monitor_stream(caller_stream, self.caller_analyzer, self.callee_analyzer),
            self._monitor_stream(callee_stream, self.callee_analyzer, self.caller_analyzer)
        )
    
    async def _monitor_stream(
        self, 
        stream: AsyncIterator[bytes], 
        analyzer: StreamAnalyzer,
        other_analyzer: StreamAnalyzer
    ):
        """Monitor a single stream for degradation."""
        async for chunk in stream:
            gap_event = analyzer.process_chunk(chunk)
            if gap_event:
                avg_speaking_energy = analyzer.get_recent_speaking_energy()
                self.validator.validate_gap(gap_event, avg_speaking_energy)
