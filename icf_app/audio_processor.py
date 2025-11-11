"""
Audio processing module adapted from LLMao backend for real-time streaming.
Handles packet loss detection and audio repair using Whisper + FLAN-T5 + XTTS.
"""
import io
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# Import audio utilities from backend
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from audio_utils import (
    transcribe_whisper,
    repair_text_with_local_model,
    synthesize_xtts,
    write_wav_bytes,
)


class AudioBuffer:
    """Manages buffering of audio chunks for processing."""
    
    def __init__(self, sample_rate: int = 16000, max_duration_ms: int = 5000):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration_ms / 1000)
        self.buffer = []
        self.total_samples = 0
    
    def add_chunk(self, audio_data: bytes):
        """Add PCM audio chunk to buffer."""
        # Convert bytes to numpy array (16-bit PCM)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer.append(audio_array)
        self.total_samples += len(audio_array)
        
        # Trim if exceeds max duration
        while self.total_samples > self.max_samples and len(self.buffer) > 0:
            removed = self.buffer.pop(0)
            self.total_samples -= len(removed)
    
    def get_audio(self) -> np.ndarray:
        """Get concatenated audio as numpy array."""
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.total_samples = 0
    
    def duration_ms(self) -> float:
        """Get current buffer duration in milliseconds."""
        return (self.total_samples / self.sample_rate) * 1000


class PacketLossDetector:
    """Detects potential packet loss or audio degradation in real-time."""
    
    def __init__(self, silence_threshold: float = 0.01, min_silence_ms: int = 100):
        self.silence_threshold = silence_threshold
        self.min_silence_ms = min_silence_ms
    
    def detect_silence_gaps(self, audio: np.ndarray, sample_rate: int) -> list:
        """
        Detect significant silence gaps that might indicate packet loss.
        Returns list of (start_ms, duration_ms) tuples.
        """
        if len(audio) == 0:
            return []
        
        min_silence_samples = int(sample_rate * self.min_silence_ms / 1000)
        gaps = []
        
        # Find silence regions
        is_silent = np.abs(audio) < self.silence_threshold
        
        # Find contiguous silent regions
        silent_start = None
        for i, silent in enumerate(is_silent):
            if silent and silent_start is None:
                silent_start = i
            elif not silent and silent_start is not None:
                duration_samples = i - silent_start
                if duration_samples >= min_silence_samples:
                    start_ms = (silent_start / sample_rate) * 1000
                    duration_ms = (duration_samples / sample_rate) * 1000
                    gaps.append((start_ms, duration_ms))
                silent_start = None
        
        # Check final region
        if silent_start is not None:
            duration_samples = len(audio) - silent_start
            if duration_samples >= min_silence_samples:
                start_ms = (silent_start / sample_rate) * 1000
                duration_ms = (duration_samples / sample_rate) * 1000
                gaps.append((start_ms, duration_ms))
        
        return gaps
    
    def has_degradation(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Quick check if audio appears degraded."""
        gaps = self.detect_silence_gaps(audio, sample_rate)
        return len(gaps) > 0


class AudioRepairer:
    """Handles the full audio repair pipeline."""
    
    def __init__(
        self,
        whisper_model: str = "tiny",
        repair_model: str = "google/flan-t5-small",
        sample_rate: int = 16000
    ):
        self.whisper_model = whisper_model
        self.repair_model = repair_model
        self.sample_rate = sample_rate
    
    def repair_audio(
        self,
        degraded_audio: np.ndarray,
        speaker_reference: bytes = None
    ) -> tuple[str, str, bytes]:
        """
        Repair degraded audio through ASR -> text repair -> TTS pipeline.
        
        Returns:
            (original_text, repaired_text, repaired_audio_bytes)
        """
        # Step 1: Transcribe degraded audio
        asr_text = transcribe_whisper(
            degraded_audio,
            self.sample_rate,
            model_size=self.whisper_model
        )
        
        # Step 2: Repair the transcribed text
        repaired_text = repair_text_with_local_model(
            asr_text,
            model_name=self.repair_model
        )
        
        # Step 3: Synthesize repaired audio
        if speaker_reference:
            repaired_audio_bytes = synthesize_xtts(
                repaired_text,
                speaker_wav_bytes=speaker_reference,
                language="en"
            )
        else:
            # If no reference, create simple PCM output
            # For now, return empty audio - TTS needs speaker reference
            repaired_audio_bytes = write_wav_bytes(
                np.zeros(len(degraded_audio), dtype=np.float32),
                self.sample_rate
            )
        
        return asr_text, repaired_text, repaired_audio_bytes
    
    def convert_wav_to_pcm(self, wav_bytes: bytes) -> bytes:
        """Convert WAV format to raw PCM for streaming."""
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype='float32')
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32768).astype(np.int16)
        return audio_int16.tobytes()
