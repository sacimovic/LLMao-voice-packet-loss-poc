#!/usr/bin/env python3
"""
LLMao Copilot - Audio Packet Loss Repair for Phone Calls

This copilot listens to phone calls and repairs audio degradation when triggered
by a specific phrase. It uses Whisper for ASR, FLAN-T5 for text repair, and
Coqui XTTS for speech synthesis.

Trigger phrase: "fix the audio" or "repair audio"
"""
import asyncio
import logging
import re
import warnings
import numpy as np
from pathlib import Path
import sys
import time
from dataclasses import dataclass
from typing import Optional, List

# Suppress noisy warnings from dependencies
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from icf_media_sdk import CopilotCall, CopilotMode, copilot_app, MixMode

from audio_processor import AudioBuffer, PacketLossDetector, AudioRepairer

# Import audio degradation detector from backend
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))
from audio_degradation_detector import AudioDegradationDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Uncomment below for detailed debug logs from the detector
#logging.getLogger("audio_degradation_detector").setLevel(logging.DEBUG)

# Configuration
TRIGGER_PATTERNS = [
    r'\bfix\s+(?:the\s+)?audio\b',
    r'\brepair\s+(?:the\s+)?audio\b',
    r'\baudio\s+(?:is\s+)?broken\b',
]
BUFFER_DURATION_MS = 5000  # Keep last 5 seconds of audio
COPILOT_PORT = 8084

# Background repair configuration
MAX_REPAIRED_DURATION_S = 30  # Maximum accumulated repaired audio
SEGMENT_GAP_THRESHOLD_S = 2   # Gap to consider segments separate
REPAIRED_SEGMENT_TTL_S = 30   # Time to keep repaired segments before discarding
MIN_SPEECH_ENERGY = 0.01      # Minimum RMS energy to consider audio has speech
MIN_REPAIR_DURATION_S = 2.0   # Minimum duration of audio to send for repair

# Create recordings directory for debugging
RECORDINGS_DIR = Path(__file__).parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)


@dataclass
class RepairedSegment:
    """A segment of repaired audio."""
    audio_pcm: bytes           # Repaired audio as PCM bytes
    start_time: float          # When degradation started (monotonic time)
    end_time: float            # When degradation ended (monotonic time)
    stream_name: str           # 'caller' or 'callee'
    
    def duration_s(self) -> float:
        return self.end_time - self.start_time
    
    def age_s(self, current_time: float) -> float:
        """Time since this segment ended."""
        return current_time - self.end_time


class BackgroundRepairTracker:
    """Tracks ongoing background repair for a stream."""
    
    def __init__(self, stream_name: str):
        self.stream_name = stream_name
        self.is_repairing = False
        self.degradation_start_time: Optional[float] = None
        self.accumulated_audio: List[np.ndarray] = []
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()
        
    def start_degradation(self):
        """Mark that degradation has started."""
        if not self.is_repairing:
            self.is_repairing = True
            self.degradation_start_time = time.monotonic()
            self.accumulated_audio = []
            logger.info(f"🔴 [{self.stream_name}] Started background repair")
    
    async def add_audio_async(self, audio: np.ndarray):
        """Add audio chunk to accumulation (async with lock)."""
        async with self.lock:
            if self.is_repairing:
                self.accumulated_audio.append(audio)
    
    def get_duration_s(self, sample_rate: int) -> float:
        """Get total duration of accumulated audio."""
        if not self.accumulated_audio:
            return 0.0
        total_samples = sum(len(a) for a in self.accumulated_audio)
        return total_samples / sample_rate
    
    async def stop_and_get_segment_async(self, sample_rate: int) -> Optional[RepairedSegment]:
        """Stop repair and return the accumulated segment (async with lock)."""
        async with self.lock:
            if not self.is_repairing or not self.accumulated_audio:
                self.is_repairing = False
                return None
            
            # Concatenate all accumulated audio
            full_audio = np.concatenate(self.accumulated_audio)
            
            # Convert to PCM bytes
            pcm_bytes = (full_audio * 32768.0).astype(np.int16).tobytes()
            
            end_time = time.monotonic()
            segment = RepairedSegment(
                audio_pcm=pcm_bytes,
                start_time=self.degradation_start_time,
                end_time=end_time,
                stream_name=self.stream_name
            )
            
            logger.info(f"🟢 [{self.stream_name}] Completed background repair: {segment.duration_s():.1f}s")
            
            # Reset state
            self.is_repairing = False
            self.degradation_start_time = None
            self.accumulated_audio = []
            
            return segment
    
    def cancel(self):
        """Cancel any ongoing repair."""
        if self.task and not self.task.done():
            self.task.cancel()
        self.is_repairing = False
        self.accumulated_audio = []


class AudioRepairCopilot:
    """Manages state for an active call with audio repair capability."""

    def __init__(self, call: CopilotCall):
        self.call = call
        self.sample_rate = call.audio_sampling_rate_khz * 1000

        # Audio buffers for both parties
        self.caller_buffer = AudioBuffer(self.sample_rate, BUFFER_DURATION_MS)
        self.callee_buffer = AudioBuffer(self.sample_rate, BUFFER_DURATION_MS)

        # Processing components
        self.detector = PacketLossDetector()
        self.repairer = AudioRepairer(sample_rate=self.sample_rate)

        # State tracking
        self.is_processing = False
        self.repair_count = 0

        # Background repair tracking
        self.caller_repair_tracker = BackgroundRepairTracker("caller")
        self.callee_repair_tracker = BackgroundRepairTracker("callee")
        self.repaired_segments: List[RepairedSegment] = []
        self.cleanup_task: Optional[asyncio.Task] = None

        # Continuous recording for debugging
        self.recording_caller = []
        self.recording_callee = []
        self.recording_enabled = False  # Set to True to enable continuous recording

        # Track injected audio (what we send via send_audio_to_caller/callee)
        self.injected_to_caller = []
        self.injected_to_callee = []

        logger.info(
            f"📞 Call started: {call.calling_number} → {call.called_number} "
            f"(Sample rate: {self.sample_rate}Hz)"
        )
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_old_segments())

    def check_trigger_phrase(self, text: str) -> bool:
        """Check if text contains a trigger phrase."""
        text_lower = text.lower()
        for pattern in TRIGGER_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False
    
    async def _cleanup_old_segments(self):
        """Periodically clean up old repaired segments."""
        try:
            while True:
                await asyncio.sleep(5)  # Check every 5 seconds
                current_time = time.monotonic()
                
                # Remove segments older than TTL
                initial_count = len(self.repaired_segments)
                self.repaired_segments = [
                    seg for seg in self.repaired_segments
                    if seg.age_s(current_time) < REPAIRED_SEGMENT_TTL_S
                ]
                
                removed = initial_count - len(self.repaired_segments)
                if removed > 0:
                    logger.info(f"🗑️ Cleaned up {removed} expired repaired segment(s)")
        except asyncio.CancelledError:
            pass
    
    def _get_most_recent_segment(self, stream_name: str) -> Optional[RepairedSegment]:
        """Get the most recent continuous segment for a stream."""
        # Filter segments for this stream
        stream_segments = [s for s in self.repaired_segments if s.stream_name == stream_name]
        
        if not stream_segments:
            return None
        
        # Sort by end time (most recent last)
        stream_segments.sort(key=lambda s: s.end_time)
        
        # Start with the most recent segment
        recent = stream_segments[-1]
        
        # Look backwards to find all segments in the continuous group
        # (segments are continuous if gap between them is < SEGMENT_GAP_THRESHOLD_S)
        continuous_group = [recent]
        
        for i in range(len(stream_segments) - 2, -1, -1):
            current_seg = stream_segments[i]
            next_seg = continuous_group[0]
            
            gap = next_seg.start_time - current_seg.end_time
            
            if gap < SEGMENT_GAP_THRESHOLD_S:
                continuous_group.insert(0, current_seg)
            else:
                break  # Found a gap, stop looking
        
        # If multiple segments in group, merge them
        if len(continuous_group) == 1:
            return continuous_group[0]
        
        # Merge multiple segments
        merged_pcm = b''.join(seg.audio_pcm for seg in continuous_group)
        return RepairedSegment(
            audio_pcm=merged_pcm,
            start_time=continuous_group[0].start_time,
            end_time=continuous_group[-1].end_time,
            stream_name=stream_name
        )

    def auto_degradation_repair_requested(self, party: str, time: float):
        """Callback when degradation is detected automatically."""
        logger.info(f"⚠️ Detected audio degradation in {party} stream")
        
        # Convert SpeakingParty enum to string if needed
        if hasattr(party, 'value'):
            party_str = party.value  # Extract "caller" or "callee" from enum
        else:
            party_str = party
        
        # Start background repair for the affected stream
        tracker = self.caller_repair_tracker if party_str == "caller" else self.callee_repair_tracker
        
        if not tracker.is_repairing:
            tracker.start_degradation()
            # Kick off the background repair task and store reference
            task = asyncio.create_task(self._background_repair_stream(party_str))
            tracker.task = task
    
    async def _background_repair_stream(self, stream_name: str):
        """Background task that continuously repairs audio as degradation is detected."""
        tracker = self.caller_repair_tracker if stream_name == "caller" else self.callee_repair_tracker
        buffer = self.caller_buffer if stream_name == "caller" else self.callee_buffer
        
        logger.info(f"🔧 [{stream_name}] Starting background repair task")
        
        try:
            # Wait for enough audio to accumulate before attempting first repair
            await asyncio.sleep(2.0)
            
            while tracker.is_repairing:
                # Get current audio from buffer
                audio = buffer.get_audio()
                
                if len(audio) == 0:
                    logger.debug(f"⏳ [{stream_name}] Buffer empty, waiting for audio...")
                    await asyncio.sleep(0.5)
                    continue
                
                # Check if there's enough audio duration
                duration_s = len(audio) / self.sample_rate
                if duration_s < MIN_REPAIR_DURATION_S:
                    logger.debug(f"⏳ [{stream_name}] Only {duration_s:.1f}s in buffer, waiting for more...")
                    await asyncio.sleep(0.5)
                    continue
                
                # Check if audio has sufficient energy (not just silence)
                rms_energy = np.sqrt(np.mean(audio ** 2))
                if rms_energy < MIN_SPEECH_ENERGY:
                    logger.warning(f"⚠️ [{stream_name}] Audio too quiet (RMS: {rms_energy:.6f}), skipping repair")
                    await asyncio.sleep(1.0)
                    continue
                
                # Good to repair - we have enough audio with sufficient energy
                logger.info(f"🔄 [{stream_name}] Repairing {duration_s:.1f}s of audio (RMS: {rms_energy:.4f})...")
                
                try:
                    # Create reference bytes for XTTS
                    import soundfile as sf
                    import io
                    reference_wav = io.BytesIO()
                    sf.write(reference_wav, audio, self.sample_rate, format='WAV')
                    reference_bytes = reference_wav.getvalue()
                    
                    # Run repair in thread pool
                    loop = asyncio.get_event_loop()
                    original_text, repaired_text, repaired_wav = await loop.run_in_executor(
                        None,
                        self.repairer.repair_audio,
                        audio,
                        reference_bytes
                    )
                    
                    # Check if we got valid output
                    if not repaired_text or not repaired_text.strip():
                        logger.warning(f"⚠️ [{stream_name}] Repair produced empty text, skipping this chunk")
                    elif not repaired_wav or len(repaired_wav) == 0:
                        logger.warning(f"⚠️ [{stream_name}] Repair produced no audio, skipping this chunk")
                    else:
                        # Convert WAV to float32 samples for accumulation
                        wav_io = io.BytesIO(repaired_wav)
                        repaired_samples, repaired_sr = sf.read(wav_io, dtype='float32')
                        
                        # Resample to 16kHz if needed (XTTS outputs at 24kHz)
                        if repaired_sr != self.sample_rate:
                            logger.debug(f"🔄 [{stream_name}] Resampling from {repaired_sr}Hz to {self.sample_rate}Hz")
                            import librosa
                            repaired_samples = librosa.resample(
                                repaired_samples, 
                                orig_sr=repaired_sr, 
                                target_sr=self.sample_rate
                            )
                        
                        # Use async method to safely add audio
                        await tracker.add_audio_async(repaired_samples)
                        
                        logger.info(f"✅ [{stream_name}] Repaired and accumulated {len(repaired_samples)/self.sample_rate:.1f}s (text: \"{repaired_text[:50]}...\")")
                    
                except Exception as e:
                    logger.error(f"❌ [{stream_name}] Error repairing chunk: {e}")
                
                # Check if we've hit duration limit
                accumulated_duration = tracker.get_duration_s(self.sample_rate)
                if accumulated_duration >= MAX_REPAIRED_DURATION_S:
                    logger.warning(f"⏱️ [{stream_name}] Hit max duration ({accumulated_duration:.1f}s), stopping repair")
                    segment = await tracker.stop_and_get_segment_async(self.sample_rate)
                    if segment:
                        self.repaired_segments.append(segment)
                    break
                
                # Wait for more audio to accumulate (longer than processing time)
                # This prevents re-processing the same audio
                await asyncio.sleep(3.0)
            
            # Exited loop normally - save any accumulated audio
            logger.info(f"✅ [{stream_name}] Repair loop exited, saving accumulated audio")
            segment = await tracker.stop_and_get_segment_async(self.sample_rate)
            if segment:
                self.repaired_segments.append(segment)
                logger.info(f"💾 [{stream_name}] Saved {segment.duration_s():.1f}s of repaired audio")
                
        except asyncio.CancelledError:
            logger.info(f"🛑 [{stream_name}] Background repair cancelled")
            # Save any accumulated audio before exiting
            segment = await tracker.stop_and_get_segment_async(self.sample_rate)
            if segment:
                self.repaired_segments.append(segment)
                logger.info(f"💾 [{stream_name}] Saved {segment.duration_s():.1f}s of repaired audio before cancellation")
            else:
                logger.info(f"ℹ️ [{stream_name}] No accumulated audio to save on cancellation")
            raise  # Re-raise to properly exit the task
        except Exception as e:
            logger.error(f"❌ [{stream_name}] Background repair error: {e}", exc_info=True)
            tracker.cancel()

    async def handle_partial_utterance(self, partial):
        """Handle incoming partial transcriptions to detect trigger phrase."""
        text = partial.text
        party = partial.speaking_party.value

        logger.debug(f"💬 {party}: \"{text}\"")

        # Check for trigger phrase
        if self.check_trigger_phrase(text):
            logger.info(f"🎯 Trigger detected from {party}: \"{text}\"")
            # If caller complains, repair callee audio (what caller hears)
            # If callee complains, repair caller audio (what callee hears)
            await self.trigger_repair(requesting_party=party)

    async def trigger_repair(self, requesting_party: str):
        """Triggered repair process - play back pre-repaired audio.

        Args:
            requesting_party: 'caller' or 'callee' - who requested the repair
        """
        if self.is_processing:
            logger.warning("⚠️ Already processing, skipping...")
            return

        self.is_processing = True

        # Determine which stream to repair (opposite of who complained)
        target_stream = "callee" if requesting_party == "caller" else "caller"

        try:
            # Stop any ongoing background repair for this stream
            tracker = self.caller_repair_tracker if target_stream == "caller" else self.callee_repair_tracker
            task_was_running = tracker.is_repairing and tracker.task and not tracker.task.done()
            
            if tracker.is_repairing:
                logger.info(f"⏸️ Stopping background repair for {target_stream}")
                
                # If task is running, wait for current repair to finish
                if tracker.task and not tracker.task.done():
                    # Announce we're waiting for repair to complete
                    if self.call.call_mode == CopilotMode.LISTEN_ONLY:
                        logger.info("🔄 Switching to bidirectional mode...")
                        await self.call.set_bidirectional()
                        # Give mode switch time to complete
                        await asyncio.sleep(0.1)
                    
                    try:
                        self.call.say(
                            "Audio repair in progress, please wait.",
                            mix_mode=MixMode.DUCK,
                            send_to=requesting_party
                        )
                        logger.info("📢 Announced repair in progress")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to announce: {e}")
                    
                    # Set flag to stop after current iteration
                    tracker.is_repairing = False
                    
                    # Wait for current repair operation to complete (up to 30 seconds)
                    logger.info(f"⏳ Waiting for current repair operation to complete...")
                    try:
                        await asyncio.wait_for(tracker.task, timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"⏱️ Repair task timed out, cancelling forcefully")
                        tracker.task.cancel()
                        try:
                            await tracker.task
                        except asyncio.CancelledError:
                            pass
                    except asyncio.CancelledError:
                        pass
                
                # Get any accumulated audio
                segment = await tracker.stop_and_get_segment_async(self.sample_rate)
                if segment:
                    self.repaired_segments.append(segment)
                    logger.info(f"💾 Saved {segment.duration_s():.1f}s from interrupted repair")
            
            # Get the most recent repaired segment for the target stream
            segment = self._get_most_recent_segment(target_stream)
            
            logger.info(f"📋 [{target_stream}] Total segments available: {len([s for s in self.repaired_segments if s.stream_name == target_stream])}")
            
            if not segment:
                # Only play "no audio available" if we didn't just wait for repair
                if task_was_running:
                    logger.info(f"❌ Repair completed but produced no usable audio for {target_stream}")
                    message = "Audio repair completed, but no clear speech was detected in the degraded audio."
                else:
                    logger.info(f"❌ No repaired audio available for {target_stream}")
                    message = "I don't have any repaired audio available for recent issues."
                
                # Switch to bidirectional if needed
                if self.call.call_mode == CopilotMode.LISTEN_ONLY:
                    logger.info("🔄 Switching to bidirectional mode...")
                    await self.call.set_bidirectional()
                
                self.call.say(
                    message,
                    mix_mode=MixMode.DUCK,
                    send_to=requesting_party
                )
                return
            
            logger.info(f"🔧 Playing repaired audio for {target_stream} (requested by {requesting_party})")
            logger.info(f"📊 Segment duration: {segment.duration_s():.1f}s, age: {segment.age_s(time.monotonic()):.1f}s")

            # Switch to bidirectional if needed
            if self.call.call_mode == CopilotMode.LISTEN_ONLY:
                logger.info("🔄 Switching to bidirectional mode...")
                await self.call.set_bidirectional()
                logger.info("✅ Mode switch complete, now in bidirectional mode")

            # Announce playback
            logger.info("📢 Announcing repaired audio playback...")
            try:
                self.call.say("Playing repaired audio", mix_mode=MixMode.DUCK, send_to=requesting_party)
                logger.info("✅ Announced playback")
            except Exception as e:
                logger.warning(f"⚠️ Failed to announce: {e}")

            # Send the repaired audio
            await self._send_repaired_audio(segment.audio_pcm, target_stream, requesting_party)

            # Remove this segment from the list (it's been played)
            self.repaired_segments = [s for s in self.repaired_segments if s != segment]

            self.repair_count += 1
            logger.info(f"✅ Audio playback completed (total repairs: {self.repair_count})")

        except Exception as e:
            logger.error(f"❌ Error during repair: {e}", exc_info=True)
            try:
                self.call.say(
                    "Sorry, I encountered an error while playing the repaired audio.",
                    mix_mode=MixMode.DUCK
                )
            except:
                pass
        finally:
            self.is_processing = False
    
    async def _send_repaired_audio(self, pcm_bytes: bytes, stream_name: str, send_to: str):
        """Send repaired PCM audio to the call.
        
        Args:
            pcm_bytes: PCM audio data
            stream_name: Which stream was repaired ('caller' or 'callee')
            send_to: Who should hear it ('caller' or 'callee')
        """
        pcm_duration_sec = len(pcm_bytes) / (self.sample_rate * 2)  # 2 bytes per sample
        logger.info(f"📡 Sending {len(pcm_bytes)} bytes of PCM audio ({pcm_duration_sec:.2f}s) to {send_to}")

        # Use 100ms chunks
        chunk_duration_ms = 100
        bytes_per_ms = (self.sample_rate * 2) // 1000
        chunk_size = bytes_per_ms * chunk_duration_ms
        chunks = [pcm_bytes[i:i + chunk_size] for i in range(0, len(pcm_bytes), chunk_size)]
        
        logger.info(f"📦 Split audio into {len(chunks)} chunks of {chunk_duration_ms}ms each")

        try:
            for i, chunk in enumerate(chunks):
                if send_to == "caller":
                    self.call.send_audio_to_caller(chunk, mix_mode=MixMode.OVERRIDE)
                    self.injected_to_caller.append(chunk)
                else:
                    self.call.send_audio_to_callee(chunk, mix_mode=MixMode.OVERRIDE)
                    self.injected_to_callee.append(chunk)

                if i == 0:
                    logger.info(f"✅ Successfully sent first chunk ({len(chunk)} bytes)")
                elif (i + 1) % 10 == 0:
                    logger.info(f"📤 Sent {i + 1}/{len(chunks)} chunks...")

            logger.info(f"✅ Sent all repaired audio to {send_to} ({len(chunks)} chunks)")
        except Exception as e:
            logger.error(f"❌ Error sending audio: {e}", exc_info=True)
            raise



def save_pcm_recording(pcm_chunks: list, filename: Path, sample_rate: int) -> None:
    """Save a list of PCM chunks to a WAV file."""
    if not pcm_chunks:
        return
    
    import soundfile as sf
    pcm_bytes = b''.join(pcm_chunks)
    audio_float = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(filename, audio_float, sample_rate, subtype='PCM_16')
    duration_sec = len(audio_float) / sample_rate
    logger.info(f"💾 Saved {filename.name}: {duration_sec:.1f}s")


@copilot_app(port=COPILOT_PORT)
async def llmao_repair_copilot(call: CopilotCall):
    """Main copilot entry point."""
    copilot = AudioRepairCopilot(call)

    try:

        # Register audio handlers to buffer incoming audio AND record continuously
        def handle_caller_audio(data: bytes):
            copilot.caller_buffer.add_chunk(data)
            if copilot.recording_enabled:
                copilot.recording_caller.append(data)

        def handle_callee_audio(data: bytes):
            copilot.callee_buffer.add_chunk(data)
            if copilot.recording_enabled:
                copilot.recording_callee.append(data)

        call.on_caller_audio(handle_caller_audio)
        call.on_callee_audio(handle_callee_audio)

        # Register partial utterance handler to detect trigger phrase
        call.on_partial_utterance(copilot.handle_partial_utterance)

        logger.info(f"🔍 Setting up audio degradation detector...")
        # Hook up audio degradation detector
        detector = AudioDegradationDetector(
            on_repair_requested=copilot.auto_degradation_repair_requested
        )
        detector_task = asyncio.create_task(
            detector.start(call.caller_audio_stream, call.callee_audio_stream)
        )

        # Wait for call to end
        await call.wait_for_end()

    except Exception as e:
        logger.error(f"❌ Error in copilot: {e}", exc_info=True)
    finally:
        # Cancel background tasks
        if copilot.cleanup_task and not copilot.cleanup_task.done():
            copilot.cleanup_task.cancel()
        
        # Stop any ongoing background repairs
        copilot.caller_repair_tracker.cancel()
        copilot.callee_repair_tracker.cancel()
        
        # Save continuous recordings when call ends
        if copilot.recording_enabled:
            logger.info(f"💾 Saving continuous recordings...")
            save_pcm_recording(copilot.recording_caller, RECORDINGS_DIR / f"{call.call_id}-caller-continuous.wav", copilot.sample_rate)
            save_pcm_recording(copilot.recording_callee, RECORDINGS_DIR / f"{call.call_id}-callee-continuous.wav", copilot.sample_rate)
            save_pcm_recording(copilot.injected_to_caller, RECORDINGS_DIR / f"{call.call_id}-injected-to-caller.wav", copilot.sample_rate)
            save_pcm_recording(copilot.injected_to_callee, RECORDINGS_DIR / f"{call.call_id}-injected-to-callee.wav", copilot.sample_rate)

        logger.info(
            f"👋 Call ended: {call.call_id} "
            f"({call.calling_number} → {call.called_number}) - "
            f"{copilot.repair_count} repairs performed"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("🎙️  LLMao Audio Repair Copilot")
    print("=" * 70)
    print(f"📡 Listening on ws://localhost:{COPILOT_PORT}")
    trigger_display = ', '.join([p.replace('\\b', '').replace('\\s+', ' ') for p in TRIGGER_PATTERNS])
    print(f"🎯 Trigger phrases: {trigger_display}")
    print(f"💾 Debug recordings: {RECORDINGS_DIR}")
    print("\nPress Ctrl+C to stop")    
    print("=" * 70)

    try:
        asyncio.run(llmao_repair_copilot.run_forever())
    except KeyboardInterrupt:
        print("\n\n👋 LLMao Copilot stopped")
