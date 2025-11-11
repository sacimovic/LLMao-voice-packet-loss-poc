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

# Suppress noisy warnings from dependencies
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from icf_media_sdk import CopilotCall, CopilotMode, copilot_app, MixMode

from audio_processor import AudioBuffer, PacketLossDetector, AudioRepairer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TRIGGER_PATTERNS = [
    r'\bfix\s+(?:the\s+)?audio\b',
    r'\brepair\s+(?:the\s+)?audio\b',
    r'\baudio\s+(?:is\s+)?broken\b',
]
BUFFER_DURATION_MS = 5000  # Keep last 5 seconds of audio
COPILOT_PORT = 8084

# Create recordings directory for debugging
RECORDINGS_DIR = Path(__file__).parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)


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

        logger.info(
            f"📞 Call started: {call.calling_number} → {call.called_number} "
            f"(Sample rate: {self.sample_rate}Hz)"
        )

    def check_trigger_phrase(self, text: str) -> bool:
        """Check if text contains a trigger phrase."""
        text_lower = text.lower()
        for pattern in TRIGGER_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

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
        """Triggered repair process - find and fix distorted audio.

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
            # Acknowledge the request
            logger.info(f"🔧 Starting audio repair for {target_stream} (requested by {requesting_party})")

            # Switch to bidirectional if needed
            if self.call.call_mode == CopilotMode.LISTEN_ONLY:
                logger.info("🔄 Switching to bidirectional mode...")
                await self.call.set_bidirectional()
                logger.info("✅ Mode switch complete, now in bidirectional mode")
            
            # TODO: Re-enable announcement once FMO TTS is configured properly
            # For now, skip the announcement to avoid connection issues
            logger.info("📢 Skipping announcement, proceeding directly to repair...")

            # Get the audio buffer for the target stream
            if target_stream == "caller":
                target_audio = self.caller_buffer.get_audio()
                target_duration = self.caller_buffer.duration_ms()
            else:
                target_audio = self.callee_buffer.get_audio()
                target_duration = self.callee_buffer.duration_ms()

            logger.info(f"📊 Analyzing {target_stream} audio - {target_duration:.0f}ms")

            # Check if target stream has degradation
            has_degradation = self.detector.has_degradation(target_audio, self.sample_rate)

            if not has_degradation:
                logger.info(f"✅ No degradation detected in {target_stream} audio")
                self.call.say(
                    "I don't detect any audio issues in the recent conversation.",
                    mix_mode=MixMode.DUCK
                )
                return

            # Repair the target audio stream and send to requesting party
            logger.info(f"✅ Detecting degradation in {target_stream} stream")
            await self.repair_stream(target_audio, target_stream, send_to=requesting_party)

            self.repair_count += 1
            logger.info(f"✅ Audio repair completed (total repairs: {self.repair_count})")

        except Exception as e:
            logger.error(f"❌ Error during repair: {e}", exc_info=True)
            self.call.say(
                "Sorry, I encountered an error while repairing the audio.",
                mix_mode=MixMode.DUCK
            )
        finally:
            self.is_processing = False

    async def repair_stream(self, audio: np.ndarray, stream_name: str, send_to: str):
        """Repair a specific audio stream.
        
        Args:
            audio: The audio data to repair
            stream_name: 'caller' or 'callee' - which stream is being repaired
            send_to: 'caller' or 'callee' - who should hear the repaired audio
        """
        logger.info(f"🔧 Repairing {stream_name} audio (will send to {send_to})...")

        try:
            # Save original corrupted audio for debugging
            import soundfile as sf
            original_file = RECORDINGS_DIR / f"{self.call.call_id}-{stream_name}-original-{self.repair_count}.wav"
            sf.write(original_file, audio, self.sample_rate)
            logger.debug(f"💾 Saved original audio: {original_file}")

            # Create a speaker reference from the audio
            import io
            reference_wav = io.BytesIO()
            sf.write(reference_wav, audio, self.sample_rate, format='WAV')
            reference_bytes = reference_wav.getvalue()

            # Run the repair pipeline in a thread pool to avoid blocking the event loop
            logger.info(f"🔄 Running AI repair pipeline in background thread...")
            try:
                loop = asyncio.get_event_loop()
                original_text, repaired_text, repaired_wav = await loop.run_in_executor(
                    None,  # Use default thread pool
                    self.repairer.repair_audio,
                    audio,
                    reference_bytes
                )
                logger.info(f"✅ AI repair pipeline completed successfully")
            except Exception as repair_error:
                logger.error(f"❌ Error in AI repair pipeline: {repair_error}", exc_info=True)
                raise

            logger.info(f"📝 Original: \"{original_text}\"")
            logger.info(f"📝 Repaired: \"{repaired_text}\"")

            # Verify we're in bidirectional mode before sending
            logger.info(f"🔍 Current call mode: {self.call.call_mode}")
            if self.call.call_mode != CopilotMode.BIDIRECTIONAL:
                logger.error(f"❌ Not in bidirectional mode! Cannot send audio.")
                return

            # Convert WAV to PCM and inject into call
            logger.info(f"🔄 Converting repaired WAV to PCM...")
            repaired_pcm = self.repairer.convert_wav_to_pcm(repaired_wav)
            pcm_duration_sec = len(repaired_pcm) / (self.sample_rate * 2)  # 2 bytes per sample (16-bit)
            logger.info(f"📡 Sending {len(repaired_pcm)} bytes of PCM audio ({pcm_duration_sec:.2f}s) to {send_to}")

            # Chunk audio into 20ms packets (standard RTP packet size)
            chunk_duration_ms = 20
            bytes_per_ms = (self.sample_rate * 2) // 1000  # 2 bytes per sample, 16-bit
            chunk_size = bytes_per_ms * chunk_duration_ms
            chunks = [repaired_pcm[i:i + chunk_size] for i in range(0, len(repaired_pcm), chunk_size)]
            logger.info(f"📦 Split audio into {len(chunks)} chunks of {chunk_duration_ms}ms each")

            # Inject the repaired audio to the requesting party only, chunk by chunk
            try:
                logger.info(f"🚀 Starting to send {len(chunks)} chunks...")
                for i, chunk in enumerate(chunks):
                    if send_to == "caller":
                        self.call.send_audio_to_caller(chunk, mix_mode=MixMode.OVERRIDE)
                    else:
                        self.call.send_audio_to_callee(chunk, mix_mode=MixMode.OVERRIDE)
                    
                    if i == 0:
                        logger.info(f"✅ Successfully sent first chunk ({len(chunk)} bytes)")
                    elif (i + 1) % 50 == 0:  # Log every 50 chunks
                        logger.info(f"📤 Sent {i + 1}/{len(chunks)} chunks...")
                    
                    # Small delay between chunks to simulate real-time streaming
                    if i < len(chunks) - 1:  # Don't delay after last chunk
                        await asyncio.sleep(chunk_duration_ms / 1000.0)
                logger.info(f"✅ Injected repaired {stream_name} audio to {send_to} ({len(chunks)} chunks)")
            except Exception as audio_error:
                logger.error(f"❌ Error sending audio: {audio_error}", exc_info=True)
                raise

            # Save repaired audio for debugging
            repaired_file = RECORDINGS_DIR / f"{self.call.call_id}-{stream_name}-repaired-{self.repair_count}.wav"
            with open(repaired_file, 'wb') as f:
                f.write(repaired_wav)
            logger.debug(f"💾 Saved repaired audio: {repaired_file}")

        except Exception as e:
            logger.error(f"❌ Failed to repair {stream_name}: {e}", exc_info=True)
            raise


@copilot_app(port=COPILOT_PORT)
async def llmao_repair_copilot(call: CopilotCall):
    """Main copilot entry point."""
    copilot = AudioRepairCopilot(call)

    try:
        # Register audio handlers to buffer incoming audio
        call.on_caller_audio(lambda data: copilot.caller_buffer.add_chunk(data))
        call.on_callee_audio(lambda data: copilot.callee_buffer.add_chunk(data))

        # Register partial utterance handler to detect trigger phrase
        call.on_partial_utterance(copilot.handle_partial_utterance)

        # Wait for call to end
        await call.wait_for_end()

    except Exception as e:
        logger.error(f"❌ Error in copilot: {e}", exc_info=True)
    finally:
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
