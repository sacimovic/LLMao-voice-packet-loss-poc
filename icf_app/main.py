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
import numpy as np
from pathlib import Path

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
            logger.info(f"🎯 Trigger detected: \"{text}\"")
            await self.trigger_repair()
    
    async def trigger_repair(self):
        """Triggered repair process - find and fix distorted audio."""
        if self.is_processing:
            logger.warning("⚠️ Already processing, skipping...")
            return
        
        self.is_processing = True
        
        try:
            # Acknowledge the request
            logger.info("🔧 Starting audio repair process...")
            
            # Switch to bidirectional if needed
            if self.call.call_mode == CopilotMode.LISTEN_ONLY:
                logger.info("🔄 Switching to bidirectional mode...")
                await self.call.set_bidirectional()
            
            # Let them know we're working on it
            self.call.say(
                "I'll fix the audio for you. One moment please.",
                mix_mode=MixMode.DUCK
            )
            
            # Analyze recent audio for degradation
            caller_audio = self.caller_buffer.get_audio()
            callee_audio = self.callee_buffer.get_audio()
            
            logger.info(
                f"📊 Analyzing audio - Caller: {self.caller_buffer.duration_ms():.0f}ms, "
                f"Callee: {self.callee_buffer.duration_ms():.0f}ms"
            )
            
            # Check which stream has degradation
            caller_degraded = self.detector.has_degradation(caller_audio, self.sample_rate)
            callee_degraded = self.detector.has_degradation(callee_audio, self.sample_rate)
            
            if not caller_degraded and not callee_degraded:
                logger.info("✅ No degradation detected in recent audio")
                self.call.say(
                    "I don't detect any audio issues in the recent conversation.",
                    mix_mode=MixMode.DUCK
                )
                return
            
            # Repair the degraded audio
            if caller_degraded:
                await self.repair_stream(caller_audio, "caller")
            
            if callee_degraded:
                await self.repair_stream(callee_audio, "callee")
            
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
    
    async def repair_stream(self, audio: np.ndarray, stream_name: str):
        """Repair a specific audio stream."""
        logger.info(f"🔧 Repairing {stream_name} audio...")
        
        try:
            # Create a speaker reference from the audio
            import io
            import soundfile as sf
            reference_wav = io.BytesIO()
            sf.write(reference_wav, audio, self.sample_rate, format='WAV')
            reference_bytes = reference_wav.getvalue()
            
            # Run the repair pipeline
            original_text, repaired_text, repaired_wav = self.repairer.repair_audio(
                audio,
                speaker_reference=reference_bytes
            )
            
            logger.info(f"📝 Original: \"{original_text}\"")
            logger.info(f"📝 Repaired: \"{repaired_text}\"")
            
            # Convert WAV to PCM and inject into call
            repaired_pcm = self.repairer.convert_wav_to_pcm(repaired_wav)
            
            # Inject the repaired audio
            self.call.send(repaired_pcm, mix_mode=MixMode.OVERRIDE)
            
            logger.info(f"✅ Injected repaired {stream_name} audio")
            
            # Save for debugging
            debug_file = RECORDINGS_DIR / f"{self.call.call_id}-{stream_name}-repair-{self.repair_count}.wav"
            with open(debug_file, 'wb') as f:
                f.write(repaired_wav)
            logger.debug(f"💾 Saved debug recording: {debug_file}")
            
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
            f"👋 Call ended - {copilot.repair_count} repairs performed"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("🎙️  LLMao Audio Repair Copilot")
    print("=" * 70)
    print(f"📡 Listening on ws://localhost:{COPILOT_PORT}")
    print(f"🎯 Trigger phrases: {', '.join([p.replace('\\b', '').replace('\\s+', ' ') for p in TRIGGER_PATTERNS])}")
    print(f"💾 Debug recordings: {RECORDINGS_DIR}")
    print("\nPress Ctrl+C to stop")
    print("=" * 70)
    
    try:
        asyncio.run(llmao_repair_copilot.run_forever())
    except KeyboardInterrupt:
        print("\n\n👋 LLMao Copilot stopped")
