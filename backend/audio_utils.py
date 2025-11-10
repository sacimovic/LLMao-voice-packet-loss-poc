import io
import os
import numpy as np
import soundfile as sf
from typing import Tuple, Union
from io import BytesIO
import importlib

os.environ.setdefault("TORCHAUDIO_USE_TORCHCODEC", "0")

def _try_import_whisper():
    try:
        import whisper  # type: ignore
        return whisper
    except Exception as e:
        raise RuntimeError("Whisper not installed. `pip install openai-whisper` and ensure ffmpeg is on PATH.") from e

def _ensure_transformers_beam_search_alias():
    try:
        transformers = importlib.import_module("transformers")
        if not hasattr(transformers, "BeamSearchScorer"):
            from transformers.generation.beam_search import BeamSearchScorer  # type: ignore
            setattr(transformers, "BeamSearchScorer", BeamSearchScorer)
    except Exception:
        pass


def _try_import_tts():
    try:
        _ensure_transformers_beam_search_alias()
        os.environ.setdefault("TORCHAUDIO_USE_TORCHCODEC", "0")
        try:
            import torchaudio  # type: ignore
            import torch  # type: ignore

            original_load = getattr(torchaudio, "load", None)

            def _safe_torchaudio_load(path, *args, **kwargs):
                try:
                    if original_load is not None:
                        return original_load(path, *args, **kwargs)
                except (ImportError, RuntimeError, ModuleNotFoundError):
                    pass
                audio, sr = sf.read(path, dtype="float32")
                if audio.ndim == 1:
                    tensor = torch.from_numpy(audio).unsqueeze(0)
                else:
                    tensor = torch.from_numpy(audio.T)
                return tensor, sr

            if original_load is not None:
                torchaudio.load = _safe_torchaudio_load  # type: ignore

            if hasattr(torchaudio, "set_audio_backend"):
                try:
                    torchaudio.set_audio_backend("soundfile")
                except Exception:
                    torchaudio.set_audio_backend("sox_io")
        except Exception:
            pass
        from TTS.api import TTS  # type: ignore
        return TTS
    except Exception as e:
        raise RuntimeError("Coqui TTS not installed. `pip install TTS`.") from e

def _try_import_transformers():
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore # noqa: F401
        return AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Transformers not installed. `pip install transformers sentencepiece`."
        ) from e

def load_audio_mono_16k(file_like: Union[str, BytesIO]) -> Tuple[int, np.ndarray]:
    y, sr = sf.read(file_like, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != 16000:
        import librosa  # type: ignore
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    if np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + 1e-8) * 0.9
    return sr, y

def write_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def degrade_audio_simulated_loss(y: np.ndarray, sr: int, loss_ratio: float = 0.3, chunk_ms: int = 40) -> np.ndarray:
    y = y.copy()
    n = len(y)
    chunk = int(sr * (chunk_ms / 1000.0))
    if chunk <= 0 or loss_ratio <= 0:
        return y
    total_chunks = n // chunk
    to_drop = int(total_chunks * loss_ratio)
    idxs = np.arange(total_chunks)
    np.random.shuffle(idxs)
    drop_idxs = idxs[:to_drop]
    for k in drop_idxs:
        start = k * chunk
        end = min(start + chunk, n)
        y[start:end] = 0.0
    return y

def degrade_audio_zero_window(y: np.ndarray, sr: int, start_ms: int, window_ms: int) -> np.ndarray:
    """Zero out a contiguous window of audio samples.

    The span is clamped to the clip boundaries. We round to the nearest sample so
    that window selections align with what the UI reports, avoiding off-by-one
    gaps that might leave a sliver of the word intact."""

    if window_ms <= 0 or len(y) == 0:
        return y

    y_out = y.copy()
    total_len = len(y_out)

    # Convert millisecond offsets to sample indices using rounding to sync with UI.
    start_samples = int(round(sr * max(0, start_ms) / 1000.0))
    window_samples = int(round(sr * max(0, window_ms) / 1000.0))
    if window_samples <= 0:
        return y_out

    start_samples = max(0, min(start_samples, total_len))
    end_samples = max(start_samples, min(start_samples + window_samples, total_len))

    if end_samples > start_samples:
        y_out[start_samples:end_samples] = 0.0

        # Zero a single safety sample on either side to avoid tiny leftover bursts.
        if start_samples > 0:
            y_out[start_samples - 1] = 0.0
        if end_samples < total_len:
            y_out[end_samples] = 0.0

    return y_out

def transcribe_whisper(y: np.ndarray, sr: int, model_size: str = "base") -> str:
    whisper = _try_import_whisper()
    tmp = io.BytesIO()
    sf.write(tmp, y, sr, format="WAV", subtype="PCM_16")
    tmp.seek(0)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(tmp.read())
        f.flush()
        path = f.name
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(path, fp16=False)
        return result.get("text", "").strip()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

_repair_cache = {"model": None, "tokenizer": None, "model_name": None}

def repair_text_with_local_model(asr_text: str, model_name: str = "google/flan-t5-small") -> str:
    if not asr_text.strip():
        return asr_text

    AutoModelForSeq2SeqLM, AutoTokenizer = _try_import_transformers()
    import torch  # type: ignore

    cache = _repair_cache
    if cache["model"] is None or cache["model_name"] != model_name:
        cache["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        cache["model"] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        cache["model_name"] = model_name

    tokenizer = cache["tokenizer"]
    model = cache["model"]

    prompt = (
        "You fix transcripts from degraded VoIP audio."
        " Return the repaired sentence only.\nTranscript: "
        f"{asr_text.strip()}"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return text or asr_text

_xtts_cache = {"model": None}

def synthesize_xtts(text: str, speaker_wav_bytes: bytes, language: str = "en") -> bytes:
    TTS = _try_import_tts()
    import torch  # type: ignore
    from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs  # type: ignore
    from TTS.config.shared_configs import BaseDatasetConfig  # type: ignore
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
    if _xtts_cache["model"] is None:
        with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]):
            _xtts_cache["model"] = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(speaker_wav_bytes)
        speaker_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        _xtts_cache["model"].tts_to_file(
            text=text,
            speaker_wav=speaker_path,
            language=language,
            file_path=f_out.name,
        )
        with open(f_out.name, "rb") as rf:
            out = rf.read()
    try:
        os.remove(speaker_path)
    except Exception:
        pass
    return out

def stitch_simple_crossfade(y_base: np.ndarray, sr: int, tts_wav: bytes, crossfade_ms: int = 25) -> bytes:
    import soundfile as sf
    import numpy as np
    import io
    y_tts, sr_tts = sf.read(io.BytesIO(tts_wav), dtype="float32")
    if y_tts.ndim > 1:
        y_tts = y_tts.mean(axis=1)
    if sr_tts != sr:
        import librosa
        y_tts = librosa.resample(y_tts, sr_tts, sr)
    n = len(y_base)
    if len(y_tts) < n:
        y_tts = np.pad(y_tts, (0, n - len(y_tts)))
    else:
        y_tts = y_tts[:n]
    ramp = (1 - np.cos(np.linspace(0, np.pi, n))) / 2.0
    mixed = (1 - ramp) * y_base + ramp * y_tts
    buf = io.BytesIO()
    sf.write(buf, mixed, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
