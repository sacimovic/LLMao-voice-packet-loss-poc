use anyhow::{Context, Result};
use aws_sdk_polly::{
    types::{Engine, OutputFormat, VoiceId},
    Client as PollyClient,
};

pub struct TTSEngine {
    client: PollyClient,
}

impl TTSEngine {
    pub async fn new() -> Result<Self> {
        let config = aws_config::load_from_env().await;
        let client = PollyClient::new(&config);

        println!("[Polly] Initialized AWS Polly TTS");
        Ok(Self { client })
    }

    pub async fn synthesize(
        &self,
        text: &str,
        speaker_wav: Option<&[u8]>,
    ) -> Result<(Vec<f32>, u32)> {
        // Analyze speaker audio to pick best voice
        let voice_id = speaker_wav
            .map(select_voice_from_audio_fast)
            .unwrap_or(VoiceId::Ruth);

        println!("[Polly] Selected voice: {:?}", voice_id);
        println!("[Polly] Synthesizing text: '{}'", text);

        let output = self
            .client
            .synthesize_speech()
            .engine(Engine::Neural)
            .output_format(OutputFormat::Pcm)
            .sample_rate("16000")
            .text(text)
            .voice_id(voice_id)
            .send()
            .await
            .context("Failed to call AWS Polly")?;

        let audio_bytes = output
            .audio_stream
            .collect()
            .await
            .context("Failed to read audio stream from Polly")?
            .into_bytes();

        // Polly PCM is 16-bit signed integers at 16kHz
        let samples = pcm_to_f32(&audio_bytes);
        println!("[Polly] Generated {} samples at 16000Hz", samples.len());

        Ok((samples, 16000))
    }
}

fn select_voice_from_audio_fast(audio_bytes: &[u8]) -> VoiceId {
    match parse_audio_samples_fast(audio_bytes, 8_000) {
        Ok(samples) if samples.len() > 500 => {
            let pitch_estimate = estimate_pitch(&samples);
            println!("[Polly] Fast pitch estimate: {:.1} Hz", pitch_estimate);
            if pitch_estimate < 150.0 {
                println!("[Polly] Detected male speaker");
                VoiceId::Matthew
            } else if pitch_estimate < 200.0 {
                println!("[Polly] Detected neutral speaker");
                VoiceId::Ruth
            } else {
                println!("[Polly] Detected female speaker");
                VoiceId::Joanna
            }
        }
        _ => {
            println!("[Polly] Voice analysis unavailable, using default");
            VoiceId::Ruth
        }
    }
}

// Simple pitch estimation using autocorrelation
fn estimate_pitch(samples: &[f32]) -> f32 {
    if samples.len() < 1000 {
        return 150.0; // Default mid-range
    }

    // Use first 2048 samples for analysis
    let n = samples.len().min(2048);
    let samples = &samples[..n];

    // Find autocorrelation peak in expected pitch range
    // For human speech: 80-400 Hz at 16kHz = lag of 40-200 samples
    let mut max_corr = 0.0;
    let mut best_lag = 100;

    for lag in 40..200 {
        if lag >= n {
            break;
        }

        let mut corr = 0.0;
        for i in 0..(n - lag) {
            corr += samples[i] * samples[i + lag];
        }

        if corr > max_corr {
            max_corr = corr;
            best_lag = lag;
        }
    }

    // Convert lag to frequency (sample_rate / lag)
    16000.0 / best_lag as f32
}

fn parse_audio_samples_fast(audio_bytes: &[u8], max_samples: usize) -> Result<Vec<f32>> {
    use std::io::Cursor;
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::conv::FromSample;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::probe::Hint;

    let cursor = Cursor::new(audio_bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("wav");

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &Default::default(), &Default::default())
        .context("Failed to probe audio format")?;

    let mut format = probed.format;
    let track = format.default_track().context("No audio track found")?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &Default::default())
        .context("Failed to create decoder")?;

    let mut collected = Vec::with_capacity(max_samples);

    while collected.len() < max_samples {
        match format.next_packet() {
            Ok(packet) if packet.track_id() == track_id => {
                if let Ok(decoded) = decoder.decode(&packet) {
                    match decoded {
                        AudioBufferRef::F32(buf) => {
                            let channels = buf.spec().channels.count();
                            let frames = buf.frames();
                            push_frames(
                                &mut collected,
                                channels,
                                frames,
                                max_samples,
                                |ch, frame| buf.chan(ch)[frame],
                            );
                        }
                        AudioBufferRef::S16(buf) => {
                            let channels = buf.spec().channels.count();
                            let frames = buf.frames();
                            push_frames(
                                &mut collected,
                                channels,
                                frames,
                                max_samples,
                                |ch, frame| f32::from_sample(buf.chan(ch)[frame]),
                            );
                        }
                        AudioBufferRef::S32(buf) => {
                            let channels = buf.spec().channels.count();
                            let frames = buf.frames();
                            push_frames(
                                &mut collected,
                                channels,
                                frames,
                                max_samples,
                                |ch, frame| f32::from_sample(buf.chan(ch)[frame]),
                            );
                        }
                        AudioBufferRef::U16(buf) => {
                            let channels = buf.spec().channels.count();
                            let frames = buf.frames();
                            push_frames(
                                &mut collected,
                                channels,
                                frames,
                                max_samples,
                                |ch, frame| f32::from_sample(buf.chan(ch)[frame]),
                            );
                        }
                        AudioBufferRef::U8(buf) => {
                            let channels = buf.spec().channels.count();
                            let frames = buf.frames();
                            push_frames(
                                &mut collected,
                                channels,
                                frames,
                                max_samples,
                                |ch, frame| f32::from_sample(buf.chan(ch)[frame]),
                            );
                        }
                        _ => {}
                    }
                }
            }
            _ => break,
        }
    }

    Ok(collected)
}

fn push_frames<F>(
    out: &mut Vec<f32>,
    channels: usize,
    frames: usize,
    max_samples: usize,
    mut sample_fn: F,
) where
    F: FnMut(usize, usize) -> f32,
{
    let remaining = max_samples.saturating_sub(out.len());
    let frames_to_pull = frames.min(remaining);
    for frame_idx in 0..frames_to_pull {
        let mut mixed = 0.0f32;
        for ch in 0..channels {
            mixed += sample_fn(ch, frame_idx);
        }
        out.push(mixed / channels as f32);
    }
}

// Convert Polly PCM to f32 samples
fn pcm_to_f32(pcm_bytes: &[u8]) -> Vec<f32> {
    let mut samples = Vec::with_capacity(pcm_bytes.len() / 2);

    for chunk in pcm_bytes.chunks_exact(2) {
        let sample_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(sample_i16 as f32 / 32768.0);
    }

    samples
}
