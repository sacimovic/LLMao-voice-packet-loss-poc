use anyhow::{Context, Result};
use aws_sdk_polly::{Client as PollyClient, types::{Engine, OutputFormat, VoiceId}};

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

    pub async fn synthesize(&self, text: &str, speaker_wav: Option<&[u8]>) -> Result<(Vec<f32>, u32)> {
        // Analyze speaker audio to pick best voice
        let voice_id = if let Some(audio) = speaker_wav {
            select_voice_from_audio(audio)?
        } else {
            VoiceId::Ruth // Default neural voice
        };
        
        println!("[Polly] Selected voice: {:?}", voice_id);
        println!("[Polly] Synthesizing text: '{}'", text);
        
        let output = self.client
            .synthesize_speech()
            .engine(Engine::Neural)
            .output_format(OutputFormat::Pcm)
            .sample_rate("16000")
            .text(text)
            .voice_id(voice_id)
            .send()
            .await
            .context("Failed to call AWS Polly")?;

        let audio_bytes = output.audio_stream
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

// Analyze audio to select best matching Polly voice
fn select_voice_from_audio(audio_bytes: &[u8]) -> Result<VoiceId> {
    // Parse the audio to get samples
    let samples = match parse_audio_samples(audio_bytes) {
        Ok(s) => s,
        Err(_) => {
            println!("[Polly] Could not analyze audio, using default voice");
            return Ok(VoiceId::Ruth);
        }
    };
    
    // Calculate average pitch (simple zero-crossing rate estimate)
    let pitch_estimate = estimate_pitch(&samples);
    
    println!("[Polly] Estimated pitch: {:.1} Hz", pitch_estimate);
    
    // Select voice based on pitch
    // Male voices: ~85-180 Hz
    // Female voices: ~165-255 Hz
    let voice = if pitch_estimate < 150.0 {
        // Lower pitch - likely male
        println!("[Polly] Detected male speaker");
        VoiceId::Matthew // Male neural voice
    } else if pitch_estimate < 200.0 {
        // Mid-range - could be either, use neutral
        println!("[Polly] Detected neutral/ambiguous speaker");
        VoiceId::Ruth // Female neural voice (versatile)
    } else {
        // Higher pitch - likely female
        println!("[Polly] Detected female speaker");
        VoiceId::Joanna // Female neural voice
    };
    
    Ok(voice)
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

// Parse audio bytes to get f32 samples (supports multiple formats via symphonia)
fn parse_audio_samples(audio_bytes: &[u8]) -> Result<Vec<f32>> {
    // Use the same audio loader as the main pipeline
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::probe::Hint;
    use std::io::Cursor;
    
    let cursor = Cursor::new(audio_bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    
    let mut hint = Hint::new();
    hint.with_extension("mp3"); // Try MP3 first since that's common
    
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &Default::default(), &Default::default())
        .context("Failed to probe audio format")?;
    
    let mut format = probed.format;
    let track = format.default_track()
        .context("No audio track found")?;
    
    let track_id = track.id;
    let codec_params = track.codec_params.clone();
    
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &Default::default())
        .context("Failed to create decoder")?;
    
    let mut samples = Vec::new();
    
    // Decode all packets
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        
        match decoder.decode(&packet) {
            Ok(decoded) => {
                use symphonia::core::audio::{AudioBufferRef, Signal};
                use symphonia::core::conv::FromSample;
                
                // Convert audio buffer to f32 samples
                match decoded {
                    AudioBufferRef::F32(buf) => {
                        let channels = buf.spec().channels.count();
                        let frames = buf.frames();
                        
                        // Convert to mono by averaging channels
                        for frame_idx in 0..frames {
                            let mut sample_sum = 0.0_f32;
                            for ch_idx in 0..channels {
                                sample_sum += buf.chan(ch_idx)[frame_idx];
                            }
                            samples.push(sample_sum / channels as f32);
                        }
                    }
                    AudioBufferRef::U8(buf) => {
                        let channels = buf.spec().channels.count();
                        let frames = buf.frames();
                        for frame_idx in 0..frames {
                            let mut sample_sum = 0.0_f32;
                            for ch_idx in 0..channels {
                                sample_sum += f32::from_sample(buf.chan(ch_idx)[frame_idx]);
                            }
                            samples.push(sample_sum / channels as f32);
                        }
                    }
                    AudioBufferRef::U16(buf) => {
                        let channels = buf.spec().channels.count();
                        let frames = buf.frames();
                        for frame_idx in 0..frames {
                            let mut sample_sum = 0.0_f32;
                            for ch_idx in 0..channels {
                                sample_sum += f32::from_sample(buf.chan(ch_idx)[frame_idx]);
                            }
                            samples.push(sample_sum / channels as f32);
                        }
                    }
                    AudioBufferRef::U24(buf) => {
                        let channels = buf.spec().channels.count();
                        let frames = buf.frames();
                        for frame_idx in 0..frames {
                            let mut sample_sum = 0.0_f32;
                            for ch_idx in 0..channels {
                                sample_sum += f32::from_sample(buf.chan(ch_idx)[frame_idx]);
                            }
                            samples.push(sample_sum / channels as f32);
                        }
                    }
                    AudioBufferRef::U32(buf) => {
                        let channels = buf.spec().channels.count();
                        let frames = buf.frames();
                        for frame_idx in 0..frames {
                            let mut sample_sum = 0.0_f32;
                            for ch_idx in 0..channels {
                                sample_sum += f32::from_sample(buf.chan(ch_idx)[frame_idx]);
                            }
                            samples.push(sample_sum / channels as f32);
                        }
                    }
                    AudioBufferRef::S8(buf) => {
                        let channels = buf.spec().channels.count();
                        let frames = buf.frames();
                        for frame_idx in 0..frames {
                            let mut sample_sum = 0.0_f32;
                            for ch_idx in 0..channels {
                                sample_sum += f32::from_sample(buf.chan(ch_idx)[frame_idx]);
                            }
                            samples.push(sample_sum / channels as f32);
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        let channels = buf.spec().channels.count();
                        let frames = buf.frames();
                        for frame_idx in 0..frames {
                            let mut sample_sum = 0.0_f32;
                            for ch_idx in 0..channels {
                                sample_sum += f32::from_sample(buf.chan(ch_idx)[frame_idx]);
                            }
                            samples.push(sample_sum / channels as f32);
                        }
                    }
                    AudioBufferRef::S24(buf) => {
                        let channels = buf.spec().channels.count();
                        let frames = buf.frames();
                        for frame_idx in 0..frames {
                            let mut sample_sum = 0.0_f32;
                            for ch_idx in 0..channels {
                                sample_sum += f32::from_sample(buf.chan(ch_idx)[frame_idx]);
                            }
                            samples.push(sample_sum / channels as f32);
                        }
                    }
                    AudioBufferRef::S32(buf) => {
                        let channels = buf.spec().channels.count();
                        let frames = buf.frames();
                        for frame_idx in 0..frames {
                            let mut sample_sum = 0.0_f32;
                            for ch_idx in 0..channels {
                                sample_sum += f32::from_sample(buf.chan(ch_idx)[frame_idx]);
                            }
                            samples.push(sample_sum / channels as f32);
                        }
                    }
                    AudioBufferRef::F64(buf) => {
                        let channels = buf.spec().channels.count();
                        let frames = buf.frames();
                        for frame_idx in 0..frames {
                            let mut sample_sum = 0.0_f32;
                            for ch_idx in 0..channels {
                                sample_sum += buf.chan(ch_idx)[frame_idx] as f32;
                            }
                            samples.push(sample_sum / channels as f32);
                        }
                    }
                }
            }
            Err(_) => continue,
        }
    }
    
    Ok(samples)
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
