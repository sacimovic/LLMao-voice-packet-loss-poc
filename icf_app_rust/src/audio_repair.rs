use anyhow::{Context, Result};
use reqwest::multipart;
use serde::Deserialize;
use std::io::Cursor;
use hound::{WavSpec, WavWriter};

#[derive(Debug, Deserialize)]
struct RepairResponse {
    asr_text: String,
    repaired_text: String,
    tts_wav_b64: String,
}

pub struct AudioRepairer {
    backend_url: String,
    client: reqwest::Client,
}

impl AudioRepairer {
    pub fn new(backend_url: String) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;
        
        Ok(Self {
            backend_url,
            client,
        })
    }

    pub async fn repair_audio(&self, samples: &[i16]) -> Result<Vec<i16>> {
        log::info!("🔧 Preparing to send {} samples to backend", samples.len());
        
        // Convert samples to WAV bytes
        let wav_bytes = samples_to_wav(samples, 16000)?;
        log::info!("📦 Created WAV file: {} bytes", wav_bytes.len());
        
        // Create multipart form
        let file_part = multipart::Part::bytes(wav_bytes)
            .file_name("audio.wav")
            .mime_str("audio/wav")?;
        
        let form = multipart::Form::new()
            .part("file", file_part)
            .text("degrade_percent", "30")
            .text("degrade_mode", "percentage")
            .text("whisper_model", "base")
            .text("repair_model", "bedrock")
            .text("synth_all_text", "true");

        // Send request to backend
        let url = format!("{}/process", self.backend_url);
        log::info!("🌐 Sending POST request to: {}", url);
        
        let response = self.client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                log::error!("❌ HTTP request failed: {}", e);
                if e.is_timeout() {
                    log::error!("   ↳ Request timed out");
                } else if e.is_connect() {
                    log::error!("   ↳ Failed to connect to backend at {}", url);
                } else if e.is_request() {
                    log::error!("   ↳ Invalid request");
                }
                e
            })
            .context("Failed to send request to backend")?;

        log::info!("📡 Received response: {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Backend returned error {}: {}", status, error_text);
        }

        let repair_response: RepairResponse = response
            .json()
            .await
            .context("Failed to parse backend response")?;

        log::info!("🔤 ASR: {}", repair_response.asr_text);
        log::info!("✨ Repaired: {}", repair_response.repaired_text);

        // Decode the repaired audio from base64
        use base64::Engine;
        let repaired_wav = base64::engine::general_purpose::STANDARD
            .decode(&repair_response.tts_wav_b64)
            .context("Failed to decode repaired audio")?;

        // Convert WAV back to samples
        let (samples, source_sample_rate) = wav_to_samples_with_rate(&repaired_wav)?;
        
        // Resample to 16kHz if needed (XTTS outputs 24kHz)
        let resampled = if source_sample_rate != 16000 {
            log::info!("🔄 Resampling from {}Hz to 16000Hz", source_sample_rate);
            resample_audio(&samples, source_sample_rate, 16000)?
        } else {
            samples
        };
        
        Ok(resampled)
    }
}

fn samples_to_wav(samples: &[i16], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec)?;
        for &sample in samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
    }

    Ok(cursor.into_inner())
}

fn wav_to_samples_with_rate(wav_bytes: &[u8]) -> Result<(Vec<i16>, u32)> {
    let cursor = Cursor::new(wav_bytes);
    let mut reader = hound::WavReader::new(cursor)?;
    let sample_rate = reader.spec().sample_rate;
    
    let samples: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
    let samples = samples.context("Failed to read samples from WAV")?;
    
    Ok((samples, sample_rate))
}

fn resample_audio(samples: &[i16], from_rate: u32, to_rate: u32) -> Result<Vec<i16>> {
    use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
    
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    
    // Convert i16 to f32
    let input_f32: Vec<f32> = samples
        .iter()
        .map(|&s| s as f32 / i16::MAX as f32)
        .collect();
    
    // Rubato expects channels as Vec<Vec<f32>>, we have mono
    let input_channels = vec![input_f32];
    
    // Create resampler
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    
    let mut resampler = SincFixedIn::<f32>::new(
        to_rate as f64 / from_rate as f64,
        2.0,
        params,
        input_channels[0].len(),
        1,
    )?;
    
    // Process audio
    let output_channels = resampler.process(&input_channels, None)?;
    
    // Convert back to i16
    let output_i16: Vec<i16> = output_channels[0]
        .iter()
        .map(|&s| (s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
        .collect();
    
    Ok(output_i16)
}
