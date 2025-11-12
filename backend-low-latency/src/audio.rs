use anyhow::{Context, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use std::io::Cursor;

pub fn load_audio_mono_16k(bytes: &[u8]) -> Result<Vec<f32>> {
    let bytes_vec = bytes.to_vec();
    let cursor = Cursor::new(bytes_vec);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    
    let hint = Hint::new();
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();
    
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .context("Failed to probe audio format")?;
    
    let mut format = probed.format;
    let track = format.tracks().iter().find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .context("No audio track found")?;
    
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Failed to create decoder")?;
    
    let mut all_samples = Vec::new();
    
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let duration = decoded.capacity() as u64;
                let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
                sample_buf.copy_interleaved_ref(decoded);
                
                let samples = sample_buf.samples();
                if spec.channels.count() == 2 {
                    for chunk in samples.chunks(2) {
                        all_samples.push((chunk[0] + chunk[1]) / 2.0);
                    }
                } else {
                    all_samples.extend_from_slice(samples);
                }
            }
            Err(_) => continue,
        }
    }
    
    let original_sr = decoder.codec_params().sample_rate.unwrap_or(16000);
    if original_sr != 16000 {
        all_samples = resample(&all_samples, original_sr, 16000);
    }
    
    Ok(all_samples)
}

fn resample(samples: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if from_sr == to_sr {
        return samples.to_vec();
    }
    
    let ratio = from_sr as f64 / to_sr as f64;
    let out_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(out_len);
    
    for i in 0..out_len {
        let src_idx = (i as f64 * ratio) as usize;
        if src_idx < samples.len() {
            output.push(samples[src_idx]);
        }
    }
    
    output
}
