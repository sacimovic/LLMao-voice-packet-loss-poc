use rand::Rng;

pub fn degrade_audio(samples: &[f32], packet_loss_percent: f32) -> Vec<f32> {
    if packet_loss_percent <= 0.0 {
        return samples.to_vec();
    }
    
    let mut rng = rand::thread_rng();
    let mut output = samples.to_vec();
    
    // 40ms frames at 16kHz
    let frame_size = 640;
    let num_frames = samples.len() / frame_size;
    
    for i in 0..num_frames {
        if rng.gen::<f32>() * 100.0 < packet_loss_percent {
            let start = i * frame_size;
            let end = (start + frame_size).min(output.len());
            for sample in &mut output[start..end] {
                *sample = 0.0;
            }
        }
    }
    
    output
}

pub fn degrade_audio_window(samples: &[f32], window_start_ms: u32, window_ms: u32) -> Vec<f32> {
    let mut output = samples.to_vec();
    let sample_rate = 16000;
    
    let start_sample = ((window_start_ms as f32 / 1000.0) * sample_rate as f32) as usize;
    let window_samples = ((window_ms as f32 / 1000.0) * sample_rate as f32) as usize;
    let end_sample = (start_sample + window_samples).min(output.len());
    
    if start_sample < output.len() {
        for sample in &mut output[start_sample..end_sample] {
            *sample = 0.0;
        }
    }
    
    output
}
