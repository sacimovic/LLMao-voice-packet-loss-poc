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
