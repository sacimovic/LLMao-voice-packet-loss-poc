use anyhow::{Context, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

pub struct WhisperModel {
    ctx: WhisperContext,
}

impl WhisperModel {
    pub fn new(model_path: &str) -> Result<Self> {
        let ctx = WhisperContext::new(model_path).context("Failed to load Whisper model")?;
        Ok(Self { ctx })
    }

    pub fn transcribe(&self, audio_samples: &[f32]) -> Result<String> {
        let mut state = self.ctx.create_state().context("Failed to create Whisper state")?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state.full(params, audio_samples).context("Failed to run Whisper inference")?;

        let num_segments = state.full_n_segments().context("Failed to get segment count")?;
        let mut full_text = String::new();

        for i in 0..num_segments {
            let segment = state.full_get_segment_text(i).context("Failed to get segment text")?;
            full_text.push_str(&segment);
            full_text.push(' ');
        }

        Ok(full_text.trim().to_string())
    }
}
