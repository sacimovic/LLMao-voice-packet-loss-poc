use anyhow::{Context, Result};
use tracing::warn;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

const DEFAULT_LANGUAGE: &str = "en";

pub struct WhisperModel {
    ctx: WhisperContext,
    thread_count: i32,
}

impl WhisperModel {
    pub fn new(model_path: &str) -> Result<Self> {
        #[allow(deprecated)]
        let ctx = WhisperContext::new(model_path).context("Failed to load Whisper model")?;
    let logical_cpus = num_cpus::get().max(1) as i32;
    let thread_count = logical_cpus.clamp(2, 6);
        Ok(Self { ctx, thread_count })
    }

    pub fn transcribe(&self, audio_samples: &[f32]) -> Result<String> {
        match self.try_transcribe(audio_samples, true) {
            Ok(text) => Ok(text),
            Err(err) => {
                warn!("Whisper speed_up run failed ({}). Retrying without speed_up.", err);
                self.try_transcribe(audio_samples, false)
            }
        }
    }

    fn try_transcribe(&self, audio_samples: &[f32], speed_up: bool) -> Result<String> {
        let mut state = self
            .ctx
            .create_state()
            .context("Failed to create Whisper state")?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_n_threads(self.thread_count);
        params.set_language(Some(DEFAULT_LANGUAGE));
        params.set_no_context(true);
        params.set_single_segment(true);
        params.set_temperature(0.0);
        params.set_speed_up(speed_up);
        params.set_max_len(120);
        params.set_translate(false);
        params.set_suppress_blank(true);
        params.set_suppress_non_speech_tokens(true);

        state
            .full(params, audio_samples)
            .context("Failed to run Whisper inference")?;

        let num_segments = state
            .full_n_segments()
            .context("Failed to get segment count")?;
        let mut full_text = String::new();

        for i in 0..num_segments {
            let segment = state
                .full_get_segment_text(i)
                .context("Failed to get segment text")?;
            full_text.push_str(&segment);
            full_text.push(' ');
        }

        Ok(full_text.trim().to_string())
    }
}
