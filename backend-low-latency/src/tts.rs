use anyhow::{Context, Result};
use aws_config;
use aws_sdk_polly::{Client as PollyClient, types::{Engine, OutputFormat, VoiceId}};

pub struct TTSEngine {
    client: PollyClient,
}

impl TTSEngine {
    pub async fn new() -> Result<Self> {
        let config = aws_config::load_from_env().await;
        let client = PollyClient::new(&config);
        Ok(Self { client })
    }

    pub async fn synthesize(&self, text: &str, _speaker_wav: Option<&[u8]>) -> Result<Vec<f32>> {
        let response = self.client
            .synthesize_speech()
            .engine(Engine::Neural)
            .output_format(OutputFormat::Pcm)
            .sample_rate("16000")
            .text(text)
            .voice_id(VoiceId::Joanna)
            .send()
            .await
            .context("Failed to synthesize speech with Polly")?;

        let audio_stream = response.audio_stream;
        let audio_bytes = audio_stream
            .collect()
            .await
            .context("Failed to collect audio stream")?
            .into_bytes();

        // Convert PCM bytes (i16) to f32
        let mut samples = Vec::new();
        for chunk in audio_bytes.chunks_exact(2) {
            let sample_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
            let sample_f32 = sample_i16 as f32 / 32768.0;
            samples.push(sample_f32);
        }

        Ok(samples)
    }
}
