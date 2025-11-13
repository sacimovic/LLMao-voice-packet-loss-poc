use anyhow::{Context, Result};
use aws_config;
use aws_sdk_bedrockruntime::{primitives::Blob, Client as BedrockClient};
use serde_json::json;

pub struct BedrockRepair {
    client: BedrockClient,
}

impl BedrockRepair {
    pub async fn new() -> Result<Self> {
        let config = aws_config::load_from_env().await;
        let client = BedrockClient::new(&config);
        Ok(Self { client })
    }

    pub async fn repair(&self, asr_text: &str) -> Result<String> {
        println!("[Bedrock] Starting repair for text: '{}'", asr_text);

        let prompt = format!(
            "The following text was transcribed from speech but may contain errors due to packet loss. \
             Please correct any obvious mistakes and return ONLY the corrected text with no explanations:\n\n{}",
            asr_text
        );

        let payload = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "text": prompt
                }]
            }],
            "inferenceConfig": {
                "maxTokens": 200,
                "temperature": 0.3
            }
        });

        println!("[Bedrock] Sending request to model: us.amazon.nova-micro-v1:0");
        println!("[Bedrock] Payload: {}", payload.to_string());

        let response = match self
            .client
            .invoke_model()
            .model_id("us.amazon.nova-micro-v1:0")
            .body(Blob::new(payload.to_string().as_bytes()))
            .send()
            .await
        {
            Ok(resp) => {
                println!("[Bedrock] Request successful!");
                resp
            }
            Err(e) => {
                eprintln!("[Bedrock] REQUEST FAILED: {:?}", e);
                eprintln!("[Bedrock] Error details: {}", e);
                return Err(anyhow::anyhow!("Failed to invoke Bedrock model: {}", e));
            }
        };

        println!("[Bedrock] Received response from Bedrock");

        let body_bytes = response.body().as_ref();
        println!("[Bedrock] Response body size: {} bytes", body_bytes.len());

        let result: serde_json::Value =
            serde_json::from_slice(body_bytes).context("Failed to parse Bedrock response")?;

        println!("[Bedrock] Parsed JSON response: {}", result);

        let repaired = result["output"]["message"]["content"][0]["text"]
            .as_str()
            .context("Missing text in Bedrock response")?
            .trim()
            .to_string();

        println!("[Bedrock] Repair complete. Result: '{}'", repaired);
        Ok(repaired)
    }
}
