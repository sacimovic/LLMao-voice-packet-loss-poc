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

        // Try Nova first, then fall back to Claude Haiku
        let models = vec![
            ("us.amazon.nova-micro-v1:0", json!({
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
            })),
            ("anthropic.claude-3-haiku-20240307-v1:0", json!({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "temperature": 0.3,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
            }))
        ];

        let mut last_error = None;
        
        for (model_id, payload) in models {
            println!("[Bedrock] Attempting with model: {}", model_id);
            println!("[Bedrock] Payload: {}", payload.to_string());

            match self
                .client
                .invoke_model()
                .model_id(model_id)
                .body(Blob::new(payload.to_string().as_bytes()))
                .send()
                .await
            {
                Ok(resp) => {
                    println!("[Bedrock] Request successful with {}!", model_id);
                    return self.parse_response(resp, model_id).await;
                }
                Err(e) => {
                    eprintln!("[Bedrock] {} FAILED: {}", model_id, e);
                    last_error = Some(e);
                    continue;
                }
            }
        }
        
        Err(anyhow::anyhow!("All Bedrock models failed. Last error: {:?}", last_error))
    }

    async fn parse_response(&self, response: aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelOutput, model_id: &str) -> Result<String> {

        println!("[Bedrock] Received response from Bedrock");

        let body_bytes = response.body().as_ref();
        println!("[Bedrock] Response body size: {} bytes", body_bytes.len());

        let result: serde_json::Value =
            serde_json::from_slice(body_bytes).context("Failed to parse Bedrock response")?;

        println!("[Bedrock] Parsed JSON response: {}", result);

        // Try Nova format first, then Claude format
        let repaired = if let Some(text) = result["output"]["message"]["content"][0]["text"].as_str() {
            // Nova format
            text.trim().to_string()
        } else if let Some(text) = result["content"][0]["text"].as_str() {
            // Claude format
            text.trim().to_string()
        } else {
            return Err(anyhow::anyhow!("Could not find text in response from {}", model_id));
        };

        println!("[Bedrock] Repair complete. Result: '{}'", repaired);
        Ok(repaired)
    }
}
