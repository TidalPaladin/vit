//! Summarize command implementation.

use std::path::Path;

use anyhow::{Context, Result};
use vit_core::ViTConfig;

/// Run the summarize command.
pub fn run(config_path: &Path, format: &str) -> Result<()> {
    // Load the config
    let config = ViTConfig::from_yaml(config_path)
        .with_context(|| format!("Failed to load config from {:?}", config_path))?;

    // Validate first
    config
        .validate()
        .with_context(|| "Configuration validation failed")?;

    // Compute summary
    let summary = config.compute_summary();

    // Output result
    match format {
        "json" => {
            let result = serde_json::json!({
                "config_path": config_path.display().to_string(),
                "architecture": {
                    "depth": summary.depth,
                    "hidden_size": summary.hidden_size,
                    "ffn_hidden_size": summary.ffn_hidden_size,
                    "num_attention_heads": summary.num_attention_heads,
                    "head_dim": summary.head_dim,
                    "activation": format!("{:?}", summary.activation),
                    "pos_enc": format!("{:?}", summary.pos_enc),
                },
                "sequence": {
                    "num_patches": summary.num_patches,
                    "seq_length": summary.seq_length,
                },
                "parameters": {
                    "total": summary.total_params,
                    "total_millions": summary.total_params as f64 / 1_000_000.0,
                    "patch_embed": summary.patch_embed_params,
                    "transformer": summary.transformer_params,
                    "per_layer": summary.params_per_layer,
                    "final_norm": summary.final_norm_params,
                    "tokens": summary.token_params,
                    "pos_encoding": summary.pos_enc_params,
                },
                "memory": {
                    "params_bytes": summary.param_memory_bytes(),
                    "params_mb": summary.param_memory_bytes() as f64 / 1_000_000.0,
                    "dtype": format!("{:?}", summary.dtype),
                },
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        _ => {
            println!("{}", summary.display());
        }
    }

    Ok(())
}
