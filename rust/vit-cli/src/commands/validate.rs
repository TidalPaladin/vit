//! Validate command implementation.

use std::path::Path;

use anyhow::{Context, Result};
use vit_core::ViTConfig;

/// Run the validate command.
pub fn run(config_path: &Path, format: &str) -> Result<()> {
    // Load the config
    let config = ViTConfig::from_yaml(config_path)
        .with_context(|| format!("Failed to load config from {:?}", config_path))?;

    // Validate the config
    config
        .validate()
        .with_context(|| "Configuration validation failed")?;

    // Output result
    match format {
        "json" => {
            let result = serde_json::json!({
                "valid": true,
                "config_path": config_path.display().to_string(),
                "depth": config.depth,
                "hidden_size": config.hidden_size,
                "num_attention_heads": config.num_attention_heads,
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        _ => {
            println!("Config is valid: {:?}", config_path);
            println!("  Depth: {}", config.depth);
            println!("  Hidden Size: {}", config.hidden_size);
            println!("  Attention Heads: {}", config.num_attention_heads);
            println!("  Patch Size: {:?}", config.patch_size);
            println!("  Image Size: {:?}", config.img_size);
        }
    }

    Ok(())
}
