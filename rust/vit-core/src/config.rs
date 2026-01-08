//! ViT configuration types mirroring the Python dataclasses.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Activation function types supported by the ViT model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Relu,
    Gelu,
    Silu,
    #[default]
    Srelu,
    Swiglu,
    Geglu,
    Reglu,
    Openswiglu,
}

impl Activation {
    /// Returns true if this is a GLU-style activation (doubles FFN intermediate size).
    pub fn is_glu(&self) -> bool {
        matches!(
            self,
            Activation::Swiglu | Activation::Geglu | Activation::Reglu | Activation::Openswiglu
        )
    }
}

/// Position encoding types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEncoding {
    #[default]
    Rope,
    Fourier,
    Learnable,
    None,
}

/// RoPE coordinate normalization modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RopeNormalizeCoords {
    Min,
    Max,
    #[default]
    Separate,
}

/// Data types for model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Float32,
    Float16,
    #[default]
    Bfloat16,
    Float64,
}

impl DType {
    /// Size of this dtype in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Float16 | DType::Bfloat16 => 2,
            DType::Float32 => 4,
            DType::Float64 => 8,
        }
    }
}

/// Configuration for a simple linear head.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadConfig {
    #[serde(default)]
    pub in_features: Option<usize>,
    #[serde(default)]
    pub out_features: Option<usize>,
    #[serde(default)]
    pub dropout: f32,
}

/// Configuration for a transposed convolution head.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransposedConv2dHeadConfig {
    #[serde(default)]
    pub in_features: Option<usize>,
    #[serde(default)]
    pub out_features: Option<usize>,
    #[serde(default = "default_kernel_size")]
    pub kernel_size: Vec<usize>,
    #[serde(default = "default_stride")]
    pub stride: Vec<usize>,
    #[serde(default = "default_padding")]
    pub padding: Vec<usize>,
    #[serde(default)]
    pub output_padding: Vec<usize>,
    #[serde(default = "default_dilation")]
    pub dilation: Vec<usize>,
    #[serde(default = "default_groups")]
    pub groups: usize,
    #[serde(default)]
    pub dropout: f32,
}

fn default_kernel_size() -> Vec<usize> {
    vec![4, 4]
}
fn default_stride() -> Vec<usize> {
    vec![2, 2]
}
fn default_padding() -> Vec<usize> {
    vec![1, 1]
}
fn default_dilation() -> Vec<usize> {
    vec![1, 1]
}
fn default_groups() -> usize {
    1
}

/// Configuration for an upsampling head.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsampleHeadConfig {
    #[serde(default)]
    pub in_features: Option<usize>,
    #[serde(default)]
    pub out_features: Option<usize>,
    #[serde(default)]
    pub num_upsample: usize,
    #[serde(default)]
    pub dropout: f32,
}

/// Union of all head configuration types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnyHeadConfig {
    Head(HeadConfig),
    TransposedConv2d(TransposedConv2dHeadConfig),
    Upsample(UpsampleHeadConfig),
}

/// Vision Transformer configuration.
///
/// This struct mirrors the Python `ViTConfig` dataclass and supports
/// loading from YAML files (including those with Python object tags).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViTConfig {
    // Inputs
    pub in_channels: usize,
    pub patch_size: Vec<usize>,
    pub img_size: Vec<usize>,

    // Transformer architecture
    pub depth: usize,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub num_attention_heads: usize,

    #[serde(default = "default_dropout")]
    pub hidden_dropout: f32,
    #[serde(default = "default_dropout")]
    pub attention_dropout: f32,
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub mlp_bias: bool,
    #[serde(default)]
    pub activation: Activation,
    #[serde(default)]
    pub drop_path_rate: f32,
    #[serde(default)]
    pub num_register_tokens: usize,
    #[serde(default)]
    pub num_cls_tokens: usize,
    #[serde(default)]
    pub pos_enc: PositionEncoding,
    #[serde(default)]
    pub layer_scale: Option<f32>,
    #[serde(default)]
    pub glu_limit: Option<f32>,
    #[serde(default)]
    pub glu_extra_bias: Option<f32>,

    // RoPE options
    #[serde(default)]
    pub rope_normalize_coords: RopeNormalizeCoords,
    #[serde(default = "default_rope_base")]
    pub rope_base: f32,
    #[serde(default)]
    pub rope_shift_coords: Option<f32>,
    #[serde(default)]
    pub rope_jitter_coords: Option<f32>,
    #[serde(default)]
    pub rope_rescale_coords: Option<f32>,

    // Trainable blocks (not used for inference, but parsed for completeness)
    #[serde(default = "default_true")]
    pub mlp_requires_grad: bool,
    #[serde(default = "default_true")]
    pub self_attention_requires_grad: bool,

    // Dtype
    #[serde(default)]
    pub dtype: DType,

    // Heads
    #[serde(default)]
    pub heads: HashMap<String, serde_yaml::Value>,
}

fn default_dropout() -> f32 {
    0.1
}
fn default_true() -> bool {
    true
}
fn default_rope_base() -> f32 {
    100.0
}

impl ViTConfig {
    /// Load a ViTConfig from a YAML file.
    ///
    /// Handles Python-tagged YAML files by stripping the tag before parsing.
    pub fn from_yaml(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| Error::ReadFile {
            path: path.to_path_buf(),
            source: e,
        })?;

        Self::from_yaml_str(&content)
    }

    /// Parse a ViTConfig from a YAML string.
    pub fn from_yaml_str(content: &str) -> Result<Self> {
        // Strip Python object tags that serde_yaml doesn't understand
        let cleaned = strip_python_tags(content);
        let config: ViTConfig = serde_yaml::from_str(&cleaned)?;
        Ok(config)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // Check patch_size dimensions match img_size
        if self.patch_size.len() != self.img_size.len() {
            return Err(Error::Validation(format!(
                "patch_size dimensions ({}) must match img_size dimensions ({})",
                self.patch_size.len(),
                self.img_size.len()
            )));
        }

        // Check that img_size is divisible by patch_size
        for (img, patch) in self.img_size.iter().zip(self.patch_size.iter()) {
            if img % patch != 0 {
                return Err(Error::Validation(format!(
                    "img_size {} must be divisible by patch_size {}",
                    img, patch
                )));
            }
        }

        // Check hidden_size is divisible by num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(Error::Validation(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }

        // Check depth is positive
        if self.depth == 0 {
            return Err(Error::Validation("depth must be > 0".to_string()));
        }

        Ok(())
    }

    /// Compute the number of tokens after patch embedding.
    pub fn num_patches(&self) -> usize {
        self.img_size
            .iter()
            .zip(self.patch_size.iter())
            .map(|(img, patch)| img / patch)
            .product()
    }

    /// Compute total sequence length including register and CLS tokens.
    pub fn seq_length(&self) -> usize {
        self.num_patches() + self.num_register_tokens + self.num_cls_tokens
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Compute a summary of the model's architecture and parameters.
    pub fn compute_summary(&self) -> ModelSummary {
        let head_dim = self.head_dim();
        let num_patches = self.num_patches();
        let seq_length = self.seq_length();

        // Patch embedding parameters: Conv weight + bias
        let patch_embed_params = self.in_channels
            * self.hidden_size
            * self.patch_size.iter().product::<usize>()
            + self.hidden_size;

        // Per-layer parameters
        let qkv_params = 3 * self.hidden_size * self.hidden_size
            + if self.attention_bias {
                3 * self.hidden_size
            } else {
                0
            };
        let attn_out_params =
            self.hidden_size * self.hidden_size + if self.attention_bias { self.hidden_size } else { 0 };

        let ffn_multiplier = if self.activation.is_glu() { 2 } else { 1 };
        let ffn_params = self.hidden_size * self.ffn_hidden_size * ffn_multiplier
            + self.ffn_hidden_size * self.hidden_size
            + if self.mlp_bias {
                self.ffn_hidden_size * ffn_multiplier + self.hidden_size
            } else {
                0
            };

        // RMSNorm has only gamma (no beta)
        let norm_params = self.hidden_size * 2; // 2 norms per layer (attn + mlp)

        let layer_scale_params = if self.layer_scale.is_some() {
            self.hidden_size * 2
        } else {
            0
        };

        let params_per_layer = qkv_params + attn_out_params + ffn_params + norm_params + layer_scale_params;
        let transformer_params = params_per_layer * self.depth;

        // Final norm
        let final_norm_params = self.hidden_size;

        // Register and CLS tokens
        let token_params =
            (self.num_register_tokens + self.num_cls_tokens) * self.hidden_size;

        // Position encoding (learnable only)
        let pos_enc_params = if matches!(self.pos_enc, PositionEncoding::Learnable) {
            num_patches * self.hidden_size
        } else {
            0
        };

        let total_params =
            patch_embed_params + transformer_params + final_norm_params + token_params + pos_enc_params;

        ModelSummary {
            total_params,
            patch_embed_params,
            transformer_params,
            params_per_layer,
            final_norm_params,
            token_params,
            pos_enc_params,
            depth: self.depth,
            hidden_size: self.hidden_size,
            ffn_hidden_size: self.ffn_hidden_size,
            num_attention_heads: self.num_attention_heads,
            head_dim,
            num_patches,
            seq_length,
            activation: self.activation,
            pos_enc: self.pos_enc,
            dtype: self.dtype,
        }
    }
}

/// Summary of model architecture and parameter counts.
#[derive(Debug, Clone)]
pub struct ModelSummary {
    pub total_params: usize,
    pub patch_embed_params: usize,
    pub transformer_params: usize,
    pub params_per_layer: usize,
    pub final_norm_params: usize,
    pub token_params: usize,
    pub pos_enc_params: usize,
    pub depth: usize,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub num_patches: usize,
    pub seq_length: usize,
    pub activation: Activation,
    pub pos_enc: PositionEncoding,
    pub dtype: DType,
}

impl ModelSummary {
    /// Estimated memory for model parameters in bytes.
    pub fn param_memory_bytes(&self) -> usize {
        self.total_params * self.dtype.size_bytes()
    }

    /// Format the summary as a human-readable string.
    pub fn display(&self) -> String {
        let param_mb = self.param_memory_bytes() as f64 / 1_000_000.0;
        let total_m = self.total_params as f64 / 1_000_000.0;

        format!(
            r#"Model Summary
=============
Architecture:
  Depth:              {}
  Hidden Size:        {}
  FFN Hidden Size:    {}
  Attention Heads:    {}
  Head Dimension:     {}
  Activation:         {:?}
  Position Encoding:  {:?}

Sequence:
  Num Patches:        {}
  Sequence Length:    {}

Parameters:
  Total:              {:.2}M
  Patch Embedding:    {}
  Transformer:        {}
  Per Layer:          {}
  Final Norm:         {}
  Tokens:             {}
  Position Encoding:  {}

Memory:
  Parameters:         {:.2} MB ({:?})"#,
            self.depth,
            self.hidden_size,
            self.ffn_hidden_size,
            self.num_attention_heads,
            self.head_dim,
            self.activation,
            self.pos_enc,
            self.num_patches,
            self.seq_length,
            total_m,
            self.patch_embed_params,
            self.transformer_params,
            self.params_per_layer,
            self.final_norm_params,
            self.token_params,
            self.pos_enc_params,
            param_mb,
            self.dtype,
        )
    }
}

/// Strip Python object tags from YAML content.
///
/// Converts lines like:
/// `!!python/object:vit.vit.ViTConfig`
/// to empty lines or just removes the tag portion.
fn strip_python_tags(content: &str) -> String {
    let mut result = String::with_capacity(content.len());
    for line in content.lines() {
        let cleaned = if let Some(idx) = line.find("!!python/object:") {
            // Find where the tag ends (next whitespace or end of line)
            let tag_start = idx;
            let rest = &line[idx + "!!python/object:".len()..];
            let tag_end = rest
                .find(|c: char| c.is_whitespace())
                .map(|i| idx + "!!python/object:".len() + i)
                .unwrap_or(line.len());

            // Remove the tag
            let mut cleaned = line[..tag_start].to_string();
            if tag_end < line.len() {
                cleaned.push_str(&line[tag_end..]);
            }
            cleaned
        } else {
            line.to_string()
        };
        result.push_str(&cleaned);
        result.push('\n');
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_config() {
        let yaml = r#"
in_channels: 3
patch_size: [16, 16]
img_size: [224, 224]
depth: 12
hidden_size: 768
ffn_hidden_size: 3072
num_attention_heads: 12
"#;
        let config = ViTConfig::from_yaml_str(yaml).unwrap();
        assert_eq!(config.depth, 12);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_patches(), 196); // (224/16) * (224/16)
    }

    #[test]
    fn test_parse_with_python_tag() {
        let yaml = r#"!!python/object:vit.vit.ViTConfig
in_channels: 3
patch_size: [16, 16]
img_size: [224, 224]
depth: 12
hidden_size: 768
ffn_hidden_size: 3072
num_attention_heads: 12
"#;
        let config = ViTConfig::from_yaml_str(yaml).unwrap();
        assert_eq!(config.depth, 12);
    }

    #[test]
    fn test_validate_mismatched_dims() {
        let yaml = r#"
in_channels: 3
patch_size: [16, 16]
img_size: [224]
depth: 12
hidden_size: 768
ffn_hidden_size: 3072
num_attention_heads: 12
"#;
        let config = ViTConfig::from_yaml_str(yaml).unwrap();
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_indivisible() {
        let yaml = r#"
in_channels: 3
patch_size: [16, 16]
img_size: [225, 224]
depth: 12
hidden_size: 768
ffn_hidden_size: 3072
num_attention_heads: 12
"#;
        let config = ViTConfig::from_yaml_str(yaml).unwrap();
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_summary() {
        let yaml = r#"
in_channels: 3
patch_size: [16, 16]
img_size: [224, 224]
depth: 12
hidden_size: 768
ffn_hidden_size: 3072
num_attention_heads: 12
"#;
        let config = ViTConfig::from_yaml_str(yaml).unwrap();
        let summary = config.compute_summary();
        assert_eq!(summary.depth, 12);
        assert_eq!(summary.head_dim, 64);
        assert_eq!(summary.num_patches, 196);
        assert!(summary.total_params > 0);
    }

    #[test]
    fn test_strip_python_tags() {
        let input = "!!python/object:vit.vit.ViTConfig\nkey: value";
        let output = strip_python_tags(input);
        assert!(!output.contains("!!python/object"));
        assert!(output.contains("key: value"));
    }
}
