//! Error types for vit-core.

use std::path::PathBuf;

/// Result type alias using vit-core Error.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur when working with ViT configurations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Failed to read config file.
    #[error("failed to read config file '{path}': {source}")]
    ReadFile {
        path: PathBuf,
        source: std::io::Error,
    },

    /// Failed to parse YAML.
    #[error("failed to parse YAML: {0}")]
    ParseYaml(#[from] serde_yaml::Error),

    /// Invalid configuration value.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Missing required field.
    #[error("missing required field: {0}")]
    MissingField(String),

    /// Invalid activation function.
    #[error("invalid activation function: {0}")]
    InvalidActivation(String),

    /// Invalid position encoding.
    #[error("invalid position encoding: {0}")]
    InvalidPositionEncoding(String),

    /// Invalid dtype.
    #[error("invalid dtype: {0}")]
    InvalidDType(String),

    /// Validation error.
    #[error("validation error: {0}")]
    Validation(String),
}
