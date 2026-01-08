//! Core types and configuration for ViT inference.
//!
//! This crate provides Rust equivalents of the Python ViT configuration
//! types, enabling config validation and model summarization without
//! requiring the full inference runtime.

mod config;
mod error;

pub use config::{
    Activation, DType, HeadConfig, ModelSummary, PositionEncoding, RopeNormalizeCoords,
    TransposedConv2dHeadConfig, UpsampleHeadConfig, ViTConfig,
};
pub use error::{Error, Result};
