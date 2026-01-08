//! CLI tool for ViT model validation, summarization, and inference.

mod commands;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "vit")]
#[command(author, version, about = "ViT model CLI for validation, summarization, and inference")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate a ViT configuration file
    Validate {
        /// Path to the YAML configuration file
        config: PathBuf,

        /// Output format (text or json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Summarize a ViT model from its configuration
    Summarize {
        /// Path to the YAML configuration file
        config: PathBuf,

        /// Output format (text or json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Run inference on a compiled model (requires vit-ffi)
    Infer {
        /// Path to the compiled .pt2 model
        #[arg(long)]
        model: PathBuf,

        /// Path to the YAML configuration file
        #[arg(long)]
        config: PathBuf,

        /// Device to run on (cpu, cuda:0, etc.)
        #[arg(long, default_value = "cpu")]
        device: String,

        /// Data type (float32, bfloat16)
        #[arg(long, default_value = "float32")]
        dtype: String,

        /// Input shape as comma-separated values (e.g., "1,3,224,224")
        #[arg(long)]
        shape: String,

        /// Number of warmup iterations
        #[arg(long, default_value = "3")]
        warmup: usize,

        /// Number of timed iterations
        #[arg(long, default_value = "10")]
        iterations: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Validate { config, format } => {
            commands::validate::run(&config, &format)
        }
        Commands::Summarize { config, format } => {
            commands::summarize::run(&config, &format)
        }
        Commands::Infer {
            model,
            config,
            device,
            dtype,
            shape,
            warmup,
            iterations,
        } => {
            commands::infer::run(
                &model,
                &config,
                &device,
                &dtype,
                &shape,
                warmup,
                iterations,
            )
        }
    }
}
