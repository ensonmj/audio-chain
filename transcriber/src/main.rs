use anyhow::Result;
use clap::Parser;

use audio_agent::model::WhichModel;

mod app;
mod decoder;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value = "true")]
    cpu: bool,

    #[arg(long)]
    model_id: Option<String>,

    /// The model to use, check out available models:
    /// https://huggingface.co/models?search=whisper
    #[arg(long)]
    revision: Option<String>,

    /// The model to be used, can be tiny, small, medium.
    #[arg(long, default_value = "tiny.en")]
    model: WhichModel,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Language.
    #[arg(long, default_value = "en")]
    language: Option<String>,

    /// Task, when no task is specified, the input tokens contain only the sot token which can
    /// improve things when in no-timestamp mode.
    #[arg(long)]
    // #[arg(long, value_parser = clap::builder::PossibleValuesParser::new(audio_agent::decoder::Task::VARIANTS).map(|s| s.parse::<audio_agent::decoder::Task>().unwrap()))]
    task: Option<audio_agent::decoder::Task>,

    /// Timestamps mode, this is not fully implemented yet.
    #[arg(long)]
    timestamps: bool,
}

pub fn main() -> Result<()> {
    tracing_log::LogTracer::init()?;

    let args = Args::parse();
    let _guard = if args.tracing {
        use tracing_subscriber::prelude::*;
        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // create app and run it
    let mut app = app::App::new(
        args.cpu,
        args.model,
        args.model_id,
        args.revision,
        args.seed,
        args.task,
        args.timestamps,
        args.language,
    )?;
    app.run_app()
}
