use anyhow::Result;
use audio_agent::{
    decoder::Decoder,
    device::{AudioChunk, AudioDevice},
    model::WhichModel,
};
use clap::Parser;

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

    /// Print the full DecodingResult structure rather than just the text.
    #[arg(long)]
    verbose: bool,
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

    let device = audio_agent::model::device(args.cpu)?;
    let model =
        audio_agent::model::Model::new(&device, args.model, args.model_id, args.revision, false)?;
    log::debug!("model init done");

    let (sender, receiver): (
        crossbeam_channel::Sender<AudioChunk>,
        crossbeam_channel::Receiver<AudioChunk>,
    ) = crossbeam_channel::unbounded();

    // capture audio
    std::thread::spawn(move || {
        let mut audio = AudioDevice::new("default").unwrap();
        audio.capture(move |msg| {
            sender.send(msg).unwrap();
        })
    });

    let mut dc = Decoder::new(
        model,
        args.seed,
        &device,
        args.task,
        args.timestamps,
        args.verbose,
    )?;
    // loop to process the audio data forever (until the user stops the program)
    for i in 0.. {
        log::debug!("try to receive accumulated data ...");
        let msg = receiver.recv().unwrap();
        log::debug!("decoding data len {}", msg.payload().len());

        let mel = dc.pcm_to_mel(msg.payload())?;

        // on the first n iteration, we detect the language and set the language token.
        if i <= 5 {
            dc.detect_language(&mel, &args.language);
        }

        dc.run(&mel, Some(msg.time_window()))?;
        dc.reset_kv_cache();
    }

    Ok(())
}
