use std::time::{Duration, Instant};

use anyhow::Result;
use audio_agent::{decoder::Decoder, model::WhichModel};
use candle_transformers::models::whisper;
use clap::Parser;
use cpal::traits::{DeviceTrait, StreamTrait};

const MAX_TIME_FRAME: usize = 15; // in seconds
const TIME_FRAME: usize = 1; // in seconds
const TIMEOUT_TIME_FRAME: u64 = 2 * TIME_FRAME as u64;

struct Msg {
    pub start: f64,
    pub end: f64,
    pub payload: Vec<f32>,
}

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

    let config = model.config();

    let (sender, receiver): (
        crossbeam_channel::Sender<Msg>,
        crossbeam_channel::Receiver<Msg>,
    ) = crossbeam_channel::unbounded();

    // capture audio
    std::thread::spawn(move || {
        // WASAPI(windows) doesn't perform resampling in shared mode so the application must match the device's sample rate
        // ie. we can't capture in whisper::SAMPLE_RATE(16kHz) directly.
        let (device, config) = audio_agent::output_device_and_config!().unwrap();
        // just use one channel for whisper
        let step = config.channels() as usize;
        // downsample for whisper
        let step = step * config.sample_rate().0 as usize / whisper::SAMPLE_RATE;

        let (tx, rx): (
            crossbeam_channel::Sender<Vec<f32>>,
            crossbeam_channel::Receiver<Vec<f32>>,
        ) = crossbeam_channel::unbounded();
        let stream = device
            .build_input_stream(
                &config.config(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let processed: Vec<_> = data.iter().step_by(step).copied().collect();
                    tx.send(processed).unwrap();
                },
                move |err| {
                    log::error!("an error occurred on stream: {}", err);
                },
                None,
            )
            .expect("failed to build audio input stream");

        stream.play().unwrap();

        let mut pcm_data = Vec::<f32>::with_capacity(whisper::N_SAMPLES);
        let mut total_accumulated_len = 0;
        let mut window_accumulated_len = 0;

        let mut start = 0.0;
        let listen_start = Instant::now();
        loop {
            let data = match rx.recv_timeout(Duration::from_secs(TIMEOUT_TIME_FRAME)) {
                Err(_) => {
                    if total_accumulated_len > whisper::HOP_LENGTH {
                        // send data and truncate
                        let end = listen_start.elapsed().as_millis() as f64 / 1000.0;
                        let msg = Msg {
                            start,
                            end,
                            payload: pcm_data.clone(),
                        };
                        sender.send(msg).unwrap();

                        pcm_data.clear();
                        total_accumulated_len = 0;
                        window_accumulated_len = 0;
                        start = listen_start.elapsed().as_millis() as f64 / 1000.0;
                        log::debug!("reset accumulated data before {:.1}s", start);
                    }
                    continue;
                }
                Ok(data) => data,
            };

            total_accumulated_len += data.len();
            window_accumulated_len += data.len();
            if total_accumulated_len > whisper::SAMPLE_RATE * MAX_TIME_FRAME {
                pcm_data.clear();
                total_accumulated_len = data.len();
                window_accumulated_len = data.len();
                start = listen_start.elapsed().as_millis() as f64 / 1000.0;
                log::debug!("reset accumulated data before {:.1}s", start);
            }
            pcm_data.extend(data);
            log::debug!("total accumulate data len {}", total_accumulated_len);

            if window_accumulated_len >= whisper::SAMPLE_RATE * TIME_FRAME {
                // send data but not truncate
                let end = listen_start.elapsed().as_millis() as f64 / 1000.0;
                let msg = Msg {
                    start,
                    end,
                    payload: pcm_data.clone(),
                };
                sender.send(msg).unwrap();
                window_accumulated_len = 0;
            }
        }
    });

    let mel_filters = audio_agent::filters::prepare_mel_filters(config.num_mel_bins);
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
        log::debug!("decoding data len {}", msg.payload.len());

        let mel = dc.pcm_to_mel(&msg.payload, &mel_filters)?;

        // on the first n iteration, we detect the language and set the language token.
        if i <= 5 {
            dc.detect_language(&mel, &args.language);
        }

        dc.run(&mel, Some((msg.start, msg.end)))?;
        dc.reset_kv_cache();
    }

    Ok(())
}
