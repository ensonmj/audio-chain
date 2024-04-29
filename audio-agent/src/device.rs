use std::time::Duration;

use candle_transformers::models::whisper;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    StreamInstant,
};

use crate::error::DeviceError;
pub type Result<T> = std::result::Result<T, DeviceError>;

const MAX_TIME_FRAME: usize = 15; // in seconds
const TIME_FRAME: usize = 1; // in seconds
const TIMEOUT_TIME_FRAME: u64 = 2 * TIME_FRAME as u64;

#[derive(Debug, Clone)]
pub struct AudioChunk {
    start: f64,
    end: f64,
    payload: Vec<f32>,
}

impl AudioChunk {
    pub fn payload(&self) -> &[f32] {
        &self.payload
    }

    pub fn time_window(&self) -> (f64, f64) {
        (self.start, self.end)
    }
}

pub struct AudioDevice {
    device: cpal::Device,
    config: cpal::StreamConfig,
}

impl AudioDevice {
    pub fn new(device: &str) -> Result<Self> {
        let (device, config) = output_device_and_config(device)?;
        Ok(Self {
            device,
            config: config.config(),
        })
    }

    pub fn capture<F>(&mut self, mut data_cb: F) -> Result<()>
    where
        F: FnMut(AudioChunk) + Send + 'static,
    {
        // WASAPI(windows) doesn't perform resampling in shared mode so the application must match the device's sample rate
        // ie. we can't capture in whisper::SAMPLE_RATE(16kHz) directly, we need to resample.

        // just use one channel for whisper
        let step = self.config.channels as usize;
        // downsample for whisper
        let step = step * self.config.sample_rate.0 as usize / whisper::SAMPLE_RATE;

        struct InnerAudioChunk {
            capture: StreamInstant,
            payload: Vec<f32>,
        }
        let (tx, rx): (
            crossbeam_channel::Sender<InnerAudioChunk>,
            crossbeam_channel::Receiver<InnerAudioChunk>,
        ) = crossbeam_channel::unbounded();
        let stream = self.device.build_input_stream(
            &self.config,
            move |data: &[f32], info: &cpal::InputCallbackInfo| {
                // resample to fit mono whisper::SAMPLE_RATE
                let payload: Vec<_> = data.iter().step_by(step).copied().collect();
                let msg = InnerAudioChunk {
                    capture: info.timestamp().capture,
                    payload,
                };
                tx.send(msg).unwrap();
            },
            move |err| {
                log::error!("an error occurred on stream: {}", err);
            },
            None,
        )?;

        stream.play()?;

        let mut callback = move |play_start: Option<StreamInstant>,
                                 start: Option<StreamInstant>,
                                 end: Option<StreamInstant>,
                                 payload: Vec<f32>| {
            let i2f = |instant: Option<StreamInstant>| {
                instant
                    .as_ref()
                    .unwrap()
                    .duration_since(play_start.as_ref().unwrap())
                    .unwrap()
                    .as_millis() as f64
                    / 1000.0
            };

            let start = i2f(start);
            let end = i2f(end);
            let msg = AudioChunk {
                start,
                end,
                payload,
            };
            data_cb(msg);
        };

        let mut play_start = None;
        let mut start = None;
        let mut end = None;
        let mut window_data = Vec::<f32>::with_capacity(whisper::N_SAMPLES);

        let mut total_accumulated_len = 0;
        let mut window_accumulated_len = 0;
        loop {
            match rx.recv_timeout(Duration::from_secs(TIMEOUT_TIME_FRAME)) {
                Ok(inner_msg) => {
                    if play_start.is_none() {
                        play_start = Some(inner_msg.capture)
                    }
                    if inner_msg.payload.iter().sum::<f32>() == 0.0 {
                        // found a long pause or paragraph.
                        if start.is_some()
                            && end.is_some()
                            && window_data.len() > whisper::HOP_LENGTH
                        {
                            callback(play_start, start, end, window_data.clone());

                            start = None;
                            total_accumulated_len = 0;
                            window_accumulated_len = 0;
                            window_data.clear();
                        }
                        continue;
                    }

                    total_accumulated_len += inner_msg.payload.len();
                    window_accumulated_len += inner_msg.payload.len();
                    if total_accumulated_len > whisper::SAMPLE_RATE * MAX_TIME_FRAME {
                        callback(play_start, start, end, window_data.clone());

                        start = None;
                        total_accumulated_len = inner_msg.payload.len();
                        window_accumulated_len = inner_msg.payload.len();
                        window_data.clear();
                    }

                    if start.is_none() {
                        start = Some(inner_msg.capture);
                    }
                    end = Some(inner_msg.capture);

                    window_data.extend(inner_msg.payload);
                    log::debug!("accumulate data len {}", window_accumulated_len);

                    if window_accumulated_len >= whisper::SAMPLE_RATE * TIME_FRAME {
                        callback(play_start, start, end, window_data.clone());

                        // reset window_accumulated_len, but not window_data. then we can accumulate multi msg into the window.
                        window_accumulated_len = 0;
                    }
                }
                Err(_) => {
                    if start.is_some() && end.is_some() && window_data.len() > whisper::HOP_LENGTH {
                        callback(play_start, start, end, window_data.clone());

                        start = None;
                        total_accumulated_len = 0;
                        window_accumulated_len = 0;
                        window_data.clear();
                    }
                }
            };
        }
    }
}

#[macro_export]
macro_rules! input_device_and_config {
    () => {
        $crate::device::input_device_and_config("default")
    };
    ($device: expr) => {
        $crate::device::input_device_and_config($device)
    };
}

pub fn input_device_and_config(
    device: &str,
) -> Result<(cpal::Device, cpal::SupportedStreamConfig)> {
    let host = cpal::default_host();

    let device = if device == "default" {
        host.default_input_device()
    } else {
        host.input_devices()?
            .find(|x| x.name().map(|y| y == device).unwrap_or(false))
    }
    .expect("failed to find input device");

    let config = device
        .default_input_config()
        .expect("Failed to get default input config");

    Ok((device, config))
}

#[macro_export]
macro_rules! output_device_and_config {
    () => {
        $crate::device::output_device_and_config("default")
    };
    ($device: expr) => {
        $crate::device::output_device_and_config($device)
    };
}

pub fn output_device_and_config(
    device: &str,
) -> Result<(cpal::Device, cpal::SupportedStreamConfig)> {
    let host = cpal::default_host();

    let device = if device == "default" {
        host.default_output_device()
    } else {
        host.output_devices()?
            .find(|x| x.name().map(|y| y == device).unwrap_or(false))
    }
    .expect("failed to find input device");

    let config = device
        .default_output_config()
        .expect("Failed to get default input config");

    Ok((device, config))
}
