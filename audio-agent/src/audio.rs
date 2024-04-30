use std::time::Duration;

use candle_transformers::models::whisper;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    StreamInstant,
};
use crossbeam_channel::{unbounded, Receiver, Sender};
use tracing::{debug, error};

use crate::error::AudioError;
pub type Result<T> = std::result::Result<T, AudioError>;

#[derive(Debug, Clone, Default)]
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

struct InnerChunk {
    capture: StreamInstant,
    payload: Vec<f32>,
}

pub struct AudioDevice {
    stream: cpal::Stream,
    receiver: Receiver<InnerChunk>,
}

impl AudioDevice {
    pub fn output_device(device: &str) -> Result<Self> {
        let (device, config) = output_device_and_config(device)?;
        let (stream, receiver) = Self::create_stream(device, config.config())?;
        Ok(Self { stream, receiver })
    }

    pub fn input_device(device: &str) -> Result<Self> {
        let (device, config) = input_device_and_config(device)?;
        let (stream, receiver) = Self::create_stream(device, config.config())?;
        Ok(Self { stream, receiver })
    }

    fn create_stream(
        device: cpal::Device,
        config: cpal::StreamConfig,
    ) -> Result<(cpal::Stream, Receiver<InnerChunk>)> {
        // WASAPI(windows) doesn't perform resampling in shared mode so the application must match the device's sample rate
        // ie. we can't capture in whisper::SAMPLE_RATE(16kHz) directly, we need to resample.
        //
        // just use one channel for whisper
        let step = config.channels as usize;
        // downsample for whisper
        let step = step * config.sample_rate.0 as usize / whisper::SAMPLE_RATE;

        let (tx, rx): (Sender<InnerChunk>, Receiver<InnerChunk>) = unbounded();
        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], info: &cpal::InputCallbackInfo| {
                // resample to fit mono whisper::SAMPLE_RATE
                let payload: Vec<_> = data.iter().step_by(step).copied().collect();
                let msg = InnerChunk {
                    capture: info.timestamp().capture,
                    payload,
                };
                tx.send(msg).unwrap();
            },
            move |err| {
                error!("an error occurred on stream: {}", err);
            },
            None,
        )?;

        Ok((stream, rx))
    }

    pub fn play(&self) -> Result<()> {
        Ok(self.stream.play()?)
    }

    pub fn pause(&self) -> Result<()> {
        Ok(self.stream.pause()?)
    }

    pub fn accumulate(&self, time_frame: usize, max_time_frame: usize) -> AudioAccumulator {
        AudioAccumulator {
            receiver: self.receiver.clone(),
            time_frame,
            max_time_frame,
            timeout_time_frame: 2 * time_frame as u64,
        }
    }
}

// AudioAccumulator accumulates multiple short audio chunks into a long chunk.
pub struct AudioAccumulator {
    receiver: Receiver<InnerChunk>,
    time_frame: usize,
    max_time_frame: usize,
    timeout_time_frame: u64,
}

impl AudioAccumulator {
    pub fn stream<F>(&self, mut data_cb: F) -> Result<()>
    where
        F: FnMut(AudioChunk) + Send + 'static,
    {
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
        let mut idle_count = 0;
        loop {
            match self
                .receiver
                .recv_timeout(Duration::from_secs(self.timeout_time_frame))
            {
                Ok(inner_msg) => {
                    if play_start.is_none() {
                        play_start = Some(inner_msg.capture)
                    }
                    if inner_msg.payload.iter().sum::<f32>() == 0.0 {
                        idle_count += 1;
                        if idle_count <= 200 {
                            // 200 * 10ms
                            continue;
                        }
                        idle_count = 0;
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
                    // reset idel_count
                    idle_count = 0;

                    total_accumulated_len += inner_msg.payload.len();
                    window_accumulated_len += inner_msg.payload.len();
                    if total_accumulated_len > whisper::SAMPLE_RATE * self.max_time_frame {
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
                    debug!("accumulate data len {}", window_accumulated_len);

                    if window_accumulated_len >= whisper::SAMPLE_RATE * self.time_frame {
                        callback(play_start, start, end, window_data.clone());

                        // reset window_accumulated_len, but not window_data.
                        // then we can accumulate multi msg into the window.
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
        $crate::audio::input_device_and_config("default")
    };
    ($device: expr) => {
        $crate::audio::input_device_and_config($device)
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
        $crate::audio::output_device_and_config("default")
    };
    ($device: expr) => {
        $crate::audio::output_device_and_config($device)
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
