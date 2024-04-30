use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use candle_transformers::models::whisper as m;
use rand::{distributions::Distribution, SeedableRng};
use tracing::{debug, info, trace};

use crate::model::Model;

use crate::error::DecodeError;
type Result<T> = std::result::Result<T, DecodeError>;

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum Task {
    Transcribe,
    Translate,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DecodingResult {
    pub text: String,
    tokens: Vec<u32>,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
}

#[derive(Debug, Clone)]
pub struct Decoder {
    model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    mel_filters: Vec<f32>,
    language_token: Option<u32>,
}

impl Decoder {
    pub fn new(
        model: Model,
        device: &Device,
        seed: u64,
        task: Option<Task>,
        timestamps: bool,
    ) -> Result<Self> {
        let no_timestamps_token = model.token_id(m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = model.token_id(m::SOT_TOKEN)?;
        let transcribe_token = model.token_id(m::TRANSCRIBE_TOKEN)?;
        let translate_token = model.token_id(m::TRANSLATE_TOKEN)?;
        let eot_token = model.token_id(m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| model.token_id(token).ok());
        let no_speech_token = match no_speech_token {
            None => {
                return Err(DecodeError::Msg(
                    "unable to find any non-speech token".into(),
                ))
            }
            Some(n) => n,
        };
        let mel_filters = Self::prepare_mel_filters(model.config().num_mel_bins);

        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            task,
            timestamps,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            no_timestamps_token,
            mel_filters,
            language_token: None,
        })
    }

    pub fn pcm_to_mel(&self, pcm_data: &[f32]) -> candle_core::Result<Tensor> {
        let config = self.model.config();
        let mel = m::audio::pcm_to_mel(config, pcm_data, &self.mel_filters);
        let mel_len = mel.len();
        Tensor::from_vec(
            mel,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            self.model.device(),
        )
    }

    pub fn from_mel(&self, mel_data: &[f32]) -> candle_core::Result<Tensor> {
        let config = self.model.config();
        let mel_len = mel_data.len();
        Tensor::from_slice(
            mel_data,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            self.model.device(),
        )
    }

    pub fn set_language(&mut self, language: &str) {
        if self.model.is_multilingual() {
            if let Ok(token_id) = self.model.token_id(&format!("<|{language}|>")) {
                self.language_token = Some(token_id);
            }
        }
    }

    // pub fn detect_language(&mut self, mel: &Tensor, language: &Option<String>) {
    //     self.language_token = match (self.model.is_multilingual(), language) {
    //         (true, None) => self.model.detect_language(mel).ok(),
    //         (true, Some(language)) => match self.model.token_id(&format!("<|{language}|>")) {
    //             Ok(token_id) => Some(token_id),
    //             Err(_) => None,
    //         },
    //         // a language cannot be set for non-multilingual models
    //         (false, _) => None,
    //     };
    //     log::debug!("detected language_token: {:?}", self.language_token);
    // }

    pub fn run(&mut self, mel: &Tensor, times: Option<(f64, f64)>) -> Result<Vec<Segment>> {
        if self.model.is_multilingual() && self.language_token.is_none() {
            self.language_token = self.model.detect_language(mel).ok();
        }

        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let start = std::time::Instant::now();
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                debug!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let mut segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            if self.timestamps {
                info!(
                    "{:.3}s -- {:.3}s",
                    segment.start,
                    segment.start + segment.duration,
                );
                let mut tokens_to_decode = vec![];
                let mut prev_timestamp_s = 0f32;
                for &token in segment.dr.tokens.iter() {
                    if token == self.sot_token || token == self.eot_token {
                        continue;
                    }
                    // The no_timestamp_token is the last before the timestamp ones.
                    if token > self.no_timestamps_token {
                        let timestamp_s = (token - self.no_timestamps_token + 1) as f32 / 50.;
                        if !tokens_to_decode.is_empty() {
                            let text = self
                                .model
                                .token_decode(&tokens_to_decode, true)
                                .map_err(DecodeError::Model)?;
                            info!("  {:.3}s-{:.3}s: {}", prev_timestamp_s, timestamp_s, text);
                            tokens_to_decode.clear()
                        }
                        prev_timestamp_s = timestamp_s;
                    } else {
                        tokens_to_decode.push(token)
                    }
                }
                if !tokens_to_decode.is_empty() {
                    let text = self
                        .model
                        .token_decode(&tokens_to_decode, true)
                        .map_err(DecodeError::Model)?;
                    if !text.is_empty() {
                        info!("  {:.3}s-...: {}", prev_timestamp_s, text);
                    }
                    tokens_to_decode.clear()
                }
            } else {
                match times {
                    Some((start, end)) => {
                        segment.start = start;
                        segment.duration = end - start;
                        info!("{:.3}s -- {:.3}s: {}", start, end, segment.dr.text)
                    }
                    None => {
                        info!(
                            "{:.3}s -- {:.3}s: {}",
                            segment.start,
                            segment.start + segment.duration,
                            segment.dr.text,
                        )
                    }
                }
            }
            trace!("{seek}: {segment:?}, in {:?}", start.elapsed());
            segments.push(segment)
        }
        Ok(segments)
    }

    pub fn reset_kv_cache(&mut self) {
        self.model.reset_kv_cache();
    }

    pub fn model(&mut self) -> &mut Model {
        &mut self.model
    }
}

impl Decoder {
    fn prepare_mel_filters(num_mel_bins: usize) -> Vec<f32> {
        let mel_bytes = match num_mel_bins {
            80 => include_bytes!("../melfilters.bytes").as_slice(),
            128 => include_bytes!("../melfilters128.bytes").as_slice(),
            nmel => unimplemented!("unexpected num_mel_bins {nmel}"),
        };

        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        mel_filters
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)
                    .map_err(|e| DecodeError::Msg(e.into()))?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self
            .model
            .token_decode(&tokens, true)
            .map_err(DecodeError::Model)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }
}
