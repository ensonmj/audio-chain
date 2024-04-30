use candle_core::{Device, IndexOp, Tensor, D};
use candle_transformers::models::whisper;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use tracing::debug;

use crate::error::ModelError;
type Result<T> = std::result::Result<T, ModelError>;

const LANGUAGES: [(&str, &str); 99] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum WhichModel {
    Tiny,
    #[value(name = "tiny.en")]
    TinyEn,
    Base,
    #[value(name = "base.en")]
    BaseEn,
    Small,
    #[value(name = "small.en")]
    SmallEn,
    Medium,
    #[value(name = "medium.en")]
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    #[value(name = "distil-medium.en")]
    DistilMediumEn,
    #[value(name = "distil-large-v2")]
    DistilLargeV2,
    #[value(name = "distil-large-v3")]
    DistilLargeV3,
}

impl WhichModel {
    fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::LargeV3
            | Self::DistilLargeV2
            | Self::DistilLargeV3 => true,
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn | Self::DistilMediumEn => {
                false
            }
        }
    }

    fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
            Self::DistilLargeV3 => ("distil-whisper/distil-large-v3", "main"),
        }
    }
}

pub fn device(cpu: bool) -> Result<Device> {
    // Ok(Device::Cpu)
    if cpu {
        Ok(Device::Cpu)
    } else if candle_core::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle_core::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

#[derive(Debug, Clone)]
enum ModelKind {
    Normal(whisper::model::Whisper),
    Quantized(whisper::quantized_model::Whisper),
}

#[derive(Debug, Clone)]
pub struct Model {
    which: WhichModel,
    device: Device,
    config: whisper::Config,
    tokenizer: Tokenizer,
    kind: ModelKind,
}

// Maybe we should use some traits rather than doing the dispatch for all these.
impl Model {
    pub fn new(
        device: &Device,
        which: WhichModel,
        model_id: Option<String>,
        revision: Option<String>,
        quantized: bool,
    ) -> Result<Self> {
        let (default_model, default_revision) = if quantized {
            ("lmz/candle-whisper", "main")
        } else {
            which.model_and_revision()
        };
        let default_model = default_model.to_string();
        let default_revision = default_revision.to_string();
        let (model_id, revision) = match (model_id, revision) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };
        debug!("try to get model {}/{} ...", model_id, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let repo = Api::new()?.repo(Repo::with_revision(model_id, RepoType::Model, revision));
            let (config, tokenizer, model) = if quantized {
                let ext = match which {
                    WhichModel::TinyEn => "tiny-en",
                    WhichModel::Tiny => "tiny",
                    _ => unimplemented!("no quantized support for {:?}", which),
                };
                (
                    repo.get(&format!("config-{ext}.json"))?,
                    repo.get(&format!("tokenizer-{ext}.json"))?,
                    repo.get(&format!("model-{ext}-q80.gguf"))?,
                )
            } else {
                let config = repo.get("config.json")?;
                let tokenizer = repo.get("tokenizer.json")?;
                let model = repo.get("model.safetensors")?;
                (config, tokenizer, model)
            };
            (config, tokenizer, model)
        };

        let config: whisper::Config =
            serde_json::from_str(&std::fs::read_to_string(config_filename)?)
                .map_err(|e| ModelError::Msg(format!("{e}").into()))?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename)?;
        let kind = if quantized {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &weights_filename,
                device,
            )?;
            ModelKind::Quantized(whisper::quantized_model::Whisper::load(
                &vb,
                config.clone(),
            )?)
        } else {
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[weights_filename],
                    whisper::DTYPE,
                    device,
                )?
            };
            ModelKind::Normal(whisper::model::Whisper::load(&vb, config.clone())?)
        };
        Ok(Self {
            which,
            device: device.clone(),
            config,
            tokenizer,
            kind,
        })
    }

    /// Returns the token id for the selected language.
    pub fn detect_language(&mut self, mel: &Tensor) -> candle_core::Result<u32> {
        let (_bsize, _, seq_len) = mel.dims3()?;
        let mel = mel.narrow(
            2,
            0,
            usize::min(seq_len, self.config().max_source_positions),
        )?;
        let device = mel.device();
        let language_token_ids = LANGUAGES
            .iter()
            .map(|(t, _)| self.token_id(&format!("<|{t}|>")))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let sot_token = self.token_id(whisper::SOT_TOKEN)?;
        let audio_features = self.encoder_forward(&mel, true)?;
        let tokens = Tensor::new(&[[sot_token]], device)?;
        let language_token_ids = Tensor::new(language_token_ids.as_slice(), device)?;
        let ys = self.decoder_forward(&tokens, &audio_features, true)?;
        let logits = self.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
        let logits = logits.index_select(&language_token_ids, 0)?;
        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
        let probs = probs.to_vec1::<f32>()?;
        let mut probs = LANGUAGES.iter().zip(probs.iter()).collect::<Vec<_>>();
        probs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
        for ((_, language), p) in probs.iter().take(5) {
            println!("{language}: {p}")
        }
        let language = self.token_id(&format!("<|{}|>", probs[0].0 .0))?;
        Ok(language)
    }

    pub fn is_multilingual(&self) -> bool {
        self.which.is_multilingual()
    }

    pub fn config(&self) -> &whisper::Config {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn token_id(&self, token: &str) -> candle_core::Result<u32> {
        match self.tokenizer.token_to_id(token) {
            None => candle_core::bail!("no token-id for {token}"),
            Some(id) => Ok(id),
        }
    }

    pub fn token_decode(
        &self,
        tokens_to_decode: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String> {
        self.tokenizer
            .decode(tokens_to_decode, skip_special_tokens)
            .map_err(ModelError::Msg)
    }

    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match &mut self.kind {
            ModelKind::Normal(m) => m.encoder.forward(x, flush),
            ModelKind::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle_core::Result<Tensor> {
        match &mut self.kind {
            ModelKind::Normal(m) => m.decoder.forward(x, xa, flush),
            ModelKind::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match &self.kind {
            ModelKind::Normal(m) => m.decoder.final_linear(x),
            ModelKind::Quantized(m) => m.decoder.final_linear(x),
        }
    }

    pub fn reset_kv_cache(&mut self) {
        match &mut self.kind {
            ModelKind::Normal(m) => m.reset_kv_cache(),
            ModelKind::Quantized(m) => m.reset_kv_cache(),
        }
    }
}
