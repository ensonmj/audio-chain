use thiserror::Error;

#[derive(Error, Debug)]
pub enum DeviceError {
    #[error("failed to get audio device: {0}")]
    Device(#[from] cpal::DevicesError),
}

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("{0}")]
    Msg(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("{0}")]
    Candle(#[from] candle_core::Error),
    #[error("{0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("{0}")]
    Msg(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("{0}")]
    Model(#[from] ModelError),
    #[error("{0}")]
    Candle(#[from] candle_core::Error),
}
