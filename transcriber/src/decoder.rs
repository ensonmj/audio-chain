use anyhow::Result;

use audio_agent::{
    decoder::{self, Segment, Task},
    device::{AudioChunk, AudioDevice},
    model::WhichModel,
};
use crossbeam_channel::{unbounded, Receiver, Sender};

pub struct AppMsg {
    pub wave: AudioChunk,
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone)]
pub struct Decoder {
    decoder: decoder::Decoder,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cpu: bool,
        model: WhichModel,
        model_id: Option<String>,
        revision: Option<String>,
        seed: u64,
        task: Option<Task>,
        timestamps: bool,
        language: Option<String>,
    ) -> Result<Self> {
        let device = audio_agent::model::device(cpu)?;

        let model = audio_agent::model::Model::new(&device, model, model_id, revision, false)?;
        log::debug!("model init done");

        let mut decoder = decoder::Decoder::new(model, &device, seed, task, timestamps)?;
        if let Some(language) = language {
            decoder.set_language(&language);
        }

        Ok(Self { decoder })
    }

    pub fn decode<F>(&self, mut data_cb: F) -> Result<(), anyhow::Error>
    where
        F: FnMut(AppMsg) + Send + 'static,
    {
        let (sender, receiver): (Sender<AudioChunk>, Receiver<AudioChunk>) = unbounded();

        let mut audio = AudioDevice::new("default").unwrap();
        std::thread::spawn(move || {
            audio.capture(move |msg| {
                sender.send(msg).unwrap();
            })
        });

        let mut decoder = (*self).clone();
        std::thread::spawn(move || loop {
            let msg = receiver.recv().unwrap();
            let wave = msg.clone();

            match decoder.decode_impl(msg) {
                Err(err) => {
                    log::warn!("{err:?}");
                    continue;
                }
                Ok(segments) => data_cb(AppMsg { wave, segments }),
            };
        });

        Ok(())
    }

    fn decode_impl(&mut self, msg: AudioChunk) -> Result<Vec<Segment>> {
        log::debug!("decoding data len {}", msg.payload().len());

        let mel = self.decoder.pcm_to_mel(msg.payload())?;
        let res = self.decoder.run(&mel, Some(msg.time_window()))?;

        self.decoder.reset_kv_cache();

        Ok(res)
    }
}
