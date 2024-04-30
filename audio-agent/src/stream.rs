use crossbeam_channel::{unbounded, Receiver, Sender};
use tracing::debug;

use crate::{
    audio::{AudioChunk, AudioDevice},
    decoder::{self, Segment, Task},
    model::WhichModel,
};

use crate::error::StreamError;
type Result<T> = std::result::Result<T, StreamError>;

pub struct StreamItem {
    pub wave: AudioChunk,
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone)]
pub struct Stream {
    decoder: decoder::Decoder,
}

impl Stream {
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
        let device = crate::model::device(cpu)?;

        let model = crate::model::Model::new(&device, model, model_id, revision, false)?;
        debug!("model init done");

        let mut decoder = decoder::Decoder::new(model, &device, seed, task, timestamps)?;
        if let Some(language) = language {
            decoder.set_language(&language);
        }

        Ok(Self { decoder })
    }

    pub fn text_stream(&self) -> Result<TextStream> {
        // channel as buffer
        let (sender, receiver): (Sender<AudioChunk>, Receiver<AudioChunk>) = unbounded();

        let audio = AudioDevice::output_device("default").unwrap();
        audio.play()?;

        const MAX_TIME_FRAME: usize = 15; // in seconds
        const TIME_FRAME: usize = 1; // in seconds
        let acc = audio.accumulate(TIME_FRAME, MAX_TIME_FRAME);
        std::thread::spawn(move || {
            acc.stream(move |msg| {
                sender.send(msg).unwrap();
            })
        });

        let decoder = self.decoder.clone();
        Ok(TextStream {
            audio,
            decoder,
            receiver,
        })
    }
}

// we must keep `audio` in the iterator to keep the handler of cpal::stream.
pub struct TextStream {
    #[allow(dead_code)]
    audio: AudioDevice,
    decoder: decoder::Decoder,
    receiver: Receiver<AudioChunk>,
}

impl TextStream {
    fn speech_to_text(&mut self, msg: &AudioChunk) -> Result<Vec<Segment>> {
        debug!("decoding data len {}", msg.payload().len());

        let mel = self.decoder.pcm_to_mel(msg.payload())?;
        let res = self.decoder.run(&mel, Some(msg.time_window()))?;

        self.decoder.reset_kv_cache();

        Ok(res)
    }
}

impl Iterator for TextStream {
    type Item = StreamItem;
    fn next(&mut self) -> Option<Self::Item> {
        let wave = self.receiver.recv().unwrap();
        let segments = self.speech_to_text(&wave).unwrap();
        Some(StreamItem { wave, segments })
    }
}
