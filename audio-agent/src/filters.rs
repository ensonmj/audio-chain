pub fn prepare_mel_filters(num_mel_bins: usize) -> Vec<f32> {
    let mel_bytes = match num_mel_bins {
        80 => include_bytes!("../melfilters.bytes").as_slice(),
        128 => include_bytes!("../melfilters128.bytes").as_slice(),
        nmel => unimplemented!("unexpected num_mel_bins {nmel}"),
    };

    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    mel_filters
}
