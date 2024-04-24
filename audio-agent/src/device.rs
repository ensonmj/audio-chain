use cpal::traits::{DeviceTrait, HostTrait};

use crate::error::DeviceError;
pub type Result<T> = std::result::Result<T, DeviceError>;

#[macro_export]
macro_rules! input_device_and_config {
    () => {
        $crate::device::input_device_and_config("default")
    };
    ($device: expr) => {
        $crate::device::input_device_and_config($device)
    }; //call function f1 when there's one variable
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
    }; //call function f1 when there's one variable
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
