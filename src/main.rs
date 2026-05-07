mod data;
mod inference;
mod model;
mod training;

use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::optim::AdamConfig;
use model::ModelConfig;
use training::{TrainingConfig, train};

fn main() {
    let use_cpu = std::env::args().any(|a| a == "--cpu");

    if use_cpu {
        // NdArray<f32, i32> matches the Backend<IntElem = i32> bound in infer().
        type B = NdArray<f32, i32>;
        let device = NdArrayDevice::default();
        train::<Autodiff<B>>(
            TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
            device,
        );
        inference::infer::<B>(device);
    } else {
        type B = Wgpu;
        let device = WgpuDevice::default();
        train::<Autodiff<B>>(
            TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
            device.clone(),
        );
        inference::infer::<B>(device);
    }
}
