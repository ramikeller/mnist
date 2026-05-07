mod data;
mod model;
mod training;

use burn::backend::{Autodiff, Wgpu};
use burn::backend::wgpu::WgpuDevice;
use burn::optim::AdamConfig;
use model::ModelConfig;
use training::{TrainingConfig, train};

type Backend = Autodiff<Wgpu>;

fn main() {
    let device = WgpuDevice::default();
    let config = TrainingConfig::new(ModelConfig::new(), AdamConfig::new());
    train::<Backend>(config, device);
}
