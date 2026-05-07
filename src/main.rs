mod data;
mod inference;
mod model;
mod training;

use burn::backend::{Autodiff, Wgpu};
use burn::backend::wgpu::WgpuDevice;
use burn::optim::AdamConfig;
use model::ModelConfig;
use training::{TrainingConfig, train};

// Autodiff<Wgpu> for training (needs gradient tracking).
// Plain Wgpu for inference (forward pass only, no overhead).
type TrainBackend = Autodiff<Wgpu>;
type InferBackend = Wgpu;

fn main() {
    let device = WgpuDevice::default();

    // Comment out whichever you don't need.
    train::<TrainBackend>(
        TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
        device.clone(),
    );
    inference::infer::<InferBackend>(device);
}
