mod data;
mod model;
mod training;

use burn::backend::{Autodiff, NdArray};
use burn::backend::ndarray::NdArrayDevice;
use burn::optim::AdamConfig;
use model::ModelConfig;
use training::{TrainingConfig, train};

type Backend = Autodiff<NdArray>;

fn main() {
    let device = NdArrayDevice::Cpu;
    let config = TrainingConfig::new(ModelConfig::new(), AdamConfig::new());
    train::<Backend>(config, device);
}
