mod data;
mod model;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::vision::MnistDataset;
use burn::data::dataset::Dataset;
use data::{MnistBatch, MnistBatcher};
use model::ModelConfig;

type Backend = NdArray;

fn main() {
    let device = NdArrayDevice::Cpu;

    // --- Data ---
    let dataset = MnistDataset::train();
    let items: Vec<_> = (0..4).map(|i| dataset.get(i).unwrap()).collect();
    let batcher = MnistBatcher;
    let batch: MnistBatch<Backend> = batcher.batch(items, &device);

    println!("Input shape : {:?}", batch.images.shape());

    // --- Model ---
    let model = ModelConfig::new().init::<Backend>(&device);
    println!("{model}");

    // --- Forward pass ---
    let output = model.forward(batch.images);
    println!("Output shape: {:?}", output.shape());
    println!("Output (logits):\n{output}");
}
