mod data;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::vision::MnistDataset;
use burn::data::dataset::Dataset;
use data::{MnistBatch, MnistBatcher};

type Backend = NdArray;

fn main() {
    let device = NdArrayDevice::Cpu;

    let dataset = MnistDataset::train();
    println!("Dataset size: {}", dataset.len());

    // Grab the first 4 items and hand them to the batcher
    let items: Vec<_> = (0..4).map(|i| dataset.get(i).unwrap()).collect();

    let batcher = MnistBatcher;
    let batch: MnistBatch<Backend> = batcher.batch(items, &device);

    println!("Images tensor shape : {:?}", batch.images.shape());
    println!("Targets tensor shape: {:?}", batch.targets.shape());
    println!("Targets (labels)    : {}", batch.targets);
}
