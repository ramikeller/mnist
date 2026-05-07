use std::sync::Arc;

use burn::{
    config::Config,
    data::{
        dataloader::{DataLoader, DataLoaderBuilder},
        dataset::vision::MnistDataset,
    },
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        Learner, SupervisedTraining,
        metric::{AccuracyMetric, LossMetric},
    },
};

use crate::{
    data::MnistBatcher,
    model::{Model, ModelConfig},
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = "10")]
    pub num_epochs: usize,
    #[config(default = "64")]
    pub batch_size: usize,
    #[config(default = "1e-4")]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(config: TrainingConfig, device: B::Device) {
    let model = config.model.init::<B>(&device);
    let optim = config.optimizer.init::<B, Model<B>>();

    // Type annotations tell the compiler which backend each dataloader uses.
    // Training uses B (with autodiff); validation uses B::InnerBackend (no gradients).
    let dataloader_train: Arc<dyn DataLoader<_, _>> = DataLoaderBuilder::<B, _, _>::new(MnistBatcher)
        .batch_size(config.batch_size)
        .shuffle(42)
        .num_workers(4)
        .build(MnistDataset::train());

    let dataloader_valid: Arc<dyn DataLoader<_, _>> =
        DataLoaderBuilder::<B::InnerBackend, _, _>::new(MnistBatcher)
            .batch_size(config.batch_size)
            .num_workers(4)
            .build(MnistDataset::test());

    let learner = Learner::new(model, optim, config.learning_rate);

    SupervisedTraining::new("./artifact", dataloader_train, dataloader_valid)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .launch(learner);
}
