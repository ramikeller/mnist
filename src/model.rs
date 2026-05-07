use burn::{
    config::Config,
    module::Module,
    nn::{loss::CrossEntropyLossConfig, Linear, LinearConfig, Relu},
    tensor::{Int, Tensor, backend::AutodiffBackend, backend::Backend},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

use crate::data::MnistBatch;

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "512")]
    pub hidden_size_1: usize,
    #[config(default = "256")]
    pub hidden_size_2: usize,
    #[config(default = "10")]
    pub num_classes: usize,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    activation: Relu,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(28 * 28, self.hidden_size_1).init(device),
            linear2: LinearConfig::new(self.hidden_size_1, self.hidden_size_2).init(device),
            linear3: LinearConfig::new(self.hidden_size_2, self.num_classes).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();
        let x = images.reshape([batch_size, height * width]);
        let x = self.activation.forward(self.linear1.forward(x));
        let x = self.activation.forward(self.linear2.forward(x));
        self.linear3.forward(x)
    }

    // Shared by training and validation: forward pass + loss computation.
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());
        ClassificationOutput::new(loss, output, targets)
    }
}

// Training step: forward + backward pass. Runs on the Autodiff-wrapped backend.
impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

// Inference step: forward pass only. Used for validation (no gradient tracking needed).
// The Learner calls this on Model<B::InnerBackend> (the non-autodiff version of the model).
impl<B: Backend> InferenceStep for Model<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
