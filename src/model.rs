use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{backend::Backend, Tensor},
};

/// Hyperparameters — how big is each layer?
#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "512")]
    pub hidden_size_1: usize,
    #[config(default = "256")]
    pub hidden_size_2: usize,
    #[config(default = "10")]
    pub num_classes: usize,
}

/// The network itself: three learned linear transformations.
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
    /// Takes a batch of images, returns raw scores (logits) for each digit class.
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Flatten [B, 28, 28] → [B, 784] so a Linear layer can process it
        let x = images.reshape([batch_size, height * width]);

        // Each linear() learns a transformation; relu() adds non-linearity
        let x = self.activation.forward(self.linear1.forward(x));
        let x = self.activation.forward(self.linear2.forward(x));

        // Final layer: no activation — raw logits for the loss function
        self.linear3.forward(x)
    }
}
