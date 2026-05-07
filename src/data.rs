use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    tensor::{backend::Backend, Int, Tensor, TensorData},
};

#[derive(Clone)]
pub struct MnistBatcher;

#[derive(Debug, Clone)]
pub struct MnistBatch<B: Backend> {
    /// Shape: [batch_size, 28, 28] — a stack of images
    pub images: Tensor<B, 3>,
    /// Shape: [batch_size] — one label per image
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| {
                // Flatten [[f32; 28]; 28] → Vec<f32> and normalize 0–255 → 0.0–1.0
                let flat: Vec<f32> = item.image
                    .iter()
                    .flatten()
                    .map(|&p| p / 255.0)
                    .collect();
                // Shape [1, 28, 28]: the leading 1 is so cat() can stack along dim 0
                Tensor::<B, 3>::from_data(TensorData::new(flat, [1, 28, 28]), device)
            })
            .collect::<Vec<_>>();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    TensorData::new(vec![item.label as i64], [1]),
                    device,
                )
            })
            .collect::<Vec<_>>();

        MnistBatch {
            // cat along dim 0 turns N × [1, 28, 28] into [N, 28, 28]
            images: Tensor::cat(images, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}
