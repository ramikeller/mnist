use burn::{
    data::dataloader::batcher::Batcher,
    data::dataset::vision::{MnistDataset, MnistItem},
    data::dataset::Dataset,
    prelude::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};

use crate::{
    data::{MnistBatch, MnistBatcher},
    model::ModelConfig,
};

fn display_image(item: &MnistItem) {
    for row in &item.image {
        for &pixel in row {
            let ch = if pixel < 50.0 { ' ' } else if pixel < 150.0 { '.' } else { '#' };
            print!("{ch}{ch}");
        }
        println!();
    }
}

pub fn infer<B: Backend<IntElem = i32>>(device: B::Device) {
    // Reconstruct the model architecture then load the saved weights into it.
    let record = CompactRecorder::new()
        .load("./artifact/checkpoint/model-10".into(), &device)
        .expect("Trained model not found — run training first.");

    let model = ModelConfig::new().init::<B>(&device).load_record(record);

    let dataset = MnistDataset::test();

    // Run predictions on a handful of test images
    for index in 0..5 {
        let item = dataset.get(index).unwrap();

        display_image(&item);

        // Wrap the single item in a batch — the model always expects batched input
        let batch: MnistBatch<B> = MnistBatcher.batch(vec![item.clone()], &device);

        // Forward pass: [1, 28, 28] → [1, 10] logits
        let output = model.forward(batch.images);

        // argmax along the class dimension gives the predicted digit
        let predicted: i32 = output
            .argmax(1)          // [1, 1]
            .flatten::<1>(0, 1) // [1]
            .into_scalar();

        println!("True label : {}", item.label);
        println!("Predicted  : {predicted}");
        println!("{}", if predicted == item.label as i32 { "✓ correct" } else { "✗ wrong" });
        println!();
    }
}
