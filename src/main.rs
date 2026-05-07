use burn::backend::NdArray;
use burn::data::dataset::vision::{MnistDataset, MnistItem};
use burn::data::dataset::Dataset;

type Backend = NdArray;

fn display_image(item: &MnistItem) {
    println!("Label: {}", item.label);
    for row in &item.image {
        for &pixel in row {
            // Map pixel brightness to ASCII characters
            let ch = if pixel < 50.0 {
                ' '
            } else if pixel < 150.0 {
                '.'
            } else {
                '#'
            };
            print!("{ch}{ch}"); // print twice so it looks square in the terminal
        }
        println!();
    }
}

fn main() {
    println!("Loading MNIST dataset...");
    let train = MnistDataset::train();
    let test = MnistDataset::test();

    println!("Training samples : {}", train.len());
    println!("Test samples     : {}", test.len());

    // Inspect the very first training image
    println!("\n--- First training image ---");
    let first = train.get(0).expect("Dataset is empty");
    display_image(&first);

    // Show a few labels to get a feel for the data
    println!("\nFirst 10 labels:");
    for i in 0..10 {
        let item = train.get(i).unwrap();
        print!("{} ", item.label);
    }
    println!();
}
