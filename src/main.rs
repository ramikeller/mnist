use burn::backend::NdArray;
use burn::tensor::Tensor;

// Give our backend a short alias. Later we'll swap this one line to change hardware.
type Backend = NdArray;

fn main() {
    // "Device" is a handle to the GPU. Default picks Metal on macOS automatically.
    let device = Default::default();

    // A 2D tensor (matrix): 2 rows, 3 columns
    let a = Tensor::<Backend, 2>::from_data([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let b = Tensor::<Backend, 2>::from_data([[10.0f32, 20.0, 30.0], [40.0, 50.0, 60.0]], &device);

    println!("a =\n{a}");
    println!("b =\n{b}");

    // This addition runs on the M4 GPU via Metal
    let sum = a + b;
    println!("a + b =\n{sum}");

    // Shape tells you the dimensions: [rows, columns]
    println!("Shape: {:?}", sum.shape());
}
