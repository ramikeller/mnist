# MNIST Digit Recognition in Rust + Burn

A handwritten digit classifier built from scratch using the [Burn](https://burn.dev) deep learning framework in Rust. Runs on Apple M4 GPU via Metal (or CPU via NdArray).

**Final accuracy: ~97.6% on the MNIST test set.**

---

## What it does

Trains a Multi-Layer Perceptron (MLP) on the MNIST dataset — 70,000 grayscale 28×28 images of handwritten digits (0–9). After training, loads the saved weights and runs predictions on 4 individual images. For example given this input, predicts:

```
                    ..##              ..
                    ..##              ..##
                    ####              ..##
                  ..##..              ..##
                  ####                ####
                ####..                ####
              ..####                ..####
              ####..                ####..
              ####                ..####
              ####                ######
              ####                ####..
              ######..........########..
                ######################
                    ........    ######
                                ..####
                                ..####
                                ..####
                                ..####
                                ####..
                                ##..


True label : 4
Predicted  : 4
✓ correct
```

---

## Project structure

```
src/
├── main.rs        Entry point — runs training then inference
├── data.rs        MnistBatcher: converts raw images to normalised tensors
├── model.rs       MLP architecture + TrainStep / InferenceStep impls
├── training.rs    TrainingConfig + SupervisedTraining loop
└── inference.rs   Loads saved weights, predicts on single images
```

---

## How to run

**Train and infer (GPU — default):**
```bash
cargo run --release
```

**Train and infer (CPU):**
```bash
cargo run --release -- --cpu
```

The `--cpu` flag switches the backend from Wgpu (Metal/GPU) to NdArray (CPU) at runtime — no code changes needed.

Checkpoints are saved to `./artifact/checkpoint/`.

---

## Architecture

```
Input [B, 784]          (28×28 pixels, flattened and normalised to 0–1)
    ↓  Linear(784 → 512) + ReLU
[B, 512]
    ↓  Linear(512 → 256) + ReLU
[B, 256]
    ↓  Linear(256 → 10)
[B, 10]                 (raw logits, one per digit class)
```

Total parameters: **535,818**

---

## Learning plan

This project was built step by step as a beginner ML tutorial.

### Step 1 — Project Setup
Set up a Cargo project and add Burn as a dependency with the `wgpu`, `train`, `metrics`, and `vision` features.

### Step 2 — Backend & First Tensor
Learned what a backend is (the engine that executes math — CPU vs GPU) and what a tensor is (a multi-dimensional array — the core data structure of ML). Configured the Wgpu backend which automatically uses Metal on macOS.

### Step 3 — The MNIST Dataset
Loaded `MnistDataset::train()` and `MnistDataset::test()`. MNIST has 60,000 training images and 10,000 test images. The train/test split exists so we can measure whether the model has genuinely learned to generalise, not just memorise.

### Step 4 — Data Pipeline (Batcher)
Built `MnistBatcher` to convert raw `MnistItem` structs into batched tensors. Two key operations:
- **Normalisation**: pixel values 0–255 → 0.0–1.0 (neural networks work better with small numbers)
- **Batching**: stacking N images into a `[N, 28, 28]` tensor so the GPU processes them in parallel

### Step 5 — Model Architecture
Defined `ModelConfig` (hyperparameters) and `Model` (the network). Used Burn's `#[derive(Module)]` macro. The `forward()` method flattens each image from `[B, 28, 28]` to `[B, 784]`, passes it through two hidden layers with ReLU activations, then outputs raw logits `[B, 10]`.

No Softmax at the end — CrossEntropyLoss applies it internally, which is more numerically stable.

### Step 6 — Loss Function & Optimizer
- **CrossEntropyLoss**: measures how wrong the model is. Compares the 10 logits against the correct label. A random model scores ~2.3 (= log 10). Lower is better.
- **Adam optimizer**: updates all 535,818 weights after each batch using gradients. Learning rate: `1e-4`.

### Step 7 — Training Loop
Used Burn's `SupervisedTraining` builder with `Learner`. Key concepts:
- **Epoch**: one full pass through all 60,000 training images
- **Backpropagation**: computing gradients — how much each weight contributed to the error
- **`Autodiff<Wgpu>`**: the backend wrapper that tracks the computation graph so gradients can flow backwards

Training for 10 epochs reached **98.3% train accuracy** and **97.6% validation accuracy**.

### Step 8 — Inference
Loaded the saved `model-10.mpk` checkpoint from disk using `CompactRecorder`. Ran predictions on individual test images using plain `Wgpu` (no `Autodiff` wrapper — gradients aren't needed for inference).

---

## Dependencies

- [burn](https://crates.io/crates/burn) 0.20 — deep learning framework
- Rust 1.93+

## Results

| Split      | Accuracy | Loss  |
|------------|----------|-------|
| Train (ep 1)  | 86.2%  | 0.555 |
| Train (ep 10) | 98.3%  | 0.059 |
| Valid (ep 1)  | 92.4%  | 0.261 |
| Valid (ep 10) | 97.6%  | 0.078 |
