# NeuroScribble  
Scratch Neural Network for MNIST Digit Recognition and Visualization

This project implements a **fully connected neural network from scratch using NumPy**, trained on the MNIST dataset to recognize handwritten digits (0–9).  
It also includes:

- A real-time drawing interface with MNIST-style preprocessing
- Multiple visualization tools, including an **interactive, pan-and-zoom neural network graph**
- No deep learning frameworks (no PyTorch, TensorFlow, or Keras)

The goal of this project is **deep understanding**, not abstraction.

---

## 1. Setup and Installation

### Requirements
- Python 3.10+
- Works on Windows, Linux, macOS

### Install all required dependencies

```bash
pip install numpy pygame scipy matplotlib networkx pyvis
```

These are the only external libraries used.  
All neural network logic is implemented manually.

---

## 2. How to Run

### 2.1 Train the Neural Network

Trains a fully connected neural network on **all 60,000 MNIST training images** and evaluates on 10,000 test images.

```bash
python train.py
```

What this does:
- Downloads MNIST (if not present)
- Trains the network using mini-batch gradient descent
- Uses ReLU + Softmax + Cross-Entropy
- Saves trained weights to `weights.npz`

---

### 2.2 Draw and Predict Digits (Interactive)

Launches a drawing canvas where you can draw digits and see predictions.

```bash
python draw.py
```

Controls:
- Mouse drag: draw digit
- Enter: predict digit and show probabilities
- C: clear canvas

The drawing input is **preprocessed to exactly match MNIST statistics**:
- Bounding box detection
- Resizing to 20×20
- Padding to 28×28
- Center-of-mass alignment
- Gaussian blur

This ensures accurate predictions.

---

### 2.3 Visualize Neural Network Structure (Interactive)

Launches an **interactive HTML visualization** of the neural network.

```bash
python visualize.py
```

This generates `network.html` and opens it in your browser.

Features:
- Smooth pan and zoom
- All neurons visible (784 → 128 → 64 → 10)
- Hourglass / funnel layout
- Real weighted connections (sampled for performance)
- No lag, GPU-accelerated rendering

---

This visualization focuses on **behavior**, not structure.

---

## 3. Project Structure

```
NeuroScribble/
│
├── train.py                    # Training loop
├── draw.py                     # Drawing UI + MNIST preprocessing
├── visualize.py    # Interactive network graph (PyVis)
│
├── data.py                     # MNIST loader
├── model.py                    # Neural network definition
├── layers.py                   # Dense, ReLU, Softmax layers
├── loss.py                     # Cross-entropy loss
├── optimizers.py               # Root Mean Square Propagation optimizer
├── utils.py                    # Accuracy, save/load
│
├── weights.npz                 # Trained model weights
└── docs/                       # Visualization samples
```

---

## 4. Neural Network Architecture

The network is a **fully connected (dense) feedforward neural network**:

```
Input Layer:   784 neurons  (28 × 28 pixels)
Hidden Layer 1: 128 neurons (ReLU)
Hidden Layer 2:  64 neurons (ReLU)
Output Layer:   10 neurons  (Softmax)
```

This forms a **progressive compression funnel**:

```
784 → 128 → 64 → 10
```

Each neuron in a layer is connected to **every neuron in the next layer**.

---

## 5. Forward Pass (Mathematics)

### 5.1 Dense Layer

For a layer with input vector `x`:

```
z = xW + b
```

Where:
- `W` is the weight matrix
- `b` is the bias vector
- `z` is the pre-activation output

---

### 5.2 ReLU Activation

Applied to hidden layers:

```
ReLU(z) = max(0, z)
```

This introduces **non-linearity**, allowing the network to learn complex decision boundaries.

Derivative used in backpropagation:

```
dReLU/dz = 1 if z > 0 else 0
```

---

### 5.3 Softmax (Output Layer)

Converts raw logits into probabilities:

```
softmax(z_i) = exp(z_i) / Σ exp(z_j)
```

Numerical stability is ensured by subtracting `max(z)`.

The output is a **probability distribution over digits 0–9**.

---

## 6. Loss Function: Categorical Cross-Entropy

For one sample:

```
L = - Σ y_true * log(y_pred)
```

Where:
- `y_true` is a one-hot encoded label
- `y_pred` is the softmax output

This loss strongly penalizes confident wrong predictions.

---

## 7. Backpropagation (Core Learning Mechanism)

Training follows this loop:

1. Forward pass → compute predictions
2. Compute loss
3. Backward pass → compute gradients
4. Update weights and biases

### 7.1 Gradient at Output Layer

For softmax + cross-entropy, the gradient simplifies to:

```
∂L/∂z = y_pred − y_true
```

---

### 7.2 Gradient Through Dense Layer

For weights:

```
∂L/∂W = xᵀ · ∂L/∂z
```

For biases:

```
∂L/∂b = Σ ∂L/∂z
```

For inputs (to propagate backward):

```
∂L/∂x = ∂L/∂z · Wᵀ
```

---

## 8. Optimization: RMSProp

Instead of a fixed learning rate, RMSProp adapts the step size per parameter.

```
cache = decay * cache + (1 - decay) * gradient²
parameter -= lr * gradient / (sqrt(cache) + ε)
```

This allows:
- Large updates early
- Smaller updates near convergence
- Stable training without oscillation

---

## 9. Training Details

- Dataset: MNIST
- Training samples: 60,000
- Test samples: 10,000
- Batch size: 64
- Epochs: 15
- Optimizer: RMSProp

Each image is seen **once per epoch**, meaning:

```
60,000 × 15 = 900,000 effective training exposures
```

---

## 10. Drawing Input and Preprocessing

Hand-drawn digits differ from MNIST.  
To solve this **distribution mismatch**, the following steps are applied:

1. Remove background noise
2. Detect bounding box of the digit
3. Resize longest side to 20 pixels
4. Pad to 28×28
5. Center using center-of-mass
6. Apply Gaussian blur
7. Normalize pixel intensities

This preprocessing is **critical** for high accuracy.

---

## 11. Visualization System

### 11.1 Structural Visualization (Interactive)

Implemented using:
- NetworkX (graph structure)
- PyVis + vis.js (GPU-accelerated rendering)

Features:
- All neurons visible
- Layer-wise centering (hourglass shape)
- Sampled strongest connections per neuron
- Smooth pan and zoom
- Browser-based rendering

Due to scale, not all 100k+ edges are drawn.  
This is a deliberate and standard design choice.

---

## 12. Notes on Design Decisions

- Dense networks are used intentionally for clarity
- CNNs are avoided to focus on fundamentals
- Visualization samples are stored in `docs/`
- Performance limitations are handled honestly, not hidden

---

## 13. Summary

This project demonstrates:
- Neural networks from first principles
- Full control over forward and backward passes
- Proper deployment preprocessing
- Scalable visualization techniques


