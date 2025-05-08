
# AML Reproducibility Project

This repository aims to reproduce and analyze the findings from the paper:  
**"A Tensor Decomposition Perspective on Second-order RNNs"**

The project focuses on implementing various second-order RNN architectures, including CP-RNN and Tucker-RNN, and evaluating their performance on character-level language modeling tasks using the Penn Treebank (PTB) dataset.

---

## 📁 Repository Structure

- `rnn_variants_*.py`  
  Implementations of baseline RNN models with varying hidden sizes (64, 128, 256, 512).

- `tucker_rnn_*.py`  
  Implementations of Tucker-RNN models with corresponding hidden sizes.

- `A Tensor Decomposition Perspective on Second-order RNNs.pdf`  
  The original paper being reproduced.

- `README.md`  
  Project overview and instructions.

---

## 📊 Implemented Models

- **RNN**  
  Standard Recurrent Neural Network.

- **MIRNN**  
  Multiplicative Integration RNN.

- **2RNN**  
  Second-order RNN with full bilinear interactions.

- **CPRNN**  
  RNN utilizing CP decomposition for parameter efficiency.

- **TuckerRNN**  
  RNN employing Tucker decomposition for enhanced flexibility.

---

## 🧪 Dataset

The models are trained and evaluated on the **Penn Treebank (PTB)** dataset, a standard benchmark for language modeling tasks.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- PyTorch
- NumPy
- Matplotlib

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Viraj-Ajani/AML_Reproducibility_Project.git
   cd AML_Reproducibility_Project
   ```

2. **Install dependencies:**

   ```bash
   pip install torch numpy matplotlib
   ```

---

## 🏃‍♂️ Running the Models

Each script corresponds to a specific model and hidden size. For example, to run the Tucker-RNN with a hidden size of 128:

```bash
python tucker_rnn_128.py
```

The scripts will automatically download the PTB dataset, train the model, and output evaluation metrics.

---

## 📈 Results

After training, the scripts will display the Bits-Per-Character (BPC) metrics for training, validation, and test sets.

---

## 📄 Reference

If you utilize this codebase or find it helpful, please cite the original paper:

> A Tensor Decomposition Perspective on Second-order RNNs

---

## 📬 Contact

For questions or feedback, feel free to reach out to [Viraj Ajani](https://github.com/Viraj-Ajani).
