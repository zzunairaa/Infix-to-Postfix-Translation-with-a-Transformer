# Infix-to-Postfix Translation with a Transformer

A neural sequence-to-sequence project that learns to translate fully parenthesized mathematical expressions from **infix notation** into **postfix (Reverse Polish) notation** using a lightweight **Transformer encoder-decoder** implemented in TensorFlow/Keras.

Instead of applying a hand-written parser at inference time, the model learns the structural transformation directly from synthetically generated examples and produces postfix expressions **autoregressively**.

## Example

```text
Infix:   ((a + b) * c)
Postfix: a b + c *
```

For a differently grouped expression:

```text
Infix:   (a + (b * c))
Postfix: a b c * +
```

Postfix notation encodes the order of operations directly and therefore does not require parentheses.

---

## Project Highlights

- Encoder-decoder **Transformer** built with Keras
- Fully synthetic training data generated from a recursive grammar
- Expressions with maximum syntactic depth **4**
- Six operands: `a, b, c, d, e, f`
- Four binary operators: `+`, `-`, `*`, `/`
- Teacher forcing during training
- Greedy **autoregressive decoding** at inference time
- Padding-aware masked loss and accuracy
- Learned positional embeddings
- On-the-fly batch generation
- Model size below the assignment's **2M parameter** limit
- Trained weights distributed through `gdown`
- Reported mean prefix accuracy: **0.998 ± 0.0047**

---

## Repository Structure

```text
infix-to-postfix-transformer/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── infix2postfix.ipynb
```

The notebook contains the complete workflow: data generation, preprocessing, Transformer construction, training, autoregressive generation, evaluation, visualization, and weight loading.

---

## Problem Formulation

The task is treated as sequence-to-sequence translation.

Given a fully parenthesized infix expression

```text
((a+b)*(c-d))
```

the model should generate

```text
ab+cd-*
```

The input structure is fully specified by parentheses, so the project intentionally does **not** rely on standard operator-precedence or associativity rules.

### Vocabulary

The vocabulary contains:

```text
PAD, SOS, EOS,
(, ), +, -, *, /,
a, b, c, d, e, f,
JUNK
```

Special tokens support padding and autoregressive sequence generation.

---

## Synthetic Data Generation

Training expressions are generated recursively.

At each recursive step, the generator either:

1. returns an identifier, or
2. creates a binary expression of the form

```text
(left_expression operator right_expression)
```

The maximum expression depth is limited to **4**.

The corresponding postfix target is produced deterministically from the generated expression and used as supervision for the neural model.

The notebook also analyzes the generated depth distribution to verify that the synthetic dataset follows the intended expression-generation process.

---

## Model Architecture

The solution uses a compact Transformer encoder-decoder.

| Component | Configuration |
|---|---:|
| Encoder layers | 2 |
| Decoder layers | 2 |
| Token embedding size | 64 |
| Positional embedding | Learned |
| Attention heads | 4 |
| Feed-forward hidden size | 128 |
| Maximum sequence length | 62 |
| Optimizer | Adam |
| Decoding | Greedy autoregressive |
| Beam search | No |

A **shared token embedding** is used by the encoder and decoder.

### Encoder

The encoder receives the complete infix expression and applies repeated blocks of:

```text
Embedding + Positional Encoding
            ↓
Multi-Head Self-Attention
            ↓
Residual Connection + Normalization
            ↓
Feed-Forward Network
            ↓
Residual Connection + Normalization
```

### Decoder

The decoder operates autoregressively and combines:

- masked self-attention over previously generated tokens
- cross-attention over encoder representations
- feed-forward layers
- residual connections and normalization

At generation time, the decoder starts with `SOS` and predicts one token at a time until `EOS` is produced or the maximum sequence length is reached.

---

## Training

The notebook uses **teacher forcing** during training.

For a target sequence such as

```text
a b + c * EOS
```

the decoder input is shifted to the right:

```text
SOS a b + c *
```

### Training setup

- Batch size: **64**
- Steps per epoch: **300**
- Effective generated samples per epoch: **19,200**
- Maximum epochs: **20**
- Early stopping patience: **5**
- Validation data: held-out generated expressions

Training samples are generated dynamically instead of being stored as one fixed dataset.

### Padding-aware optimization

`PAD` tokens do not represent meaningful output symbols, so the notebook uses masked loss and masked accuracy to prevent padded positions from influencing optimization or evaluation.

---

## Autoregressive Inference

Inference follows the actual sequence-generation setting required by the task.

```text
Infix Expression
      ↓
Transformer Encoder
      ↓
SOS
 ↓
Decoder → token₁
          ↓
Decoder → token₂
          ↓
        ...
          ↓
         EOS
```

The decoder never receives the ground-truth postfix sequence during evaluation.

Beam search is not used; generation is greedy.

---

## Evaluation

The main metric is **prefix accuracy**.

For each generated sequence, the metric measures the fraction of the sequence that matches the target continuously from the beginning.

For example:

```text
Target:     a b c * d + /
Prediction: a b c * EOS
```

The prediction receives partial credit for the correct prefix rather than being scored only as a complete failure.

### Evaluation Protocol

The notebook follows the prescribed protocol:

- Generate **30 unseen expressions**
- Evaluate the model on all 30
- Repeat the process for **10 independent rounds**
- Report the mean and standard deviation

### Reported Result

```text
Mean Prefix Accuracy: 0.998
Standard Deviation:    0.0047
```

The notebook also includes qualitative examples comparing:

```text
Infix input
Ground-truth postfix
Model prediction
```

---

## Pretrained Weights

The notebook downloads the trained network parameters from Google Drive using `gdown`.

Google Drive file ID:

```text
1zFA90wzsVMipVmnSF35UF_Qpt9Pdb-1p
```

From Python:

```python
import gdown

file_id = "1zFA90wzsVMipVmnSF35UF_Qpt9Pdb-1p"
url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
output = "transformer_infix2postfix.weights.h5"

gdown.download(url, output, quiet=False)
```

The notebook then reconstructs the architecture and loads the weights with:

```python
model.load_weights("transformer_infix2postfix.weights.h5")
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd infix-to-postfix-transformer
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook notebooks/infix2postfix.ipynb
```

The notebook is also suitable for Google Colab.

---

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook
- gdown

---

## What This Project Demonstrates

The project connects a classical symbolic transformation problem with modern neural sequence modeling. In particular, it demonstrates:

- recursive synthetic-data generation
- sequence tokenization and padding
- teacher forcing
- encoder-decoder architectures
- self-attention and cross-attention
- causal masking
- autoregressive sequence generation
- custom masked metrics
- sequence-level evaluation
- model serialization and reproducible weight loading

It is also a useful small-scale example of how a Transformer can learn a deterministic syntax transformation without being explicitly programmed with the conversion algorithm at inference time.

---

## References

1. Vaswani, A. et al. (2017). **Attention Is All You Need.** NeurIPS.
2. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). **Sequence to Sequence Learning with Neural Networks.** NeurIPS.
3. Keras documentation — `MultiHeadAttention`.
4. TensorFlow documentation — `SparseCategoricalCrossentropy`.
5. Jurafsky, D. & Martin, J. — *Speech and Language Processing*.
6. Aho, A. V. et al. — *Compilers: Principles, Techniques, and Tools*.
7. Dijkstra, E. W. — Shunting Yard Algorithm.

---

## Notes

This repository is an educational deep-learning project. The deterministic infix-to-postfix routine is used to create supervised targets; the trained Transformer itself receives only the infix expression during inference and generates the postfix translation autoregressively.
