# Transformer from Scratch

This project implements a basic **Encoder–Decoder Transformer** from scratch in **TensorFlow / Keras**.  
It demonstrates the core building blocks of the Transformer architecture, including:

- **Text vectorization**
- **Positional embeddings**
- **Scaled dot-product attention**
- **Multi-head attention**
- **Encoder and decoder blocks**
- **Causal masking**
- **Sequence-to-sequence translation**

The notebook trains the model on a simple **English → French translation** task.

## Overview

The goal of this notebook is to help explain how Transformers work internally by building one step by step instead of using a prebuilt library model.

### What the notebook does

1. Loads and prepares an English–French translation dataset
2. Tokenizes and vectorizes source and target text
3. Builds custom Transformer layers from scratch
4. Trains the Transformer on the translation task
5. Tests the model by translating random sentences

## Architecture

```mermaid
flowchart TD
    A[English sentence] --> B[Source TextVectorization]
    C[French sentence with [start] and [end] tokens] --> D[Target TextVectorization]

    B --> E[Positional Embedding]
    D --> F[Positional Embedding]

    E --> G[Transformer Encoder]
    F --> H[Transformer Decoder]

    G --> H
    H --> I[Dropout]
    I --> J[Dense Softmax over Vocabulary]

    J --> K[Predicted French tokens]

    subgraph Encoder
        E --> G
    end

    subgraph Decoder
        F --> H
        H --> I
        I --> J
    end
```

## Model Components

### Positional Embedding
Adds learnable token embeddings and position embeddings so the model can understand word order.

### Scaled Dot-Product Attention
Computes attention scores between query, key, and value tensors.

### Multi-Head Attention
Splits attention across multiple heads so the model can learn different relationships in parallel.

### Transformer Encoder
Applies:
- multi-head self-attention
- feed-forward network
- residual connections
- layer normalization

### Transformer Decoder
Applies:
- causal self-attention
- cross-attention over encoder outputs
- feed-forward network
- residual connections
- layer normalization

## Dataset

The notebook uses an English–French translation CSV dataset and adds:

- `[start]` token at the beginning of each target sentence
- `[end]` token at the end of each target sentence

It also:
- shuffles the dataset
- splits it into **train / validation / test**
- vectorizes text with `TextVectorization`

## Training Setup

- **Embedding dimension:** 512
- **Feed-forward dimension:** 2048
- **Attention heads:** 8
- **Batch size:** 64
- **Maximum source length:** 30
- **Maximum target length:** 31
- **Optimizer:** RMSprop
- **Loss:** Sparse categorical cross-entropy
- **Metric:** Accuracy
- **Epochs:** 50

### Callbacks
The notebook uses:
- `ReduceLROnPlateau`
- `EarlyStopping`
- `ModelCheckpoint`

## Inference

After training, the notebook decodes sentences token by token using greedy decoding until it reaches `[end]` or the maximum decoding length.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook / JupyterLab

## Notes

The notebook expects the translation dataset to be available at:

```text
/kaggle/input/language-translation-englishfrench/eng_-french.csv
```

If you are not running on Kaggle, update the file path in the notebook before executing it.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [TensorFlow Transformer Tutorial](https://www.tensorflow.org/text/tutorials/transformer)
- GPT-2 attention masking implementation inspiration
- Karpathy's minGPT
- Deep Learning with Python
- NLP with Transformers

## Project Structure

```text
.
├── transformer-from-scratch.ipynb
└── README.md
```

## How to Run

1. Open the notebook in Jupyter or Kaggle.
2. Make sure the dataset path is correct.
3. Run all cells from top to bottom.
4. Train the model.
5. Test translation on random English sentences.
