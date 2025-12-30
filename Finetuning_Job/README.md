# Fine-tuning Module

This directory contains the pipeline for training and fine-tuning models.

> **Note**: For the complete project overview, architecture details, and setup guide, please refer to the [Root README](../README.md).

## Directory Contents
- **`src/`**: Contains `main.py` (training loop), `config.py` (hyperparameters), and `dataset.py` (data loading).
- **`data/`**: Storage for input CSV datasets (Supports Instruction/Response format).
- **`model_store/`**: Destination for saved model checkpoints and LoRA adapters.
- **`outputs/`**: Training logs and loss visualization.

## Technical Capabilities
- **QLoRA (4-bit)**
- **Gradient Checkpointing**
- **Instruction Tuning Support**
