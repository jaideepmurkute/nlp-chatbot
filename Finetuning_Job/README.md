
# Fine-tuning Job

This directory contains the code to fine-tune a causal LLM (e.g., DialoGPT) on a custom dialogue dataset.

## Structure

- `src/`: Source code.
  - `config.py`: Configuration parameters.
  - `dataset.py`: Data loading and processing.
  - `train.py`: Main training loop.
  - `utils.py`: Helper functions.
- `data/`: Data storage. Contains `sample_dataset.csv`.
- `model_store/`: Where the fine-tuned model will be saved.
- `outputs/`: Training logs and loss curves.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Navigate to the `src` directory:
   ```bash
   cd src
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

3. Monitor progress:
   - Loss values are printed to the console.
   - After training, the loss curve is saved to `outputs/loss_curve.png`.
   - The fine-tuned model is saved to `model_store/finetuned_model`.

## Configuration

Modify `src/config.py` to change:
- `model_name`: Base model (default: `microsoft/DialoGPT-small`).
- Training hyperparameters (epochs, batch size, learning rate).
- Data paths.

## Dataset Format

The dataset should be a CSV file with two columns:
- `context`: The conversation history or prompt.
- `response`: The target response.

Example in `data/sample_dataset.csv`.
