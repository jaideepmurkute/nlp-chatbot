# LLM Fine-Tuning & Chatbot Pipeline

**An end-to-end system for fine-tuning Large Language Models and serving them via a context-aware Flask application.**

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-App-green?logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Strategy Pattern](https://img.shields.io/badge/Pattern-Strategy-blue)
![Quantization](https://img.shields.io/badge/Quantization-4--bit-yellow)

## Overview

This repository contains two main components:

1.  **Finetuning_Job**: A pipeline for training causal LLMs (e.g., Qwen, Llama, DialoGPT) using memory-efficient techniques like **QLoRA** and **Gradient Checkpointing**.
2.  **ChatBot_Flask**: A containerized Flask application featuring a custom **Context Management** system that dynamically processes conversation history to fit within model constraints.

The primary focus of this project is to explore efficient fine-tuning techniques (QLoRA) and implement a custom memory management system that handles context windows more logically than standard API wrappers.

---

## Technical Features

### 1. Intelligent Context Management
A custom-built engine (`history_manager.py`) designed to maintain conversation coherence within strict token limits, replacing standard "message list" approaches.
*   **Strategy Pattern**: Decouples memory logic from the application, allowing hot-swappable strategies.
*   **Token-Limited Truncation**: Real-time tokenization ensures the constraints are strictly met by pruning the oldest history while locking the System Prompt.
*   **Recursive Summarization**: Automatically condenses overflowing history into a summary and injects it back into the context, effectively extending the model's memory.

### 2. Efficient Fine-Tuning Pipeline
A robust training module focused on adapting large models with minimal hardware resources.
*   **QLoRA & 4-Bit Quantization**: Enables fine-tuning of 7B+ parameter models on consumer GPUs using `bitsandbytes` and Low-Rank Adapters (LoRA).
*   **Gradient Checkpointing**: Optimizes memory usage during training.
*   **Data Standardization**: Custom dataset classes convert raw CSVs into structured instruction-tuning formats (User/Assistant).

### 3. Software Architecture & MLOps
Built with production engineering principles for reliability and scalability.
*   **Singleton Pattern**: Manages the loading of heavy Model/Tokenizer objects to ensure thread safety and resource efficiency.
*   **Dependency Injection**: Injects dependencies into strategies for better modularity and testing.
*   **Dockerized**: Fully containerized environment ensuring consistency across development and deployment.
*   **Experiment Tracking**: Integration with Weights & Biases (W&B) for monitoring training metrics.

---

## Architecture

### Part 1: Fine-Tuning (`/Finetuning_Job`)
A standalone pipeline to adapting base models.
*   **Data Pipeline**: Converts raw CSV data into instruction-tuning formats (User/Assistant).
*   **Training**: Applies 4-bit quantization (bitsandbytes) and trains LoRA adapters using Causal LM loss.
*   **Output**: Saves adapter weights (small file size) to the `model_store`.

### Part 2: Inference Service (`/ChatBot_Flask`)
A Flask API that manages the user session and generation loop.
*   **Request Flow**: User input -> Context Manager -> Prompt Construction -> Generation.
*   **Context Construction**:
    1.  Check total token count against the model's `max_len`.
    2.  Execute the active Strategy (Truncate or Summarize) to fit the budget.
    3.  Pass the optimized tensor/prompt to the model.
*   **State Management**: Updates session logs asynchronously.

---

## Project Structure

```bash
NLP Chatbot/
├── Finetuning_Job/              # Training Module
│   ├── src/
│   │   ├── config.py            # Hyperparameters
│   │   ├── main.py              # Training script
│   │   ├── dataset.py           # Data processing
│   │   └── ...
│   └── data/                    # Training data
│
├── ChatBot_Flask/               # Serving Module
│   ├── src/
│   │   ├── app.py               # Flask routes
│   │   ├── chatbot.py           # Main logic class
│   │   ├── history_manager.py   # Memory/Context logic
│   │   ├── summarizer.py        # Summarization service
│   │   ├── model_singleton.py   # Model loader
│   │   └── templates/           # Simple UI
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── README.md
```

---

## Setup & Usage

### Prerequisites
*   Python 3.8+
*   NVIDIA GPU (Required for 4-bit training)
*   Docker (Optional)

### 1. Installation
```bash
git clone https://github.com/yourusername/nlp-chatbot.git
cd "NLP Chatbot"
pip install -r requirements.txt
```

### 2. Training (Effective Fine-Tuning)
1.  Add your dataset to `Finetuning_Job/data/`.
2.  Update `Finetuning_Job/src/config.py` (e.g., set `use_4bit=True`).
3.  Run the pipeline:
    ```bash
    python Finetuning_Job/src/main.py
    ```

### 3. Serving
**Using Docker (Recommended)**
```bash
cd ChatBot_Flask
docker-compose up --build
```
The UI will be available at `http://localhost:5100`.

**Local Python**
```bash
# Configure model path in ChatBot_Flask/src/CFG.py first
python ChatBot_Flask/src/app.py
```

---

## Roadmap

*   [x] **Summarization Strategy**: Recursive summary injection for long-term memory.
*   [ ] **RAG Integration**: Vector DB connection for document retrieval.
*   [ ] **Frontend**: Upgrade to React/Next.js.

## License
MIT License.
