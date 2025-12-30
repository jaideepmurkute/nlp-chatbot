# Generative AI Chatbot & Fine-Tuning Pipeline

**An end-to-end system for fine-tuning Large Language Models (LLMs) and serving them via a context-aware Flask application.**

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-green?logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![LoRA](https://img.shields.io/badge/Fine--Tuning-LoRA%20%2F%20QLoRA-purple)
![Quantization](https://img.shields.io/badge/Quantization-4--bit-yellow)
![Architecture](https://img.shields.io/badge/Design%20Pattern-lightgrey)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)

## 📌 Project Overview

This project implements a complete lifecycle for a conversational AI agent, covering both the **model training** and **inference serving** stages. It connects a custom fine-tuning pipeline with a backend application designed to manage LLM context windows efficiently.

The system is split into two core modules:

1.  **Finetuning_Job**: A pipeline for training causal LLMs (e.g., Qwen, Llama, DialoGPT) using memory-efficient techniques like **QLoRA** and **Gradient Checkpointing**.
2.  **ChatBot_Flask**: A containerized Flask application featuring a modular **Context Management** system that dynamically processes conversation history to fit within model constraints.

---

## 🚀 Implementation Highlights

This repository demonstrates the practical application of the following engineering concepts:

### 🧠 LLM & Data Engineering
- **Efficient Fine-Tuning**: Implements **QLoRA (4-bit)** and **LoRA** (Low-Rank Adaptation) adapters to enable fine-tuning of large models on consumer-grade hardware.
- **Training Optimization**: Uses **Gradient Accumulation** and **Gradient Checkpointing** to handle memory constraints effectively.
- **Data Processing**: Custom `Dataset` implementation that standardizes disparate data formats into modern **Instruction Tuning** templates (System/User/Assistant).

### 💻 Software Architecture
- **Design Patterns**: 
    - **Strategy Pattern**: Decouples memory management logic from the core chatbot application, allowing for modular switching between different history retention strategies (e.g., Token Truncation vs. Summarization).
    - **Singleton Pattern**: Manages the Model and Tokenizer instances to ensure efficient resource usage during API requests.
- **Containerization**: Application is fully Dockerized to ensure consistent runtime environments.

---

## 📂 System Architecture

The project is organized into two distinct workflows:

### Phase 1: The Finetuning Job (`/Finetuning_Job`)
*Objective: Adapt a base model to specific dataset requirements.*
- **Input**: CSV datasets (Instruction-Response pairs).
- **Process**: 
    - Loads Base Model (e.g., `Qwen2.5-0.5B`).
    - Applies 4-bit Quantization (bitsandbytes).
    - Attaches LoRA adapters.
    - Trains using Causal LM loss.
- **Output**: Saved Adapter weights in `/model_store`.

### Phase 2: The Application (`/ChatBot_Flask`)
*Objective: Serve the model with stateful interaction.*
- **Input**: User text via REST API.
- **Context Manager**: 
    - Calculates available tokens.
    - **Smart Truncation**: Dynamically prunes history, preserving the System Prompt and the most recent interactions while discarding the oldest turns.
- **Generation**: Runs inference (Beam Search / Sampling).
- **Storage**: Logs sessions and conversation history to JSON/Pickle.

---

## 🛠️ Project Structure

```bash
NLP Chatbot/
├── Finetuning_Job/              # MODULE 1: Training
│   ├── src/
│   │   ├── config.py            # Hyperparams (QLoRA, LR, Epochs)
│   │   ├── main.py              # Main training loop
│   │   ├── inference.py         # Inference & Testing script
│   │   └── dataset.py           # Custom Pytorch Dataset
│   └── data/                    # Training datasets
│
├── ChatBot_Flask/               # MODULE 2: Serving
│   ├── src/
│   │   ├── app.py               # Flask Entrypoint
│   │   ├── chatbot.py           # Core - API routes, inference & session management
│   │   ├── model_singleton.py   # Model Loading - Singleton Pattern
│   │   ├── history_manager.py   # Context Management - Strategy Pattern
│   │   ├── CFG.py               # Configuration file for Flask app
│   │   └── templates/           # HTML Frontend
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── README.md                    # Project Documentation
```

---

## ⚡ Quick Start Guide

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized run)
- GPU with CUDA (Recommended for training)

### 1. Installation
Clone the repo and set up a unified virtual environment (or separate ones for each module if preferred).

```bash
git clone https://github.com/yourusername/nlp-chatbot.git
cd "NLP Chatbot"
pip install -r requirements.txt
```

### 2. Fine-Tuning a Model (Optional)
If you do not want to fine-tune, you can skip this and set a standard generic model name (like `Qwen/Qwen2.5-0.5B-Instruct`) in the Chatbot config.

1.  **Prepare Data**: Place your CSV in `Finetuning_Job/data/`.
2.  **Configure**: Edit `Finetuning_Job/src/config.py` (Set `use_4bit=True` if you have a GPU).
3.  **Train**:
    ```bash
    cd Finetuning_Job/src
    python main.py
    ```
    *Artifacts will be saved to `Finetuning_Job/model_store`.*

### 3. Running the Chatbot
You can run the chatbot locally or via Docker.

**Option A: Local Python**
1.  Configure `ChatBot_Flask/src/CFG.py`:
    *   Set `model_name` to your specific HuggingFace model or your local fine-tuned path.
2.  Run:
    ```bash
    cd ChatBot_Flask/src
    python app.py
    ```

**Option B: Docker (Recommended)**
1.  Navigate to the Flask folder:
    ```bash
    cd ChatBot_Flask
    ```
2.  Build and Run:
    ```bash
    docker-compose up --build
    ```
3.  Access the UI at `http://localhost:5100`.

---

## 🔮 Future Roadmap

- [ ] **Summarization Strategy**: Implement a new `HistoryManager` strategy that summarizes old context instead of truncating it.
- [ ] **RAG Integration**: Connect the Chatbot to a Vector Database for retrieval-augmented generation.
- [ ] **Frontend Upgrade**: Port the simple HTML templates to a React/Next.js interface.

## 📜 License
This project is licensed under the MIT License.
