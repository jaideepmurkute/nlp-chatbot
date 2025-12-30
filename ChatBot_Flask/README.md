# Chatbot Application Module

This directory contains the Flask-based serving application and Docker configuration.

> **Note**: For the complete project overview, architecture details, and setup guide, please refer to the [Root README](../README.md).

## Directory Contents
- **`src/`**: Application source code.
  - `app.py`: Flask routes.
  - `chatbot.py`: Main logic.
  - `history_manager.py`: Context management strategies.
- **`Dockerfile`** & **`docker-compose.yml`**: Containerization setup.
- **`templates/`**: HTML frontend files.

## Technical Capabilities
- **Dockerized Deployment**
- **Strategy Pattern for Context Management**
- **Dual Pipeline (Instruct/Legacy)**
