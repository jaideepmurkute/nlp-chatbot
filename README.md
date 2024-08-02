# NLP Chatbot with Flask

This project is a Natural Language Processing (NLP) based chatbot built using Flask.  
The chatbot can generate responses based on user input and provides model information through a RESTful API.
The chatbot performs explicit context management to improve performance of the models with with small context size. 
It does so by methods like 'dynamic proportion based allocation' and 'summarization plus proportional allocation' methods.  


<!-- ## Features

- **Chat Endpoint**: Accepts user input and returns a generated response.
- **Model Info Endpoint**: Provides information about the chatbot model.
- **Logging**: Logs important events and errors for debugging purposes. -->

## Requirements

- Python 3.x
- Flask
- Other dependencies listed in `requirements.txt`

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/nlp-chatbot-flask.git
    cd nlp-chatbot-flask
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Flask application**:
    ```sh
    python app.py
    ```
    OR 
    ```sh
    flask run --host=0.0.0.0 --port=5100
    ```
    OR
    ```sh
    # Dockerize the application:  
    # Build Image:   
    docker build -t my-flask-chatbot .
    # Run Image:   
    docker run -p 5100:5100 my-flask-chatbot:latest
    ```
    
2. **Set the config dictionary**:

<!-- 2. **Access the endpoints**:
    - **Chat Endpoint**: `POST /chat`
        - **Request**: `user_input` (form data)
        - **Response**: JSON with the generated response
    - **Model Info Endpoint**: `GET /model_info`
        - **Response**: JSON with model information -->


## Project Structure
- `app.py`: Main application file.
- `CFG.py`: Configuration settings for the chatbot.
- `chatbot.py`: Chatbot logic and response generation.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.


## License

This project is licensed under the MIT License.