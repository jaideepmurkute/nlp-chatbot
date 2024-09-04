# NLP Chatbot with Flask

NLP-LLM based chatbot built using Flask; with custom context handling.  
The chatbot performs explicit context management to improve performance of the models with small context size. 
It does so by methods like 'dynamic proportion based allocation' and 'summarization plus proportional allocation' methods.  
The application handles user and model interactions through a RESTful API.

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
    ```
    cd src
    ```
    Start the app:
    ```sh
    python app.py
    ```
    OR 
    ```sh
    flask run --host=0.0.0.0 --port=5100
    ```
    OR
    ```sh
    # Using Docker
    docker-compose up --build
    ```
    
3. **[Optional] Customize the chatbot behavior**:
    Update the config dictionary to customize behavior.
    1. `model_name`: Any HuggingFace conversational model name.
    2. `max_tot_input_prop`: Upper ceiling for total input (history + current input)  
        - `max_hist_input_prop`: Maximum proportion of the history tokens within total input.   
        - `min_hist_input_prop`: Minimum proportion of the history tokens within total input.     
        #### (Note: Actual proportaions are updated dynamically after each user input)

<!-- 2. **Access the endpoints**:
    - **Chat Endpoint**: `POST /chat`
        - **Request**: `user_input` (form data)
        - **Response**: JSON with the generated response
    - **Model Info Endpoint**: `GET /model_info`
        - **Response**: JSON with model information -->


## License
This project is licensed under the MIT License.
