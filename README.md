# Flask Chatbot Application with Ngrok Integration

This repository contains the source code for a Flask-based chatbot application that integrates with Ngrok for easy deployment and testing. The application includes functionalities for processing text and audio inputs, converting text to speech, and querying a chatbot for responses.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Testing the Application](#testing-the-application)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.11.9 or higher
- Flask
- SpeechRecognition library
- langchain_community
- langchain
- Ollama
- pyttsx3

## Installation

1. Clone the repository:
bash git clone https://github.com/Abdulwadood39/Darazbot.git cd Darazbot

2. Install the required Python packages:
bash pip install -r requirements.txt


## Running the Application

1. Ensure Ngrok is installed on your system. If not, download and install it from [Ngrok's official website](https://ngrok.com/download).

2. Start the Flask application:

bash curl -fsSL https://ollama.com/install.sh | sh
bash ollama pull mistral 
bash ollama pull nomic-embed-text
bash python3 vector_db.py
bash python3 app.py

3. Open a new terminal window and start Ngrok by specifying the port your Flask application is running on (default is 5001):

bash ./ngrok http 5001


For Windows users, the command might be slightly different, e.g., `ngrok http 5001`.

4. Copy the Ngrok URL displayed in the terminal. This URL is a tunnel to your local Flask application.

## Testing the Application

Access your application through the Ngrok URL. For example, if you have a route `/submit`, you can access it at `http://12345678.ngrok.io/submit`.

## Contributing

Contributions are welcome Please feel free to submit a pull request or open an issue if you encounter any problems or have suggestions for improvement.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
