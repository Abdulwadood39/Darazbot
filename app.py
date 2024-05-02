import warnings
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request, jsonify
import os
import speech_recognition as sr
import chatbot
import tempfile
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

Darazbot = chatbot.Darazbot()

def getResponse(prompt:str):
    '''
        prompt is the question
    '''
    response = Darazbot.query(prompt)
    return response

def text_to_speech(text:str):
    '''
        text to speak
    '''
    
    Darazbot.talk(text)



@app.route('/submit', methods=['POST'])
def submit():
    text = request.json['text']
    
    response = getResponse(text)

    return jsonify({'response': response})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_file = request.files['audio_data']
    audio_data = audio_file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file.seek(0)
        
        r = sr.Recognizer()

        with sr.AudioFile(temp_audio_file.name) as source:
            audio = r.record(source)

    transcription = r.recognize_whisper(audio, model="medium", language="en")
    print(transcription)

    return jsonify({'transcription': transcription})

@app.route('/process_text', methods=['POST'])
def process_text():    
    if request.method == "POST":
        text_data = request.json['text_data']
        text_to_speech(text_data)
    else:
        return jsonify({"Success":False, "message":"Text Not coming as input."}), 504
    return jsonify({"Success":True}), 200

@app.route("/")
def default():
    return render_template("home.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/index", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        audio_data = request.files['audio_data']
        audio_data.save('./file.wav')
        if os.path.isfile('./file.wav'):
            print("./file.wav exists")
        return render_template('index.html', request="POST")   
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)