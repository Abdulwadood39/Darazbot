import pyttsx3

def TTS(text:str): 
    engine = pyttsx3.init('nsss')
    voices = engine.getProperty('voices')
    selected_voice = voices[66].id
    engine.setProperty('voice', selected_voice) 
    engine.setProperty('rate', 170) 
    
    if engine._inLoop:
        engine.endLoop()
        
    # text = "hello world."
    engine.say(text)
    engine.runAndWait()
