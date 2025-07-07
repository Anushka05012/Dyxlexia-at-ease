import json
from vosk import Model, KaldiRecognizer
import pyaudio

# Load your word list from spell.json
with open('words.json', 'r') as f:
    data = json.load(f)

word_list = data['words']  # Assuming your JSON structure has a 'words' key with the list

# Prepare grammar string for Vosk recognizer
grammar = f'["{"\",\"".join(word_list)}"]'

# Initialize Vosk model and recognizer with grammar
model = Model("vosk-model-small-en-in-0.4")  # path to your model
rec = KaldiRecognizer(model, 16000, grammar)

# Setup audio stream for recognition
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

print("Say something...")

while True:
    data = stream.read(4000, exception_on_overflow=False)
    if rec.AcceptWaveform(data):
        result = rec.Result()
        print(result)
    else:
        partial = rec.PartialResult()
        print(partial)
