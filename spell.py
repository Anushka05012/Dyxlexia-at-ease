import sounddevice as sd
import queue
import sys
import json
import pyttsx3
import time
from vosk import Model, KaldiRecognizer
import numpy as np
import difflib

def load_vocabulary(file_path, max_words=5000):
    with open(file_path, "r") as f:
        words = [line.strip().lower() for line in f.readlines()[:max_words]]
    vocab_json = json.dumps(words + ["[unk]"])
    return words, vocab_json

def levenshtein_distance(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

engine = pyttsx3.init()

vocab_path = "C:/Users/agbha/OneDrive/Desktop/B.tech/SY/sem II/EDAI/EDAI(Dyxlexia)/common_words_5000.txt"
model_path = "C:/Users/agbha/OneDrive/Desktop/B.tech/SY/sem II/EDAI/EDAI(Dyxlexia)/vosk-model-small-en-in-0.4"

words, vocabulary_json = load_vocabulary(vocab_path)
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000, vocabulary_json)
q = queue.Queue()

def callback(indata, frames, time_info, status):
    if status:
        print("Mic status:", status, file=sys.stderr)
    q.put(bytes(indata))

def speak_word(word, slow=False):
    engine.setProperty('rate', 150)
    if slow:
        engine.setProperty('rate', 100)
    engine.say(word)
    engine.runAndWait()

def listen_and_recognize():
    print("Say any common English word (from vocab of 5,000 words)")

    with sd.RawInputStream(samplerate=16000, blocksize=8192, dtype='int16', channels=1, callback=callback):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip().lower()
                if text:
                    print(f"Recognized: '{text}'")

                    best_match = max(words, key=lambda word: levenshtein_distance(word, text))
                    match_score = levenshtein_distance(best_match, text)

                    if match_score == 1:
                        print("Correct pronunciation!")
                        speak_word(text)
                    elif match_score > 0.25:
                        print("Almost there! Keep trying!")
                        speak_word(text, slow=True)
                    else:
                        print("Wrong pronunciation. Try again!")
                        speak_word(best_match, slow=True)

                    time.sleep(2)

            else:
                partial = json.loads(recognizer.PartialResult())
                word = partial.get("partial", "").strip()
                if word:
                    print(f"(Partial): {word}", end="\r")

                    best_match_partial = max(words, key=lambda word: levenshtein_distance(word, word))
                    match_score_partial = levenshtein_distance(best_match_partial, word)

                    if match_score_partial > 0.25:
                        print("Almost there! Keep trying!")
                        speak_word(word, slow=True)
                    else:
                        print("Pronunciation might be unclear or wrong.")

                    time.sleep(2)

if __name__ == "_main_":
    listen_and_recognize()
