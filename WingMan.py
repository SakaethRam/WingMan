# --------------------------------------------------------------
# BASIC WingMan – voice-controlled Gemini chat only
# --------------------------------------------------------------
import speech_recognition as sr
import google.generativeai as genai
from gtts import gTTS
import tempfile, os, time
from pygame import mixer

# ---------- CONFIG ----------
genai.configure(api_key="<GEMINI_API_KEY>")
MODEL = "<GEMINI_MODEL>"
LANG = "en"                     # en / hi
# --------------------------------

recognizer = sr.Recognizer()
model = genai.GenerativeModel(MODEL)
mixer.init()

def speak(text: str):
    lang = "hi" if LANG == "hi" else "en"
    tts = gTTS(text=text, lang=lang, slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        path = f.name
    mixer.music.load(path)
    mixer.music.play()
    while mixer.music.get_busy():
        time.sleep(0.1)
    os.unlink(path)

def listen() -> str | None:
    with sr.Microphone() as src:
        recognizer.adjust_for_ambient_noise(src, 0.5)
        print("[Listening]")
        try:
            audio = recognizer.listen(src, timeout=5, phrase_time_limit=10)
            txt = recognizer.recognize_google(audio, language="hi-IN" if LANG=="hi" else "en-IN")
            print(f"You: {txt}")
            return txt
        except sr.UnknownValueError:
            speak("Sorry, I didn’t catch that.")
        except sr.RequestError:
            speak("No internet.")
        return None

def gemini_reply(prompt: str) -> str:
    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"Error: {e}"

# ---------- MAIN ----------
if __name__ == "__main__":
    greet = "Hello!" if LANG == "en" else "नमस्ते!"
    print(f"WingMan X: {greet}")
    speak(greet)

    while True:
        user = listen()
        if not user:
            continue
        if any(w in user.lower() for w in ["bye","exit"]):
            bye = "Bye!" if LANG == "en" else "अलविदा!"
            print(f"WingMan X: {bye}")
            speak(bye)
            break

        reply = gemini_reply(user)
        print(f"WingMan X: {reply}")
        speak(reply)
