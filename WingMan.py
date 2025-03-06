import re
import speech_recognition as sr
import google.generativeai as genai
from googletrans import Translator
from gtts import gTTS
import os

# Configure Gemini API
genai.configure(api_key= "<API KEYS>" )

# Function to recognize speech and convert it to text
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Bonjour ! Comment puis-je vous aider ?")
        print("(Hello!, How can I help you ?)")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="auto")
        print(f"Prompt: {text}")
        return text.lower()  # Convert to lowercase for easy comparison
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results, please check your connection.")
        return None


# Function to translate text to English
def translate_text(text, target_lang="en"):
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    return translation.text, translation.src


# Function to get response from Gemini AI
def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text  # Extracting the generated response


# Function to clean text for audio output
def clean_text_for_audio(text):
    text = re.sub(r"\*+", "", text)  # Remove all '*' symbols
    text = re.sub(r"\(.*?\)", "", text)  # Remove content inside parentheses
    return text.strip()


# Function to convert text to speech and play the audio
def speak_text(text, lang="en"):
    cleaned_text = clean_text_for_audio(text)  # Remove unwanted characters
    if cleaned_text:  # Only generate audio if there's text left
        tts = gTTS(text=cleaned_text, lang=lang)
        tts.save("response.mp3")
        os.system("start response.mp3" if os.name == "nt" else "afplay response.mp3")


# Main loop to keep listening and responding
while True:
    user_input = recognize_speech()

    if user_input:  # Only proceed if speech recognition succeeds
        if "wingman" in user_input:  # If "audio" keyword is detected, trigger MP3 response
            print("Wingman mode activated.")
            user_input = user_input.replace("using wingman", "").strip()  # Remove "audio" keyword from prompt

            if user_input:  # If there's still a valid query, proceed
                translated_input, detected_lang = translate_text(user_input, "en")
                gemini_response = get_gemini_response(translated_input)
                speak_text(gemini_response, detected_lang)  # Play audio only
        else:
            translated_input, detected_lang = translate_text(user_input, "en")
            if detected_lang != "en":
                print(f"Translated Prompt (English): {translated_input}")

            gemini_response = get_gemini_response(translated_input)
            print(f"Gemini Response ({detected_lang}): {gemini_response}")
    else:
        print("No valid input detected. Please try again.")

