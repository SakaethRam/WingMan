import pyttsx3
import speech_recognition as sr
import datetime
import webbrowser
import os
import psutil  # Library to monitor system utilization
import threading
import spotipy
from spotipy.oauth2 import SpotifyOAuth

WAKE_WORD = "zenith"  # Change this to any other word you prefer
is_paused = False  # Global flag to handle wait/start commands

# Spotify credentials
SPOTIFY_CLIENT_ID = "your_client_id"  # Replace with your Client ID
SPOTIFY_CLIENT_SECRET = "your_client_secret"  # Replace with your Client Secret
SPOTIFY_REDIRECT_URI = "http://localhost:8080/callback"  # Must match Redirect URI in Spotify Developer Dashboard


# Set up Spotipy
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope="user-read-playback-state,user-modify-playback-state,user-read-currently-playing"
))

def play_spotify_playlist():
    """Play a Spotify playlist in a separate thread."""
    try:
        # Spotify playlist URI
        playlist_uri = "spotify:playlist:your_playlist_id"

        # Get the user's active device
        devices = sp.devices()
        if not devices["devices"]:
            speak("No active Spotify devices found. Please open Spotify on your device.")
            return

        # Use the first available device
        device_id = devices["devices"][0]["id"]

        # Start playback
        sp.start_playback(device_id=device_id, context_uri=playlist_uri)
        speak("Playing your Spotify playlist.")
    except Exception as e:
        print(f"Error with Spotify playback: {e}")
        speak("I couldn't connect to Spotify. Please check your credentials or open Spotify.")

def play_music_in_background():
    """Run the Spotify playback logic in a new thread."""
    music_thread = threading.Thread(target=play_spotify_playlist)
    music_thread.start()

def speak(text):
    """Convert text to speech"""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen():
    """Capture voice input and convert it to text"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout="")
            print("Recognizing...")
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            return command.lower()
        except sr.UnknownValueError:
            if not is_paused:
                speak("Sorry, I didn't catch that. Can you say it again?")
        except sr.RequestError:
            if not is_paused:
                speak("There was an issue with the speech recognition service.")
        return ""

def wake_word():
    """Handle the 'wait' and 'start' commands to pause or resume the assistant."""
    global is_paused
    while is_paused:
        command = listen()
        if command:
            if "start" in command:
                speak("I am back. How can I assist you?")
                is_paused = False
                break
            elif "stop" in command:
                speak("Goodbye! I'm going to sleep now.")
                exit()

def check_system_utilization():
    """Check and report CPU and memory usage"""
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    speak(f"The current CPU usage is {cpu_usage} percent.")
    speak(f"The memory usage is {memory_usage} percent.")
    if cpu_usage < 50 and memory_usage < 50:
        speak("The system is performing optimally.")
    else:
        speak("The system is under high load. You may experience slower performance.")

def help_command():
    """Provide a list of available commands and their descriptions."""
    help_text = (
        "Here are the commands you can use: "
        "1. 'time' - Tells the current time. "
        "2. 'open youtube' or 'youtube' - Opens YouTube in your browser. "
        "3. 'google' - Opens Google in your browser. "
        "4. 'music' - Plays the songs from your Spotify playlist. "
        "5. 'system utilization' or 'cpu and memory' - Reports the CPU and memory usage. "
        "6. 'wait' - Pauses the assistant until you say 'start'. "
        "7. 'stop' or 'bye' - Exits the assistant. "
        "8. 'assist me' - Explains the available commands. "
    )
    speak(help_text)

def handle_command(command):
    """Process the command and perform actions"""
    global is_paused
    if "time" in command:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The current time is {current_time}")

    elif any(keyword in command for keyword in ["open youtube", "youtube"]):
        webbrowser.open("https://www.youtube.com")
        speak("Opening YouTube")

    elif "google" in command:
        webbrowser.open("https://www.google.com")
        speak("Opening Google")

    elif "music" in command:
        music_dir = "C:\\Users\\Username\\Music"  # Update with your music directory path
        try:
            music_thread = threading.Thread(target=play_spotify_playlist)
            music_thread.start()
        except FileNotFoundError:
            speak("The music directory was not found. Please check the path.")

    elif "stop" in command or "bye" in command:
        speak("Goodbye! Have a great day!")
        exit()

    elif "wait" in command:
        speak("Okay, I will wait. Call me when you are ready.")
        is_paused = True

    elif "system utilisation" in command or "cpu and memory" in command:
        check_system_utilization()

    elif "assist me" in command:
        help_command()

    else:
        speak("I'm sorry, I don't know how to do that yet.")

def main():
    global is_paused
    speak(f"Hello! I am your virtual assistant. How can I help you today? Call me by saying '{WAKE_WORD}'.")
    while True:
        speak("Do you want to hear about the available commands? Please say 'yes, I would like to' or 'no, i don't'.")
        initial_command = listen()
        if "yes" in initial_command:
            help_command()
            break
        elif "no" in initial_command:
            break
        else:
            speak("I didn't catch that.")

    speak("Alright. You can now give me a command.")
    while True:
        if not is_paused:
            command = listen()
            if command:
                if WAKE_WORD in command:
                    speak("Yes, how can I help you?")
                    continue
                handle_command(command)
        else:
            wake_word()

if __name__ == "__main__":
    main()
