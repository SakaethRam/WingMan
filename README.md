# WingMan: Your AI Voice Wingman

## Introduction  
WingMan is an advanced AI-powered voice assistant designed to provide real-time speech recognition, language translation, and AI-generated responses using Google’s Gemini API. It acts as a conversational AI that understands multilingual inputs, translates them, and provides intelligent responses. Additionally, WingMan can generate speech responses, making it an interactive and accessible assistant.  

---

![WINGMAN](https://github.com/user-attachments/assets/4752e7ac-e04e-420f-8dec-5c1e779fa52c)

---

## Key Features  

1. **Speech Recognition** – Uses Google’s Speech Recognition API to capture and transcribe spoken words accurately.  
2. **Multilingual Support** – Detects and translates languages automatically, ensuring a seamless conversation experience.  
3. **AI-Powered Responses** – Integrates Google’s Gemini API to generate insightful responses based on user queries.  
4. **Text-to-Speech Conversion** – Uses gTTS (Google Text-to-Speech) to convert AI-generated responses into natural-sounding audio.  
5. **Interactive Voice Mode** – Saying **"WingMan"** in your input triggers an **audio response** in `response.mp3`.  
6. **Real-Time Execution** – Works in a continuous loop, always ready to listen, process, and respond to user commands.  

---

## Technologies and Libraries Used  

- **Speech Recognition:** `speech_recognition` (Google Speech Recognition API)  
- **AI Response Generation:** `google-generativeai` (Google Gemini API)  
- **Language Translation:** `googletrans` (Google Translate API)  
- **Text-to-Speech:** `gTTS` (Google Text-to-Speech)  
- **Regular Expressions:** `re` (for cleaning text before audio processing)  
- **System Interaction:** `os` (for playing audio responses)  

---

## Insights and Benefits  

- **Multilingual AI Assistant** – WingMan makes AI-driven interactions accessible to people speaking different languages.  
- **Real-Time Communication** – Enables hands-free and instant responses for users.  
- **Seamless Translation** – Removes language barriers, allowing users to communicate effortlessly.  
- **Smart Voice Interaction** – Enhances engagement with AI by converting text responses into speech when "WingMan" is detected.  
- **Flexible Execution** – Works on both Windows and macOS with voice command-based activation.  

---

## How to Set Up and Execute WingMan  

### **1. Install Required Libraries**  
Before running WingMan, install the necessary Python packages using:  
```bash
pip install speechrecognition google-generativeai googletrans==4.0.0-rc1 gtts
```

### **2. Set Up Google Gemini API**  
Replace `"<API KEYS>"` with your Google Gemini API key in the script:  
```python
genai.configure(api_key="<YOUR_GEMINI_API_KEY>")
```

### **3. Run WingMan**  
Save the script as `wingman.py` and execute:  
```bash
python wingman.py
```

### **4. Activate Audio Response Mode**  
- If your input contains **"WingMan"**, the AI response will be **converted to speech** and saved as `response.mp3`, which will then be played.  
- For all other queries, the response will be displayed as text in the terminal.  

---

## Model Details
Check the available "Gemini Models" using the provided code snippet:  
```python
models = genai.list_models()
for model in models:
    print(model.name) 
```

---

## Docker File

### **1. Build Docker Image**  
Run the following command in the directory where your `DockerFile` is located:  
```bash
docker build -t wingman .
```

### **2. Run the Docker container**  
After building the image, start the container with: 
```bash
docker run --rm -it --device /dev/snd wingman
```

---

## **Contribution Guidelines**  
Contributions are welcome. To contribute:  
1. Fork the repository.  
2. Create a feature branch.  
3. Implement your changes.  
4. Submit a pull request with a clear description of modifications.  

---

Built by `S A M` – Building intelligence to voice.
