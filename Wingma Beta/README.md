### Arkin X: Arkin X proudly presents  Wingman Beta, a cutting-edge virtual assistant designed to redefine task automation and elevate user experiences.  

### Wingman Beta:  

With increasing globalization, many users face difficulties in interacting with AI-powered virtual assistants due to language barriers. A seamless multilingual voice assistant is needed to allow users to speak in their native language while ensuring that the response is both spoken in the same language and subtitled in English for better understanding.


 ![WINGMAN BETA](https://github.com/user-attachments/assets/dfe6a235-1a9e-475d-bb99-d1bd90ef21b5)


### **Introduction**  
This project develops a multilingual AI-powered voice assistant capable of understanding and responding in multiple languages. The assistant recognizes user speech, translates it into English for processing by OpenAI's GPT model, and then generates responses in the original spoken language. Additionally, the assistant provides an English subtitle option and is controlled via a GUI button for starting and stopping voice input.

### **Prerequisites**  

Before running the project, ensure the following:  

1. **Library Installations** – Install all necessary Python libraries, including:  
   - `SpeechRecognition` for speech-to-text conversion  
   - `openai` for generating AI-based responses  
   - `googletrans` for automatic language translation  
   - `gtts` (Google Text-to-Speech) for converting responses to speech  
   - `pillow` for handling images in the GUI  

2. **API Key Configuration** – Obtain an OpenAI API key from the OpenAI Developer Portal and replace the placeholder in the script.  

3. **GUI Image Preparation** – Upload a PNG file to be used as a button in the GUI interface. Ensure the image is properly formatted and accessible within the script.  

4. **Microphone Setup** – The system must have a functioning microphone to capture user speech accurately. If using a virtual environment or remote server, ensure access to local audio input.  

5. **Machine Learning Model Usage** – The project relies on **pre-trained AI models** for:  
   - **Speech Recognition:** Google Web Speech API  
   - **Natural Language Processing (NLP):** OpenAI GPT-3.5 Turbo  
   - **Translation Model:** Google Translate API  
   - **Text-to-Speech Conversion:** Google Text-to-Speech (gTTS)  

### **Future Advancements (Recommended)**  

1. **Offline Functionality** – Integrating offline speech recognition and language translation models to eliminate dependency on internet-based APIs. This can be achieved using **Mozilla DeepSpeech** for offline speech recognition and **OpenNMT** for on-device language translation.  

2. **Voice Customization and Synthesis** – Enhancing user experience by integrating **advanced speech synthesis models** like **Amazon Polly** or **Google WaveNet**, allowing users to choose from multiple voice profiles, accents, and tones to personalize their interactions.  

3. **Context-Aware Conversational Memory** – Implementing **dialogue history tracking** using **Transformer-based models** (e.g., GPT with memory augmentation) to retain context across conversations. This will enable the assistant to provide **coherent multi-turn responses** and improve user engagement.  
