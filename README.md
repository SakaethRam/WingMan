# WingMan ITOps: Agentic AI for IT Operations

## Introduction

**WingMan X** is a powerful, voice-driven AI assistant built for strategy tracking, task management, and intelligent conversation. It combines **Gemini AI**, **CRM Integration**, **persistent memory**, and **multilingual speech** (English & Hindi) into a seamless, hands-free productivity tool.

Whether you're logging deliverables, recalling past decisions, or brainstorming with an AI || All By Voice || WingMan X: Keeps You In Flow

---

![WINGMAN](https://github.com/user-attachments/assets/4752e7ac-e04e-420f-8dec-5c1e779fa52c)

---

## Key Features

1. **Voice-First Interaction:** Speak naturally; get spoken responses.
2. **Multilingual Support:** English & Hindi with real-time Devanagari conversion.
3. **Airtable CRM Sync:** Add, update, and fetch deliverables via voice commands.
4. **Persistent Memory:** Remembers chat history, language, and mode across sessions.
5. **Smart Response Caching:** Uses TF-IDF to reuse past answers (saves API calls).
6. **Developer & User Modes:** Toggle debug logs and command visibility.
7. **Chat History Tools:** Show, search, and export conversation logs.
8. **Auto CRM Updates:** Periodic refresh of latest tasks.
9. **Offline Graceful Degradation:** Handles no-internet scenarios smoothly.

---

## Technologies and Framework

| Category              | Technology / Library                                   |
|-----------------------|--------------------------------------------------------|
| **Framework**         | Custom Python engine (`wingmanx.py`)                   |
| **AI Model**          | Google Gemini (`gemini-2.0-flash`)                     |
| **Vector Store**      | Scikit-learn `TfidfVectorizer` + `cosine_similarity`   |
| **Speech Recognition**| `speech_recognition` + Google Web API                  |
| **Text-to-Speech**    | `gTTS` (Google Text-to-Speech)                         |
| **Audio Playback**    | `pygame.mixer`                                         |
| **CRM Integration**   | Airtable API (REST)                                    |
| **Translation**       | `googletrans`                                          |
| **Utilities**         | `json`, `requests`, `re`, `datetime`, `tempfile`, `os` |
| **Configuration**     | API keys via `WingManX()` constructor                  |
| **Optional TTS**      | Falls back to text print if audio fails                |

---

## Setup and Execution

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/WingMan-X.git
cd WingMan-X
```

### **2. Configure Environment Variables and Get API Keys**

1. Gemini API Key: Google AI Studio
2. Airtable Personal Access Token: Airtable → Account → Personal Access Tokens
3. Airtable Base ID: Found in your base URL → appXXXXXXXXXXXXXX
4. Create a `.env` file or export variables directly:

```bash
export OPENAI_API_KEY="sk-..."
export Gemini_API ="<GEMINI_API_KEYS>"
export Airtable Personal Access Token ="<PERSONAL_ACCESS_TOKEN>"
export Airtable Base ID ="<BASE_ID>"
```


### **3. Run the Assistant [Test the Advanced Engine]**

Use the provided WingMan X StudioCode.py to instantly launch and test the full WingMan X Engine:
```bash
# WingMan X StudioCode.py
from wingmanx import WingManX

WingManX(
    gemini_key="your-gemini-api-key",
    airtable_key="your-airtable-pat",
    base_id="appXXXXXXXXXXXXXX"
).run()
```
>This is your sandbox, plug in your keys and start talking. No setup beyond dependencies and keys.

---

## Project Structure

```Text
WingMan-X/
├── wingmanx.py              # Core engine (advanced model)
├── WingMan X StudioCode.py  # Test runner (this file)
├── Memory.json              # Auto-generated: chat history & settings
├── requirements.txt
└── README.md
```

---

## Voice Commands

| Action                | Say…                                                   |
|-----------------------|--------------------------------------------------------|
| **Switch mode**       | "mode" or "developer" / "user"                         |
| **Change language**   | "language"                                             |
| **Add deliverable**   | "add update: Launch MVP"                               |
| **Update last task**  | "update last: delayed to Friday"                       |
| **Get all updates**   | "update" or "status"                                   |
| **Show chat history** | "show history"                                         |
| **Search past chat**  | "search: MVP"                                          |
| **Export chat**       | "export chat"                                          |
| **Exit**              | "bye" or "exit"                                        |

---

## Important Notes

1. **Internet Required** for speech recognition, TTS, Gemini, and Airtable.
2. **Microphone Access** must be granted.
3. Hindi TTS uses `gTTS` with `tld='co.in'` for natural Indian accent.
4. **Memory Cleanup:** Old messages (>200) are auto-trimmed.
5. **Developer Mode** shows debug logs and command hints in console.
   
---

## Contribution Guidelines

1. Fork the repository.
2. Create a new feature branch.
3. Add or improve functionality.
4. Submit a pull request with a clear description of changes.

---

Built by **S A M** – Intelligent automation for IT Operations.
