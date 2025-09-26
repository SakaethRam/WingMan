# WingMan ITOps: Agentic AI for IT Operations

## Introduction

**WingMan ITOps** is an agentic AI assistant designed for IT Operations, automation, and task management. It leverages a **Supervisor–Worker architecture**, **hybrid memory (short-term + FAISS long-term)**, and **priority-based scheduling** to handle IT tasks intelligently. Using OpenAI models with FastAPI, it integrates seamlessly with calendars, task managers, and IT operations systems.

---

![WINGMAN](https://github.com/user-attachments/assets/4752e7ac-e04e-420f-8dec-5c1e779fa52c)

---

WingMan ITOps Gist:

1. Understand user queries via natural language.
2. Decompose them into atomic subtasks.
3. Call the appropriate tools or workers (e.g., scheduling, tasking, ITOps, retrieval).
4. Return a synthesized, context-aware final response.

---

## Key Features

1. **Supervisor–Worker Framework** – Central supervisor parses intent and delegates to specialized workers.
2. **Hybrid Memory System** – Short-term rolling context + FAISS vector-based long-term memory.
3. **Priority Scheduler** – Ensures urgent tasks (e.g., restarting services) take precedence.
4. **ITOps Automation** – Health checks, service restarts, and system monitoring via API.
5. **Calendar & Task Management** – Book meetings and manage tasks through connected APIs.
6. **Context-Aware Responses** – Integrates retrieved long-term memory into final answers.
7. **FastAPI Interface** – Provides an API surface for easy integration with existing systems.
8. **Extendable Tools** – Simple structure to add more external integrations.

---

## Technologies and Libraries Used

* **Framework:** `FastAPI`, `Uvicorn`
* **AI Models:** `openai` (Chat & Embeddings APIs)
* **Vector Store:** `faiss-cpu` for long-term memory retrieval
* **Utilities:** `numpy`, `pydantic`, `httpx`, `queue`, `uuid`
* **Configuration:** `python-dotenv` for environment variables
* **Optional TTS:** `gTTS`, `playsound` (placeholder implementation)

---

## Insights and Benefits

* **Automated IT Operations** – Simplifies health checks, restarts, and monitoring.
* **Intelligent Scheduling** – Books meetings or creates tasks with contextual awareness.
* **Scalable Agent Design** – Easily extendable to new workers and APIs.
* **Prioritized Task Handling** – Critical operations are handled first via a priority queue.
* **Persistent Knowledge** – Retains and recalls preferences, policies, and historical events.

---

## Setup and Execution

### **1. Install Dependencies**

```bash
pip install fastapi uvicorn pydantic[dotenv] openai faiss-cpu numpy httpx python-dotenv
# Optional (for demo TTS)
pip install gTTS playsound
```

### **2. Configure Environment Variables**

Create a `.env` file or export variables directly:

```bash
export OPENAI_API_KEY="sk-..."
# Optional demo API tokens:
export CALENDAR_API_BASE="https://example.com/calendar"
export TASK_API_BASE="https://example.com/tasks"
export ITOPS_API_BASE="https://example.com/itops"
```

### **3. Run the Application**

```bash
uvicorn wingman_app:app --reload --port 8000
```

### **4. Example API Usage**

Send a chat request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Check health of payments-api"}'
```

Response:

```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "final": "The payments-api service is healthy and running normally.",
  "tool_calls": [{"name": "check_system_health", "args": {"service": "payments-api"}}],
  "tool_results": [{"ok": true, "provider": "itops", "data": {"status": "healthy"}}],
  "retrieved_memory": ["Critical service: payments-api. Restart only with confirmation."]
}
```

---

## API Endpoints

### `POST /chat`

* **Input:**

```json
{
  "session_id": "optional-session-id",
  "text": "Restart the payments-api service"
}
```

* **Output:**

```json
{
  "session_id": "...",
  "final": "The payments-api service has been restarted successfully.",
  "tool_calls": [...],
  "tool_results": [...],
  "retrieved_memory": [...]
}
```

---

## Extending WingMan ITOps

* **Add new tools** by defining functions (`tool_...`) for external APIs.
* **Add new workers** by subclassing `BaseWorker` and implementing the `run` method.
* **Modify supervisor behavior** by updating the system prompt or tool specifications.

---

## Contribution Guidelines

1. Fork the repository.
2. Create a new feature branch.
3. Add or improve functionality.
4. Submit a pull request with a clear description of changes.

---

Built by **S A M** – Intelligent automation for IT Operations.
