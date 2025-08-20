"""
WingMan – Agentic AI Assistant (Supervisor–Worker, Memory, Tools, FastAPI)

Dependencies (pip install):
  fastapi uvicorn pydantic[dotenv] openai faiss-cpu numpy httpx python-dotenv
Optional (for local TTS placeholder): gTTS playsound

Environment:
  export OPENAI_API_KEY="sk-..."
  # Optional demo API tokens for external services if you have them:
  export CALENDAR_API_BASE="https://example.com/calendar"
  export TASK_API_BASE="https://example.com/tasks"
  export ITOPS_API_BASE="https://example.com/itops"

Run:
  uvicorn wingman_app:app --reload --port 8000
"""

import os
import json
import time
import uuid
import queue
import faiss
import httpx
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings
from openai import OpenAI


# =========================
# Settings & Globals
# =========================

class Settings(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    calendar_api_base: str = os.getenv("CALENDAR_API_BASE", "https://example.com/calendar")
    task_api_base: str = os.getenv("TASK_API_BASE", "https://example.com/tasks")
    itops_api_base: str = os.getenv("ITOPS_API_BASE", "https://example.com/itops")

    memory_dim: int = 1536  # text-embedding-3-small dimension
    max_short_context: int = 12  # rolling short-term turns to keep
    retrieval_k: int = 5

    # Scheduler
    default_priority: int = 5  # lower number = higher priority
    conflict_backoff_sec: float = 0.1

settings = Settings()
client = OpenAI(api_key=settings.openai_api_key)


# =========================
# Memory: Hybrid (Short-term + FAISS Long-term)
# =========================

class HybridMemory:
    """
    Short-term rolling buffer + FAISS vector store for long-term memory.
    """
    def __init__(self, dim: int, retrieval_k: int = 5):
        self.dim = dim
        self.retrieval_k = retrieval_k

        # FAISS index (Inner Product for cosine with normalized vectors)
        self.index = faiss.IndexFlatIP(dim)
        self.vectors = []  # keeps numpy vectors (for rebuild if needed)
        self.payloads: List[Dict[str, Any]] = []

        # Short-term session memory
        self.short_term: Dict[str, deque] = {}

    # -------- Embeddings -------- #
    def embed(self, text: str) -> np.ndarray:
        resp = client.embeddings.create(
            model=settings.openai_embed_model,
            input=[text.strip()[:8000]],
        )
        v = np.array(resp.data[0].embedding, dtype=np.float32)
        # normalize for cosine/IP
        v /= np.linalg.norm(v) + 1e-12
        return v

    # -------- Short-term -------- #
    def get_short(self, session_id: str) -> List[Dict[str, str]]:
        return list(self.short_term.get(session_id, deque()))

    def add_short(self, session_id: str, role: str, content: str):
        buf = self.short_term.setdefault(session_id, deque(maxlen=settings.max_short_context))
        buf.append({"role": role, "content": content})

    # -------- Long-term -------- #
    def add_long(self, text: str, meta: Optional[Dict[str, Any]] = None):
        meta = meta or {}
        v = self.embed(text)
        self.index.add(v.reshape(1, -1))
        self.vectors.append(v)
        self.payloads.append({"text": text, "meta": meta, "id": str(uuid.uuid4())})

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        if len(self.payloads) == 0:
            return []
        k = k or self.retrieval_k
        qv = self.embed(query).reshape(1, -1)
        scores, idxs = self.index.search(qv, min(k, len(self.payloads)))
        results = []
        for i, score in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            item = self.payloads[i]
            results.append({"score": float(score), **item})
        return results


memory = HybridMemory(dim=settings.memory_dim, retrieval_k=settings.retrieval_k)


# =========================
# Task Scheduler (priority-based)
# =========================

@dataclass(order=True)
class ScheduledTask:
    priority: int
    created_at: float
    task_id: str = field(compare=False)
    action: Callable[..., Any] = field(compare=False, default=lambda: None)
    args: tuple = field(compare=False, default=())
    kwargs: dict = field(compare=False, default_factory=dict)
    description: str = field(compare=False, default="")
    owner: str = field(compare=False, default="")  # worker id

class PriorityScheduler:
    def __init__(self):
        self.q: "queue.PriorityQueue[ScheduledTask]" = queue.PriorityQueue()

    def submit(self, task: ScheduledTask):
        self.q.put(task)

    def run_once(self) -> Optional[Tuple[str, Any]]:
        if self.q.empty():
            return None
        task: ScheduledTask = self.q.get()
        try:
            result = task.action(*task.args, **task.kwargs)
            return (task.task_id, result)
        except Exception as e:
            return (task.task_id, {"error": str(e)})

    def run_all(self) -> List[Tuple[str, Any]]:
        results = []
        while not self.q.empty():
            r = self.run_once()
            if r is not None:
                results.append(r)
            time.sleep(settings.conflict_backoff_sec)
        return results

scheduler = PriorityScheduler()


# =========================
# Tools (3rd-party & system)
# =========================

class ToolError(Exception):
    pass

async def http_get_json(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None):
    async with httpx.AsyncClient(timeout=20.0) as s:
        r = await s.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r.json()

async def http_post_json(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None):
    async with httpx.AsyncClient(timeout=20.0) as s:
        r = await s.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

# ---- Calendar (demo) ---- #
async def tool_schedule_meeting(title: str, when_iso: str, attendees: List[str]) -> Dict[str, Any]:
    url = f"{settings.calendar_api_base}/events"
    payload = {"title": title, "when": when_iso, "attendees": attendees}
    try:
        data = await http_post_json(url, payload)
        return {"ok": True, "provider": "calendar", "data": data}
    except Exception as e:
        raise ToolError(f"calendar_error: {e}")

# ---- Tasks (demo) ---- #
async def tool_create_task(title: str, due_iso: Optional[str] = None, notes: Optional[str] = None) -> Dict[str, Any]:
    url = f"{settings.task_api_base}/tasks"
    payload = {"title": title, "due": due_iso, "notes": notes}
    try:
        data = await http_post_json(url, payload)
        return {"ok": True, "provider": "tasks", "data": data}
    except Exception as e:
        raise ToolError(f"task_error: {e}")

# ---- ITOps (demo) ---- #
async def tool_check_system_health(service_name: str) -> Dict[str, Any]:
    url = f"{settings.itops_api_base}/services/{service_name}/health"
    try:
        data = await http_get_json(url)
        return {"ok": True, "provider": "itops", "data": data}
    except Exception as e:
        raise ToolError(f"health_error: {e}")

async def tool_restart_service(service_name: str) -> Dict[str, Any]:
    url = f"{settings.itops_api_base}/services/{service_name}/restart"
    try:
        data = await http_post_json(url, {})
        return {"ok": True, "provider": "itops", "data": data}
    except Exception as e:
        raise ToolError(f"restart_error: {e}")

# ---- Info Retrieval (public HTTP GET demo) ---- #
async def tool_retrieve_info(url: str) -> Dict[str, Any]:
    try:
        data = await http_get_json(url)
        return {"ok": True, "provider": "http", "data": data}
    except Exception as e:
        raise ToolError(f"retrieve_error: {e}")


# =========================
# Workers
# =========================

class BaseWorker:
    name: str = "base"

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class SchedulingWorker(BaseWorker):
    name = "scheduling"

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected task = {"type": "schedule_meeting", "title": "...", "when": "...", "attendees": [...]}
        """
        return await tool_schedule_meeting(
            title=task["title"], when_iso=task["when"], attendees=task.get("attendees", [])
        )

class TaskWorker(BaseWorker):
    name = "tasking"

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected task = {"type": "create_task", "title": "...", "due": "...", "notes": "..."}
        """
        return await tool_create_task(
            title=task["title"], due_iso=task.get("due"), notes=task.get("notes")
        )

class ITOpsWorker(BaseWorker):
    name = "itops"

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        task types: "check_health", "restart_service"
        """
        t = task["type"]
        if t == "check_health":
            return await tool_check_system_health(task["service"])
        if t == "restart_service":
            return await tool_restart_service(task["service"])
        raise ToolError(f"Unknown itops task: {t}")

class RetrievalWorker(BaseWorker):
    name = "retrieval"

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected task = {"type": "retrieve_info", "url": "..."}
        """
        return await tool_retrieve_info(task["url"])


WORKERS: Dict[str, BaseWorker] = {
    "scheduling": SchedulingWorker(),
    "tasking": TaskWorker(),
    "itops": ITOpsWorker(),
    "retrieval": RetrievalWorker(),
}


# =========================
# Supervisor (LLM-driven with tool/function calling)
# =========================

TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "description": "Book a meeting in the user's calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "when": {"type": "string", "description": "ISO 8601 datetime"},
                    "attendees": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["title", "when"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a task in the user's task manager.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "due": {"type": "string", "description": "Optional ISO 8601 due datetime"},
                    "notes": {"type": "string"}
                },
                "required": ["title"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_system_health",
            "description": "Check health status for a given service via ITOps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"}
                },
                "required": ["service"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": "Restart a given service via ITOps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"}
                },
                "required": ["service"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_info",
            "description": "Retrieve JSON from a public HTTP endpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"]
            }
        },
    }
]

SYSTEM_PROMPT = """You are WingMan's Supervisor Agent.
- Parse user intent and decompose into one or more atomic subtasks.
- Choose appropriate functions to call. Prefer minimal, correct calls.
- Use concise JSON for tool arguments. Include ISO timestamps when required.
- After tools complete, synthesize a helpful, final answer for the user.
- Maintain continuity: consider retrieved long-term memory snippets when relevant.
- If scheduling or tasks involve preferences (e.g., working hours), incorporate from memory if available.
"""

class Supervisor:
    def __init__(self):
        pass

    def _chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        resp = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            temperature=0.2,
        )
        return resp

    async def plan_and_dispatch(
        self,
        session_id: str,
        user_text: str,
        context_snippets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Returns: {"final": "...", "tool_calls":[...], "tool_results":[...]}
        """
        # Inject memory snippets
        mem_snips = context_snippets or []
        mem_text = "\n".join([f"- {s}" for s in mem_snips]) if mem_snips else "None."

        short_ctx = memory.get_short(session_id)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if mem_snips:
            messages.append({"role": "system", "content": f"Long-term memory hints:\n{mem_text}"})
        messages.extend(short_ctx)
        messages.append({"role": "user", "content": user_text})

        # Step 1: Let the LLM pick functions to call
        llm_out = self._chat(messages, tools=TOOL_SPEC)
        msg = llm_out.choices[0].message

        tool_calls = getattr(msg, "tool_calls", None) or []
        tool_results = []

        # Step 2: For each tool call, dispatch to the right worker
        for call in tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")

            # Map function -> worker task
            if name == "schedule_meeting":
                worker = WORKERS["scheduling"]
                task = {
                    "type": "schedule_meeting",
                    "title": args["title"],
                    "when": args["when"],
                    "attendees": args.get("attendees", []),
                }
            elif name == "create_task":
                worker = WORKERS["tasking"]
                task = {
                    "type": "create_task",
                    "title": args["title"],
                    "due": args.get("due"),
                    "notes": args.get("notes"),
                }
            elif name == "check_system_health":
                worker = WORKERS["itops"]
                task = {
                    "type": "check_health",
                    "service": args["service"],
                }
            elif name == "restart_service":
                worker = WORKERS["itops"]
                task = {
                    "type": "restart_service",
                    "service": args["service"],
                }
            elif name == "retrieve_info":
                worker = WORKERS["retrieval"]
                task = {
                    "type": "retrieve_info",
                    "url": args["url"],
                }
            else:
                continue

            # Priority heuristic (safety: restarts > health > scheduling > tasks > retrieval)
            prio = {
                "restart_service": 1,
                "check_health": 2,
                "schedule_meeting": 3,
                "create_task": 4,
                "retrieve_info": 5,
            }.get(task["type"], settings.default_priority)

            # Enqueue for scheduler (can be expanded to parallel workers)
            task_id = str(uuid.uuid4())

            async def runner(w=worker, t=task):
                return await w.run(t)

            # Wrap the async runner for the sync scheduler using an event loop bridge
            import asyncio
            def action():
                return asyncio.run(runner())

            scheduler.submit(ScheduledTask(
                priority=prio,
                created_at=time.time(),
                task_id=task_id,
                action=action,
                description=f"{worker.name}:{task['type']}",
                owner=worker.name,
            ))

        # Step 3: Execute scheduled tasks now (simple, single-threaded loop)
        results = scheduler.run_all()
        tool_results_map = {tid: res for tid, res in results}

        # Step 4: Feed tool results back to the LLM for final synthesis
        if tool_calls:
            tool_messages = []
            for call in tool_calls:
                tid = call.id
                # The openai tool call id may not match our scheduled task id; use a simple mapping by order:
                # We'll pair results in the same order:
                # Build in order fallback
            ordered_results = [r for _, r in results]
            tool_msg_pack = json.dumps(ordered_results, ensure_ascii=False, indent=2)

            messages.append({
                "role": "tool",
                "content": tool_msg_pack,
            })

            final = self._chat(messages, tools=None)
            final_text = final.choices[0].message.content
        else:
            # No tools needed; ask the LLM to answer directly
            final2 = self._chat(messages, tools=None)
            final_text = final2.choices[0].message.content

        # Update memories
        memory.add_short(session_id, "user", user_text)
        memory.add_short(session_id, "assistant", final_text)

        # Persist salient facts (crude heuristic)
        salient = f"User said: {user_text}\nAssistant replied: {final_text[:500]}"
        memory.add_long(salient, meta={"session": session_id, "ts": time.time()})

        return {
            "final": final_text,
            "tool_calls": [
                {"name": c.function.name, "args": json.loads(c.function.arguments or "{}")}
                for c in tool_calls
            ],
            "tool_results": ordered_results if tool_calls else [],
        }


supervisor = Supervisor()


# =========================
# FastAPI Surface
# =========================

class ChatIn(BaseModel):
    session_id: Optional[str] = None
    text: str

class ChatOut(BaseModel):
    session_id: str
    final: str
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    retrieved_memory: List[str] = []

app = FastAPI(title="WingMan API", version="1.0.0")

@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn):
    session_id = body.session_id or str(uuid.uuid4())

    # Retrieve relevant long-term memory for context injection
    retrievals = memory.retrieve(body.text, k=settings.retrieval_k)
    snippets = [r["text"] for r in retrievals]

    result = await supervisor.plan_and_dispatch(
        session_id=session_id,
        user_text=body.text,
        context_snippets=snippets,
    )

    return ChatOut(
        session_id=session_id,
        final=result["final"],
        tool_calls=result["tool_calls"],
        tool_results=result["tool_results"],
        retrieved_memory=snippets,
    )


# ================
# Optional: Simple text-to-speech placeholder (sync)
# ================
# You can integrate a proper TTS engine or a cloud service here.
# Provided as a stub to show where "Final response synthesized into natural voice output" occurs.

def synthesize_voice_to_file(text: str, outfile: str = "wingman_tts.mp3") -> str:
    """
    Replace with your preferred TTS engine.
    Example with gTTS (requires internet and 'gTTS' package):
        from gtts import gTTS
        tts = gTTS(text=text, lang="en")
        tts.save(outfile)
    """
    # Stub: write text to a .txt as placeholder if TTS not available
    placeholder = outfile.replace(".mp3", ".txt")
    with open(placeholder, "w", encoding="utf-8") as f:
        f.write(text)
    return placeholder


# =========================
# Example: Boot-time seed memories (optional)
# =========================

if not memory.payloads:
    memory.add_long("User prefers meetings between 10:00–17:00 on weekdays.", meta={"type": "preference"})
    memory.add_long("Critical service: payments-api. Restart only with confirmation.", meta={"type": "policy"})
    memory.add_long("Pending project: Automate weekly on-call report generation.", meta={"type": "task"})


# =========================
# Notes
# =========================
# - This file implements a minimal, production-lean skeleton of the described system:
#   Supervisor–Worker decomposition, hybrid memory (short + FAISS long-term), priority scheduler,
#   dynamic tool selection (OpenAI function calling), and a FastAPI surface.
# - Workers call external services via async HTTP; swap example URLs with real integrations.
# - For multi-agent frameworks (LangChain / CrewAI), you can wrap each Worker with an Agent class
#   and/or use their task orchestration primitives in place of the lightweight scheduler above.
# - Add authentication, rate limiting, observability (traces/metrics), and retries for production use.
