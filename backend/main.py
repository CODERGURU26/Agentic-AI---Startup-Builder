"""
main.py - FastAPI Server

Implements the full 6-step user flow:

  POST /api/build          → Steps 1-5 (sync, returns full result)
  GET  /api/build/stream   → Steps 1-5 (SSE streaming, live progress)
  GET  /api/health         → Health check
  GET  /api/info           → LLM config info

Run:
    uvicorn main:app --reload --port 8000

Then open:
    http://localhost:8000/docs    ← Interactive API docs (Swagger UI)
"""

import asyncio
import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
from crew import build_startup_plan
from llm_factory import get_llm_info

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(
    title="🚀 AI Startup Builder API",
    description=(
        "Multi-agent system that transforms a startup idea into a complete blueprint.\n\n"
        "**User Flow:**\n"
        "1. User Input → POST body\n"
        "2. Idea Processing → LLM initialization\n"
        "3. Multi-Agent Analysis → 4 specialized CrewAI agents\n"
        "4. Collaborative Decision Making → sequential context passing\n"
        "5. Output Generation → structured blueprint\n"
        "6. Results Displayed → JSON response to Next.js frontend"
    ),
    version="1.0.0",
)

# Allow Next.js frontend (Step 3 of tech stack)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Next.js dev server
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────

class BuildRequest(BaseModel):
    idea: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Your startup idea in plain English",
        example="AI fitness app that creates personalized workout plans using sleep and DNA data",
    )


class AgentOutput(BaseModel):
    pm_output:      str
    market_output:  str
    finance_output: str
    dev_output:     str


class BuildResponse(BaseModel):
    startup_idea:   str
    generated_at:   str
    llm_info:       dict
    pm_output:      str
    market_output:  str
    finance_output: str
    dev_output:     str
    full_report:    str
    report_path:    str


class SSEEvent(BaseModel):
    agent:   str
    status:  str   # started | thinking | done | complete | error
    content: str


# ── Routes ────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
def health():
    """Step 0 — confirm the server is running."""
    return {"status": "ok", "message": "🚀 AI Startup Builder API is live"}


@app.get("/api/info", tags=["System"])
def info():
    """Returns which LLM provider + model is active."""
    try:
        config.validate()
        return {"status": "configured", **get_llm_info()}
    except EnvironmentError as e:
        return {"status": "misconfigured", "error": str(e)}


@app.post("/api/build", response_model=BuildResponse, tags=["Startup Builder"])
def build(request: BuildRequest):
    """
    **Steps 1–5 (synchronous)**
    
    Submit a startup idea, wait for all 4 agents to finish,
    receive the complete blueprint as JSON.
    
    ⚠️  This can take 2–5 minutes. Use `/api/build/stream` for live updates.
    """
    try:
        config.validate()
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        result = build_startup_plan(request.idea)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crew failed: {str(e)}")


@app.get("/api/build/stream", tags=["Startup Builder"])
async def build_stream(
    idea: str = Query(
        ...,
        min_length=5,
        description="Your startup idea",
        example="AI fitness app using sleep data",
    )
):
    """
    **Steps 1–6 (Server-Sent Events streaming)**
    
    Each agent emits progress events in real-time as it works.
    The Next.js frontend listens to this endpoint to show live updates.
    
    **Event format:**
    ```
    data: {"agent": "product_manager", "status": "started", "content": "..."}
    ```
    
    **Status values:**
    - `started`  → agent has begun working
    - `thinking` → intermediate reasoning step
    - `done`     → agent finished, content = full output
    - `complete` → all agents done
    - `error`    → something went wrong
    """
    try:
        config.validate()
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return StreamingResponse(
        _stream_generator(idea),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",    # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── SSE Generator ─────────────────────────────────────────────────

async def _stream_generator(idea: str) -> AsyncGenerator[str, None]:
    """
    Runs the crew in a thread (it's blocking) and yields SSE events.
    
    The on_progress callback bridges the sync crew world → async SSE world
    via an asyncio.Queue.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_progress(agent: str, status: str, content: str):
        """Called from the crew thread — puts events onto the queue."""
        event = {"agent": agent, "status": status, "content": content}
        loop.call_soon_threadsafe(queue.put_nowait, event)

    # Run blocking crew.kickoff() in a background thread
    async def run_crew():
        try:
            await asyncio.to_thread(build_startup_plan, idea, on_progress)
        except Exception as e:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"agent": "system", "status": "error", "content": str(e)}
            )
        finally:
            # Sentinel — tells the generator to stop
            loop.call_soon_threadsafe(queue.put_nowait, None)

    asyncio.create_task(run_crew())

    # Yield events as they arrive
    while True:
        event = await queue.get()
        if event is None:
            break
        yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(0)  # allow other tasks to run

    yield "data: {\"agent\": \"system\", \"status\": \"stream_end\", \"content\": \"\"}\n\n"


# ── Dev runner ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT, reload=True)