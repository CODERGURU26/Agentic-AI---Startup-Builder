"""
crew.py - Step 4 (Collaborative Decision Making) of the user flow

Orchestrates all 4 agents sequentially.
Each agent receives the previous agent's output as context —
this IS the "collaborative decision making" where agents share insights.

User Flow mapping:
  Step 1 (User Input)              → startup_idea param
  Step 2 (Idea Processing via LLM) → llm_factory.get_llm()
  Step 3 (Multi-Agent Analysis)    → each agent runs its task
  Step 4 (Collaborative Decision)  → context=[prior_tasks] chaining
  Step 5 (Output Generation)       → saves blueprint .md + returns dict
"""

import os
import json
from datetime import datetime
from crewai import Crew, Process

from agents import create_agents
from tasks import create_tasks
from llm_factory import get_llm, get_llm_info
import config


# ── Progress callback type ────────────────────────────────────────
# Passed in from main.py so the API can stream progress to the frontend
ProgressCallback = callable  # fn(agent_name: str, status: str, content: str)


def build_startup_plan(
    startup_idea: str,
    on_progress: ProgressCallback = None,
) -> dict:
    """
    Runs the full 4-agent pipeline for a given startup idea.

    Args:
        startup_idea:  Plain-English startup description
        on_progress:   Optional callback(agent_name, status, content)
                       Called when each agent starts/finishes.
                       Used by the streaming API endpoint.

    Returns:
        dict with all agent outputs + metadata
    """
    config.validate()
    llm = get_llm()
    llm_info = get_llm_info()

    _emit(on_progress, "system", "started", f"🚀 Starting analysis for: {startup_idea}")

    # ── Step 3: Create agents ─────────────────────────────────────
    agents = create_agents(llm)
    tasks  = create_tasks(agents, startup_idea)

    _emit(on_progress, "system", "agents_ready", "🤖 4 agents assembled and ready")

    # ── Step 4: Assemble crew (collaborative pipeline) ────────────
    # Process.sequential = each agent's output feeds into the next
    # This implements "Agents share insights and combine results"
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=False,
        # step_callback fires after every agent action
        step_callback=lambda step: _handle_step(step, on_progress),
    )

    # ── Fire! ─────────────────────────────────────────────────────
    _emit(on_progress, "product_manager", "started", "🧭 Product Manager is defining your MVP...")
    result = crew.kickoff()

    # ── Step 5: Package outputs ───────────────────────────────────
    task_outputs = result.tasks_output

    agent_names = ["product_manager", "market_researcher", "financial_planner", "developer"]
    agent_labels = {
        "product_manager":   "🧭 Product Manager",
        "market_researcher": "📊 Market Research",
        "financial_planner": "💰 Financial Planner",
        "developer":         "⚙️ Developer",
    }

    outputs = {}
    for i, name in enumerate(agent_names):
        raw = task_outputs[i].raw if i < len(task_outputs) else ""
        outputs[f"{name}_output"] = raw
        _emit(on_progress, name, "done", raw)

    # ── Step 5: Save blueprint to disk ────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_idea  = "".join(c if c.isalnum() else "_" for c in startup_idea[:30])
    md_path    = os.path.join(config.OUTPUT_DIR, f"{timestamp}_{safe_idea}.md")
    json_path  = os.path.join(config.OUTPUT_DIR, f"{timestamp}_{safe_idea}.json")

    # Save markdown report
    with open(md_path, "w") as f:
        f.write(f"# 🚀 Startup Blueprint: {startup_idea}\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Model: {llm_info['model']}*\n\n---\n\n")
        for name in agent_names:
            f.write(f"## {agent_labels[name]}\n\n{outputs[name + '_output']}\n\n---\n\n")

    # Save JSON for API consumers
    full_output = {
        "startup_idea":    startup_idea,
        "generated_at":    datetime.now().isoformat(),
        "llm_info":        llm_info,
        "pm_output":       outputs["product_manager_output"],
        "market_output":   outputs["market_researcher_output"],
        "finance_output":  outputs["financial_planner_output"],
        "dev_output":      outputs["developer_output"],
        "full_report":     str(result),
        "report_path":     md_path,
    }

    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=2)

    _emit(on_progress, "system", "complete", f"✅ Blueprint saved to {md_path}")
    print(f"\n✅  Reports saved:\n   MD:   {md_path}\n   JSON: {json_path}")

    return full_output


# ── Helpers ───────────────────────────────────────────────────────

def _emit(callback, agent: str, status: str, content: str):
    """Safely call the progress callback if provided."""
    if callback:
        try:
            callback(agent, status, content)
        except Exception:
            pass  # never let callback errors crash the crew


def _handle_step(step, callback):
    """Called by CrewAI after every agent reasoning step."""
    if not callback:
        return
    try:
        agent_role = getattr(step, "agent", {})
        if hasattr(agent_role, "role"):
            agent_role = agent_role.role.lower().replace(" ", "_")
        thought = getattr(step, "output", "") or getattr(step, "thought", "")
        if thought:
            _emit(callback, str(agent_role), "thinking", str(thought)[:300])
    except Exception:
        pass


# ── CLI runner ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    idea = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AI fitness app using sleep and DNA data"

    def print_progress(agent, status, content):
        icons = {"started": "▶", "thinking": "💭", "done": "✅", "complete": "🏁", "agents_ready": "🤖", "system": "⚙"}
        icon = icons.get(status, "•")
        print(f"\n{icon} [{agent.upper()}] {status}")
        if status in ("done", "complete"):
            print(content[:500] + ("..." if len(content) > 500 else ""))

    result = build_startup_plan(idea, on_progress=print_progress)

    print("\n" + "═" * 60)
    print("📋  DONE — Blueprint files:")
    print(f"   {result['report_path']}")