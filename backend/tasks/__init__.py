"""
tasks/__init__.py - Tasks for each agent (Step 4: Collaborative Decision Making)

Key pattern: context=[prior_tasks] — each agent sees what the others wrote.
This is what makes it "collaborative": agents build on each other's insights.
"""

from crewai import Task


def create_tasks(agents: dict, startup_idea: str) -> list:

    # ── Task 1: Product Vision ────────────────────────────────────
    task_product = Task(
        description=(
            f"Startup idea: '{startup_idea}'\n\n"
            "Deliver a structured report with these 5 sections:\n"
            "1. PRODUCT VISION — 2-sentence vision statement\n"
            "2. CORE MVP FEATURES — exactly 5 features with 1-line descriptions\n"
            "3. USER PERSONAS — 2 personas (name, pain point, how product helps)\n"
            "4. MVP SCOPE — what gets built in the first 3 months\n"
            "5. SUCCESS METRICS — 3 KPIs to track in Month 1\n\n"
            "Be opinionated. Cut anything non-essential for launch."
        ),
        expected_output="Structured report with 5 labeled sections.",
        agent=agents["product_manager"],
    )

    # ── Task 2: Market Research ───────────────────────────────────
    task_market = Task(
        description=(
            f"Startup idea: '{startup_idea}'\n\n"
            "Using the Product Manager's vision as context, deliver:\n"
            "1. MARKET SIZE — TAM/SAM/SOM with reasoning\n"
            "2. TOP 3 COMPETITORS — name, strength, weakness\n"
            "3. OUR EDGE — 3 specific differentiators\n"
            "4. TARGET CUSTOMER — demographics, behaviour, willingness to pay\n"
            "5. GO-TO-MARKET — 90-day acquisition strategy with channels\n\n"
            "Back every claim with reasoning."
        ),
        expected_output="Market analysis with 5 labeled sections.",
        agent=agents["market_researcher"],
        context=[task_product],
    )

    # ── Task 3: Financial Plan ────────────────────────────────────
    task_finance = Task(
        description=(
            f"Startup idea: '{startup_idea}'\n\n"
            "Using the Product and Market analysis as context, deliver:\n"
            "1. STARTUP COSTS — itemised list with USD amounts\n"
            "2. PRICING MODEL — recommended tiers with prices\n"
            "3. REVENUE PROJECTIONS — Year 1 / 2 / 3 with assumptions\n"
            "4. BURN RATE & RUNWAY — monthly burn and how long $50K seed lasts\n"
            "5. BREAK-EVEN — when does the business become profitable?\n\n"
            "Be conservative. State every assumption clearly."
        ),
        expected_output="Financial plan with 5 labeled sections.",
        agent=agents["financial_planner"],
        context=[task_product, task_market],
    )

    # ── Task 4: Technical Blueprint ───────────────────────────────
    task_dev = Task(
        description=(
            f"Startup idea: '{startup_idea}'\n\n"
            "Using the full context from all 3 prior agents, deliver:\n"
            "1. TECH STACK — frontend, backend, DB, auth, hosting (justify each)\n"
            "2. SYSTEM ARCHITECTURE — key components and how they interact\n"
            "3. SPRINT ROADMAP — 4 x 2-week sprints with specific deliverables\n"
            "4. TECHNICAL RISKS — top 3 with likelihood, impact, mitigation\n"
            "5. TEAM REQUIRED — roles needed (be realistic about headcount)\n\n"
            "Align the roadmap with the MVP scope defined by the Product Manager."
        ),
        expected_output="Technical blueprint with 5 labeled sections.",
        agent=agents["developer"],
        context=[task_product, task_market, task_finance],
    )

    return [task_product, task_market, task_finance, task_dev]