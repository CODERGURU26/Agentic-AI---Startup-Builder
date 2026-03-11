"""
config.py - Centralized config with LLM provider selection

Supports:
  - OpenAI  (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
  - Anthropic Claude (claude-3-5-sonnet, claude-3-haiku)
  
The provider is auto-detected from which API key is present.
If both are set, PREFERRED_PROVIDER in .env decides.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Model config ──────────────────────────────────────────────
PREFERRED_PROVIDER = os.getenv("PREFERRED_PROVIDER", "openai").lower()  # "openai" | "anthropic"

# Default models per provider
DEFAULT_MODELS = {
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
}

MODEL_NAME = os.getenv("MODEL_NAME") or DEFAULT_MODELS.get(PREFERRED_PROVIDER, "gpt-4o-mini")

# ── App config ────────────────────────────────────────────────
APP_ENV    = os.getenv("APP_ENV", "development")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
PORT       = int(os.getenv("PORT", 8000))


def get_active_provider() -> str:
    """Returns which provider will actually be used."""
    if PREFERRED_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        return "anthropic"
    if OPENAI_API_KEY:
        return "openai"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    return "none"


def validate():
    """Call at startup — raises if no key is configured."""
    provider = get_active_provider()
    if provider == "none":
        raise EnvironmentError(
            "\n❌  No API key found!\n"
            "    Copy .env.example → .env and add either:\n"
            "      OPENAI_API_KEY=sk-...\n"
            "    or\n"
            "      ANTHROPIC_API_KEY=sk-ant-...\n"
        )
    print(f"✅  Config OK | provider={provider} | model={MODEL_NAME} | env={APP_ENV}")
    return provider