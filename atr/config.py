"""ATR runtime configuration: LLM backbones + offline / online hyper-parameters.

Backbones are addressed by string key (the ``--backbone`` flag); each entry in
``config_mapping`` is the kwargs passed to ``atr.clients.chat_utils.get_chat_result``.
All credentials / endpoints are env-var-overridable so the same code runs on a
laptop, a workstation, or a managed deployment.

Headline experiments in the paper use ``gemini-2.5-flash`` (the default). The
cross-backbone matrix (Table 4) additionally uses ``gemini-2.5-pro``,
``claude-haiku-45``, ``qwen35``, ``llama4scout``, and ``llama33``. The
remaining keys are provided for convenience.
"""
import json
import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# Vertex AI credentials
# ──────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VERTEX_CREDENTIALS_PATH = _REPO_ROOT / "vertexai.json"
DEFAULT_VERTEX_LOCATION = os.getenv("VERTEXAI_LOCATION", "us-central1")


def _load_project_id(credentials_path: Path) -> str:
    with open(credentials_path, "r", encoding="utf-8") as f:
        creds = json.load(f)
    project_id = creds.get("project_id")
    if not project_id:
        raise ValueError(f"Missing 'project_id' in {credentials_path}")
    return project_id


vertex_credentials_path = Path(
    os.getenv("VERTEXAI_CREDENTIALS_PATH", str(DEFAULT_VERTEX_CREDENTIALS_PATH))
).expanduser()


def _vertex_openai_compat(model: str) -> dict:
    return {
        "provider": "vertex_openai_compat",
        "url": (
            f"https://{DEFAULT_VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/"
            f"{vertex_project_id}/locations/{DEFAULT_VERTEX_LOCATION}/endpoints/openapi"
        ),
        "model": model,
        "credentials_path": str(vertex_credentials_path),
    }


# Fallback config used when vertexai.json is unavailable (e.g. running tests
# without Vertex creds). Backbones that depend on Vertex will silently fall
# back to this and fail at call time with a clear OpenAI-style error.
_openai_fallback = {
    "provider": "openai",
    "url": os.getenv("LLM_URL", ""),
    "model": os.getenv("LLM_MODEL", "gpt-4o"),
    "api_key": os.getenv("OPENAI_API_KEY", ""),
}

try:
    vertex_project_id = _load_project_id(vertex_credentials_path)
    _vertex_flash = _vertex_openai_compat(
        os.getenv("VERTEXAI_MODEL", "google/gemini-2.5-flash")
    )
    _vertex_pro = _vertex_openai_compat(
        os.getenv("VERTEXAI_PRO_MODEL", "google/gemini-2.5-pro")
    )
    _vertex_flashlite = _vertex_openai_compat("google/gemini-2.5-flash-lite")
except Exception:
    _vertex_flash = _vertex_pro = _vertex_flashlite = _openai_fallback


def _anthropic_vertex(model: str, thinking_effort: str | None = None) -> dict:
    cfg = {
        "provider": "anthropic_vertex",
        "model": model,
        "region": os.getenv("ANTHROPIC_VERTEX_REGION", "global"),
        "credentials_path": str(vertex_credentials_path),
    }
    if thinking_effort:
        cfg["thinking_effort"] = thinking_effort
    return cfg


# ──────────────────────────────────────────────────────────────────────────
# config_mapping: register each backbone under one or more aliases.
# Paper-headline backbones are flagged with [paper] in the comment;
# the rest are convenience / extension points.
# ──────────────────────────────────────────────────────────────────────────
config_mapping = {
    # ─── Google Vertex Gemini ────────────────────────────────────────────
    "gemini":                _vertex_flash,                     # [paper] default
    "gemini-2.5-flash":      _vertex_flash,                     # [paper] alias
    "v3":                    _vertex_flash,                     # legacy alias
    "gemini-pro":            _vertex_pro,                       # [paper] Table 4
    "gemini-2.5-pro":        _vertex_pro,                       # [paper] alias
    "gemini-flashlite":      _vertex_flashlite,                 # cheap fallback
    "gemini-2.5-flash-lite": _vertex_flashlite,                 # alias

    # ─── Anthropic Claude on Vertex ──────────────────────────────────────
    "claude-haiku-45":       _anthropic_vertex("claude-haiku-4-5"),        # [paper] Table 4
    "claude-opus-47":        _anthropic_vertex("claude-opus-4-7"),
    "claude-opus-47-thinking": _anthropic_vertex(
        "claude-opus-4-7",
        thinking_effort=os.getenv("CLAUDE_THINKING_EFFORT", "high"),
    ),

    # ─── OpenAI (managed) ────────────────────────────────────────────────
    "gpt_mini": {
        "provider": "openai",
        "url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.getenv("OPENAI_MINI_MODEL", "gpt-4o-mini"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.getenv("OPENAI_MINI_MODEL", "gpt-4o-mini"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
    },

    # ─── Local vLLM endpoints (open-weight backbones) ────────────────────
    # Override the URLs to point at your vLLM server, e.g.
    #   export VLLM_QWEN35_URL=http://my-gpu-box:8000/v1
    "qwen35": {
        "provider": "openai",
        "url": os.getenv("VLLM_QWEN35_URL", "http://localhost:8000/v1"),
        "model": "Qwen3.5-122B-A10B-GPTQ-Int4",                # [paper] Table 4
        "api_key": "EMPTY",
    },
    "qwen3.5": {
        "provider": "openai",
        "url": os.getenv("VLLM_QWEN35_URL", "http://localhost:8000/v1"),
        "model": "Qwen3.5-122B-A10B-GPTQ-Int4",
        "api_key": "EMPTY",
    },
    "llama4scout": {
        "provider": "openai",
        "url": os.getenv("VLLM_LLAMA4_URL", "http://localhost:8000/v1"),
        "model": "Llama-4-Scout-17B-16E-quantized-w4a16",      # [paper] Table 4
        "api_key": "EMPTY",
    },
    "llama33": {
        "provider": "openai",
        "url": os.getenv("VLLM_LLAMA33_URL", "http://localhost:8000/v1"),
        "model": os.getenv("VLLM_LLAMA33_MODEL", "llama-3.3-70b-instruct-awq"),  # [paper] Table 4
        "api_key": "EMPTY",
    },
}

# ──────────────────────────────────────────────────────────────────────────
# Flask SQL service (View 5)
# ──────────────────────────────────────────────────────────────────────────
sql_service_url: str = os.getenv("SQL_SERVICE_URL", "").strip()

# ──────────────────────────────────────────────────────────────────────────
# Offline index settings
# ──────────────────────────────────────────────────────────────────────────
CELL_INDEX_BUDGET: int = int(os.getenv("CELL_INDEX_BUDGET", "10000"))
SCHEMA_TOP_K: int = int(os.getenv("SCHEMA_TOP_K", "5"))
CELL_TOP_K: int = int(os.getenv("CELL_TOP_K", "15"))   # K = 10–20 (§3.5)
ROW_TOP_K: int = int(os.getenv("ROW_TOP_K", "10"))
RERANK_CANDIDATE_MULTIPLIER: int = int(
    os.getenv("RERANK_CANDIDATE_MULTIPLIER", "6")
)

# ──────────────────────────────────────────────────────────────────────────
# Online inference settings
# ──────────────────────────────────────────────────────────────────────────
MAX_ITER: int = int(os.getenv("AGENTIC_MAX_ITER", "5"))
VERIFIER_THRESHOLD: float = float(os.getenv("VERIFIER_THRESHOLD", "0.1"))
