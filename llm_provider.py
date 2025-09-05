from __future__ import annotations
"""
llm_provider.py
- Single provider: Fireworks (OpenAI-compatible) with Qwen3-Coder-30B-A3B-Instruct
- Optional fallback: Hugging Face Inference Router (provider="fireworks-ai") if HF_TOKEN present
- No heartbeats (avoid rate hits); conservative throttling & retries
"""

import os, time, random, threading, logging
from typing import List

from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Fireworks (OpenAI-compatible)
from openai import OpenAI
from openai import RateLimitError

# HF Router (provider routing)
from huggingface_hub import InferenceClient

load_dotenv()
log = logging.getLogger("fraud-analyst")
logging.basicConfig(level=logging.INFO)

SUMMARY_NOTICE = "🔌 Please connect to an inference point to generate summary."

def _first_env(*names: List[str]):
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None

# Secrets
FIREWORKS_API_KEY = _first_env(
    "fireworks_api_huggingface", "FIREWORKS_API_HUGGINGFACE",
    "FIREWORKS_API_KEY", "OPENAI_API_KEY"
)
HF_TOKEN = _first_env("HF_TOKEN", "HUGGINGFACE_TOKEN")

# Models (Qwen only)
FW_PRIMARY_MODEL = os.getenv("FW_PRIMARY_MODEL", "accounts/fireworks/models/qwen3-coder-30b-a3b-instruct")
HF_PRIMARY_MODEL = os.getenv("HF_PRIMARY_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

# Throttle / Retry (conservative for demo)
MAX_NEW_TOKENS  = int(os.getenv("LLM_MAX_NEW_TOKENS", "96"))
TEMP            = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_RETRIES     = int(os.getenv("LLM_MAX_RETRIES", "2"))
MIN_INTERVAL_S  = float(os.getenv("LLM_MIN_INTERVAL_S", "1.0"))
MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "1"))

# Ensure OpenAI SDK itself doesn't also retry
os.environ.setdefault("OPENAI_MAX_RETRIES", "0")

# Global throttle across all instances
_CALL_LOCK = threading.BoundedSemaphore(MAX_CONCURRENCY)
_last_call_ts = 0.0
_ts_lock = threading.Lock()

def _pace():
    """Global pacing to avoid hitting 429s."""
    global _last_call_ts
    with _ts_lock:
        now = time.monotonic()
        dt = now - _last_call_ts
        if dt < MIN_INTERVAL_S:
            time.sleep(MIN_INTERVAL_S - dt)
        _last_call_ts = time.monotonic()

def _with_retries(fn):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except RateLimitError:
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(0.6 * attempt + random.random() * 0.5)
        except Exception:
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(0.4 * attempt)

# ========================== Fireworks (OpenAI-compatible) ==========================
FW_BASE = os.getenv("OPENAI_API_BASE", "https://api.fireworks.ai/inference/v1")

class FireworksOpenAIChat(BaseChatModel):
    """Qwen on Fireworks via /chat/completions."""
    model: str
    api_key: str | None = None
    temperature: float = TEMP
    max_new_tokens: int = MAX_NEW_TOKENS

    def __init__(self, **data):
        super().__init__(**data)
        self._client = OpenAI(base_url=FW_BASE, api_key=self.api_key, max_retries=0)

    @property
    def _llm_type(self) -> str:
        return "fireworks_openai_chat"

    def _convert(self, messages):
        out=[]
        for m in messages:
            if isinstance(m, SystemMessage): out.append({"role":"system","content":m.content})
            elif isinstance(m, HumanMessage): out.append({"role":"user","content":m.content})
            elif isinstance(m, AIMessage): out.append({"role":"assistant","content":m.content})
            else: out.append({"role":"user","content":str(getattr(m,"content",m))})
        return out

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if not self.api_key:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))],
                              llm_output={"error": "no_api_key"})
        def _call():
            with _CALL_LOCK:
                _pace()
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=self._convert(messages),
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_new_tokens),
                    stream=False,
                )
        try:
            resp = _with_retries(_call)
            text = ""
            if getattr(resp, "choices", None):
                ch = resp.choices[0]
                if getattr(ch, "message", None) and getattr(ch.message, "content", None):
                    text = ch.message.content
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text or ""))],
                              llm_output={"model": self.model, "endpoint": "chat"})
        except Exception as e:
            # Return empty output; UI will show notice if needed
            logging.warning(f"Fireworks(Qwen) failed: {type(e).__name__}: {str(e)[:200]}")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))],
                              llm_output={"error": str(e)})

# ========================== HF Router (provider="fireworks-ai") ==========================
class HFRouterChat(BaseChatModel):
    """Fallback only if FIREWORKS_API_KEY is absent but HF_TOKEN is set."""
    model: str
    hf_token: str | None = None
    temperature: float = TEMP
    max_new_tokens: int = MAX_NEW_TOKENS

    def __init__(self, **data):
        super().__init__(**data)
        self._client = InferenceClient(provider="fireworks-ai", api_key=self.hf_token)

    @property
    def _llm_type(self) -> str:
        return "hf_router_fireworks"

    def _convert(self, messages):
        out=[]
        for m in messages:
            if isinstance(m, SystemMessage): out.append({"role":"system","content":m.content})
            elif isinstance(m, HumanMessage): out.append({"role":"user","content":m.content})
            elif isinstance(m, AIMessage): out.append({"role":"assistant","content":m.content})
            else: out.append({"role":"user","content":str(getattr(m,"content",m))})
        return out

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if not self.hf_token:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))],
                              llm_output={"error":"no_hf_token"})
        def _call():
            with _CALL_LOCK:
                _pace()
                return self._client.chat.completions.create(
                    model=self.model,  # "Qwen/Qwen3-Coder-30B-A3B-Instruct"
                    messages=self._convert(messages),
                    stream=False,
                    max_tokens=kwargs.get("max_tokens", self.max_new_tokens),
                    temperature=kwargs.get("temperature", self.temperature),
                )
        try:
            resp = _with_retries(_call)
            text = ""
            if getattr(resp, "choices", None):
                ch = resp.choices[0]
                if getattr(ch, "message", None) and getattr(ch.message, "content", None):
                    text = ch.message.content
                elif getattr(ch, "text", None):
                    text = ch.text
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text or ""))],
                              llm_output={"model": self.model})
        except Exception as e:
            logging.warning(f"HF Router(Qwen) failed: {type(e).__name__}: {str(e)[:200]}")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))],
                              llm_output={"error": str(e)})

# =============================== Selection ===============================
def build_chat_llm():
    # Prefer Fireworks direct (Qwen)
    if FIREWORKS_API_KEY:
        log.info(f"Using Fireworks chat model: {FW_PRIMARY_MODEL}")
        return FireworksOpenAIChat(model=FW_PRIMARY_MODEL, api_key=FIREWORKS_API_KEY)

    # Fallback to HF Router (if provided)
    if HF_TOKEN:
        log.info(f"Using HF Router chat model: {HF_PRIMARY_MODEL}")
        return HFRouterChat(model=HF_PRIMARY_MODEL, hf_token=HF_TOKEN)

    log.warning("No working chat model; notice will be shown.")
    return None

CHAT_LLM = build_chat_llm()
