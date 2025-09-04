from __future__ import annotations
import os, time, random, threading, logging
from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from openai import OpenAI
from openai import RateLimitError, BadRequestError
from huggingface_hub import InferenceClient

load_dotenv()
log = logging.getLogger("fraud-analyst")
logging.basicConfig(level=logging.INFO)

SUMMARY_NOTICE = "🔌 Please connect to an inference point to generate summary."

def _first_env(*names):
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None

# ---------- Secrets ----------
FIREWORKS_API_KEY = _first_env(
    "fireworks_api_huggingface",
    "FIREWORKS_API_HUGGINGFACE",
    "FIREWORKS_API_KEY",
    "OPENAI_API_KEY",
)
HF_TOKEN = _first_env("HF_TOKEN", "HUGGINGFACE_TOKEN")

# ---------- Model IDs ----------
# Fireworks (OpenAI-compatible)
FW_PRIMARY_MODEL   = os.getenv("FW_PRIMARY_MODEL",   "accounts/fireworks/models/gpt-oss-20b")
FW_SECONDARY_MODEL = os.getenv("FW_SECONDARY_MODEL", "accounts/fireworks/models/qwen3-coder-30b-a3b-instruct")

# HF Router (if you also set HF_TOKEN)
HF_PRIMARY_MODEL   = os.getenv("HF_PRIMARY_MODEL",   "fireworks/gpt-oss-20b")
HF_SECONDARY_MODEL = os.getenv("HF_SECONDARY_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

# ---------- Throttle / Retry ----------
MAX_NEW_TOKENS   = int(os.getenv("LLM_MAX_NEW_TOKENS", "128"))
TEMP             = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_RETRIES      = int(os.getenv("LLM_MAX_RETRIES", "3"))
MIN_INTERVAL_S   = float(os.getenv("LLM_MIN_INTERVAL_S", "0.25"))
MAX_CONCURRENCY  = int(os.getenv("LLM_MAX_CONCURRENCY", "1"))

_CALL_LOCK = threading.BoundedSemaphore(MAX_CONCURRENCY)
_last_call_ts = 0.0
_ts_lock = threading.Lock()

def _pace():
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
        except RateLimitError as e:
            if attempt >= MAX_RETRIES:
                raise
            # jittered backoff
            time.sleep(0.5 * attempt + random.random() * 0.4)
        except Exception:
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(0.3 * attempt)

# ============================== Fireworks via OpenAI SDK ==============================
FW_BASE = os.getenv("OPENAI_API_BASE", "https://api.fireworks.ai/inference/v1")

class FireworksOpenAIChat(BaseChatModel):
    """
    Primary driver: uses Fireworks /chat/completions.
    If the model errors as chat-incompatible, we raise a sentinel to try the completion driver.
    """
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
            gen = ChatGeneration(message=AIMessage(content=""))
            return ChatResult(generations=[gen], llm_output={"error":"no_api_key"})
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
            gen = ChatGeneration(message=AIMessage(content=text or ""))
            return ChatResult(generations=[gen], llm_output={"model": self.model})
        except BadRequestError as e:
            # Known symptom for models that prefer /completions:
            # "Failed to format non-streaming choice: Unexpected EOS while waiting for..."
            msg = str(e)
            if "Failed to format non-streaming choice" in msg or "invalid_request_error" in msg:
                raise _ChatIncompatible()
            log.warning(f"FW chat BadRequest for {self.model}: {msg[:200]}")
            gen = ChatGeneration(message=AIMessage(content=""))
            return ChatResult(generations=[gen], llm_output={"error": msg})
        except Exception as e:
            log.warning(f"FW chat failed for {self.model}: {type(e).__name__}: {str(e)[:200]}")
            gen = ChatGeneration(message=AIMessage(content=""))
            return ChatResult(generations=[gen], llm_output={"error": str(e)})

class _ChatIncompatible(Exception):
    pass

class FireworksOpenAICompletionChat(BaseChatModel):
    """
    Fallback driver: uses Fireworks /completions and wraps it as a chat model.
    Works for models like gpt-oss-20b that may not behave with chat-completions.
    """
    model: str
    api_key: str | None = None
    temperature: float = TEMP
    max_new_tokens: int = MAX_NEW_TOKENS

    def __init__(self, **data):
        super().__init__(**data)
        self._client = OpenAI(base_url=FW_BASE, api_key=self.api_key, max_retries=0)

    @property
    def _llm_type(self) -> str:
        return "fireworks_openai_completion_chat"

    def _to_prompt(self, messages) -> str:
        # Simple, robust conversion to a prompt
        parts=[]
        for m in messages:
            if isinstance(m, SystemMessage):
                parts.append(f"[System] {m.content}")
            elif isinstance(m, HumanMessage):
                parts.append(f"[User] {m.content}")
            elif isinstance(m, AIMessage):
                parts.append(f"[Assistant] {m.content}")
            else:
                parts.append(f"[User] {str(getattr(m,'content',m))}")
        parts.append("[Assistant]")
        return "\n".join(parts)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if not self.api_key:
            gen = ChatGeneration(message=AIMessage(content=""))
            return ChatResult(generations=[gen], llm_output={"error":"no_api_key"})
        prompt = self._to_prompt(messages)
        def _call():
            with _CALL_LOCK:
                _pace()
                return self._client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_new_tokens),
                )
        try:
            resp = _with_retries(_call)
            text = ""
            if getattr(resp, "choices", None):
                ch = resp.choices[0]
                if getattr(ch, "text", None):
                    text = ch.text
            gen = ChatGeneration(message=AIMessage(content=text or ""))
            return ChatResult(generations=[gen], llm_output={"model": self.model, "endpoint":"completions"})
        except Exception as e:
            log.warning(f"FW completion failed for {self.model}: {type(e).__name__}: {str(e)[:200]}")
            gen = ChatGeneration(message=AIMessage(content=""))
            return ChatResult(generations=[gen], llm_output={"error": str(e)})

def _heartbeat_fw_chat(model_id: str) -> bool:
    if not FIREWORKS_API_KEY: return False
    try:
        cli = OpenAI(base_url=FW_BASE, api_key=FIREWORKS_API_KEY, max_retries=0)
        _ = cli.chat.completions.create(model=model_id, messages=[{"role":"user","content":"ping"}], max_tokens=1)
        return True
    except BadRequestError as e:
        # Treat as chat-incompatible but alive
        if "Failed to format non-streaming choice" in str(e) or "invalid_request_error" in str(e):
            return False  # signal to try completion heartbeat
        return False
    except Exception:
        return False

def _heartbeat_fw_completion(model_id: str) -> bool:
    if not FIREWORKS_API_KEY: return False
    try:
        cli = OpenAI(base_url=FW_BASE, api_key=FIREWORKS_API_KEY, max_retries=0)
        _ = cli.completions.create(model=model_id, prompt="ping", max_tokens=1)
        return True
    except Exception:
        return False

# ============================== HF Router (provider routing) ==============================
class HFRouterChat(BaseChatModel):
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
            gen = ChatGeneration(message=AIMessage(content=""))
            return ChatResult(generations=[gen], llm_output={"error":"no_hf_token"})
        def _call():
            with _CALL_LOCK:
                _pace()
                return self._client.chat.completions.create(
                    model=self.model,
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
            gen = ChatGeneration(message=AIMessage(content=text or ""))
            return ChatResult(generations=[gen], llm_output={"model": self.model})
        except Exception as e:
            log.warning(f"HF Router call failed for {self.model}: {type(e).__name__}: {str(e)[:200]}")
            gen = ChatGeneration(message=AIMessage(content=""))
            return ChatResult(generations=[gen], llm_output={"error": str(e)})

def _heartbeat_hf_router(model_id: str) -> bool:
    if not HF_TOKEN: return False
    try:
        cli = InferenceClient(provider="fireworks-ai", api_key=HF_TOKEN)
        _ = cli.chat.completions.create(model=model_id, messages=[{"role":"user","content":"ping"}], stream=False, max_tokens=1)
        return True
    except Exception:
        return False

# ============================== Selection ==============================
def build_chat_llm():
    # Prefer Fireworks direct
    if FIREWORKS_API_KEY:
        # Try chat first; if that fails, try completion wrapper
        if _heartbeat_fw_chat(FW_PRIMARY_MODEL):
            log.info(f"Using Fireworks chat model: {FW_PRIMARY_MODEL}")
            return FireworksOpenAIChat(model=FW_PRIMARY_MODEL, api_key=FIREWORKS_API_KEY)
        elif _heartbeat_fw_completion(FW_PRIMARY_MODEL):
            log.info(f"Using Fireworks COMPLETION-wrapped model: {FW_PRIMARY_MODEL}")
            return FireworksOpenAICompletionChat(model=FW_PRIMARY_MODEL, api_key=FIREWORKS_API_KEY)

        # Secondary
        if _heartbeat_fw_chat(FW_SECONDARY_MODEL):
            log.info(f"Using Fireworks chat model (fallback): {FW_SECONDARY_MODEL}")
            return FireworksOpenAIChat(model=FW_SECONDARY_MODEL, api_key=FIREWORKS_API_KEY)
        elif _heartbeat_fw_completion(FW_SECONDARY_MODEL):
            log.info(f"Using Fireworks COMPLETION-wrapped model (fallback): {FW_SECONDARY_MODEL}")
            return FireworksOpenAICompletionChat(model=FW_SECONDARY_MODEL, api_key=FIREWORKS_API_KEY)

    # HF Router fallback
    if HF_TOKEN and _heartbeat_hf_router(HF_PRIMARY_MODEL):
        log.info(f"Using HF Router chat model: {HF_PRIMARY_MODEL}")
        return HFRouterChat(model=HF_PRIMARY_MODEL, hf_token=HF_TOKEN)
    if HF_TOKEN and _heartbeat_hf_router(HF_SECONDARY_MODEL):
        log.info(f"Using HF Router chat model (fallback): {HF_SECONDARY_MODEL}")
        return HFRouterChat(model=HF_SECONDARY_MODEL, hf_token=HF_TOKEN)

    log.warning("No working chat model; notice will be shown.")
    return None

CHAT_LLM = build_chat_llm()