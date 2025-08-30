from __future__ import annotations
import os, logging
from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Providers
from openai import OpenAI  # Fireworks OpenAI-compatible API
from huggingface_hub import InferenceClient  # HF Router (provider routing)

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

# --- Secrets (your HF repo secret name included) ---
FIREWORKS_API_KEY = _first_env(
    "fireworks_api_huggingface",      # your repo secret
    "FIREWORKS_API_HUGGINGFACE",
    "FIREWORKS_API_KEY",
    "OPENAI_API_KEY"                  # okay to reuse for FW key if you prefer
)
HF_TOKEN = _first_env("HF_TOKEN", "HUGGINGFACE_TOKEN")

# --- Model IDs ---
# Fireworks (OpenAI-compatible; MUST use the accounts/fireworks path)
FW_PRIMARY_MODEL   = os.getenv("FW_PRIMARY_MODEL",   "accounts/fireworks/models/gpt-oss-20b")
FW_SECONDARY_MODEL = os.getenv("FW_SECONDARY_MODEL", "accounts/fireworks/models/qwen3-coder-30b-a3b-instruct")

# HF Router (Hugging Face token + provider routing)
# Use huggingface model slugs; they’ll be routed to Fireworks via provider="fireworks-ai"
HF_PRIMARY_MODEL   = os.getenv("HF_PRIMARY_MODEL",   "fireworks/gpt-oss-20b")
HF_SECONDARY_MODEL = os.getenv("HF_SECONDARY_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

# ============================== Fireworks (OpenAI-compatible) ==============================
class FireworksOpenAIChat(BaseChatModel):
    model: str
    api_key: str | None = None
    temperature: float = 0.2
    max_new_tokens: int = 256

    def __init__(self, **data):
        super().__init__(**data)
        self._client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE", "https://api.fireworks.ai/inference/v1"),
            api_key=self.api_key,
        )

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
        try:
            resp = self._client.chat.completions.create(
                model=self.model,  # e.g., accounts/fireworks/models/gpt-oss-20b
                messages=self._convert(messages),
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_new_tokens),
                stream=False,
            )
            text = ""
            if getattr(resp, "choices", None):
                ch = resp.choices[0]
                if getattr(ch, "message", None) and getattr(ch.message, "content", None):
                    text = ch.message.content
            gen = ChatGeneration(message=AIMessage(content=text or ""))
            return ChatResult(generations=[gen], llm_output={"model": self.model})
        except Exception as e:
            log.warning(f"Fireworks(OpenAI) call failed for {self.model}: {type(e).__name__}: {str(e)[:200]}")
            gen = ChatGeneration(message=AIMessage(content=""))
            return ChatResult(generations=[gen], llm_output={"error": str(e)})

def _heartbeat_fireworks(model_id: str) -> bool:
    if not FIREWORKS_API_KEY: return False
    try:
        cli = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=FIREWORKS_API_KEY)
        _ = cli.chat.completions.create(model=model_id, messages=[{"role":"user","content":"ping"}], max_tokens=1)
        return True
    except Exception as e:
        log.warning(f"FW heartbeat failed for {model_id}: {type(e).__name__}: {str(e)[:200]}")
        return False

# ============================== HF Router (provider routing) ==============================
class HFRouterChat(BaseChatModel):
    model: str
    hf_token: str | None = None
    temperature: float = 0.2
    max_new_tokens: int = 256

    def __init__(self, **data):
        super().__init__(**data)
        # Use HF Router to reach Fireworks; requires HF_TOKEN
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
        try:
            resp = self._client.chat.completions.create(
                model=self.model,  # e.g., "fireworks/gpt-oss-20b"
                messages=self._convert(messages),
                stream=False,
                max_tokens=kwargs.get("max_tokens", self.max_new_tokens),
                temperature=kwargs.get("temperature", self.temperature),
            )
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
    except Exception as e:
        log.warning(f"HF Router heartbeat failed for {model_id}: {type(e).__name__}: {str(e)[:200]}")
        return False

# ============================== Selection ==============================
def build_chat_llm():
    # Prefer direct Fireworks with your Fireworks key
    if FIREWORKS_API_KEY and _heartbeat_fireworks(FW_PRIMARY_MODEL):
        log.info(f"Using Fireworks chat model: {FW_PRIMARY_MODEL}")
        return FireworksOpenAIChat(model=FW_PRIMARY_MODEL, api_key=FIREWORKS_API_KEY)
    if FIREWORKS_API_KEY and _heartbeat_fireworks(FW_SECONDARY_MODEL):
        log.info(f"Using Fireworks fallback chat model: {FW_SECONDARY_MODEL}")
        return FireworksOpenAIChat(model=FW_SECONDARY_MODEL, api_key=FIREWORKS_API_KEY)

    # Fallback to HF Router (requires HF_TOKEN)
    if HF_TOKEN and _heartbeat_hf_router(HF_PRIMARY_MODEL):
        log.info(f"Using HF Router chat model: {HF_PRIMARY_MODEL}")
        return HFRouterChat(model=HF_PRIMARY_MODEL, hf_token=HF_TOKEN)
    if HF_TOKEN and _heartbeat_hf_router(HF_SECONDARY_MODEL):
        log.info(f"Using HF Router fallback chat model: {HF_SECONDARY_MODEL}")
        return HFRouterChat(model=HF_SECONDARY_MODEL, hf_token=HF_TOKEN)

    log.warning("No working chat model; notice will be shown.")
    return None

CHAT_LLM = build_chat_llm()