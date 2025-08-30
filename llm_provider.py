from __future__ import annotations
import os, logging
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

load_dotenv()
log = logging.getLogger("fraud-analyst")
logging.basicConfig(level=logging.INFO)

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY") or os.getenv("HF_TOKEN")
FW_PRIMARY_MODEL   = os.getenv("FW_PRIMARY_MODEL",   "openai/gpt-oss-20b")
FW_SECONDARY_MODEL = os.getenv("FW_SECONDARY_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

SUMMARY_NOTICE = "🔌 Please connect to an inference point to generate summary."

class FireworksHFChat(BaseChatModel):
    model: str
    api_key: str | None = None
    temperature: float = 0.2
    max_new_tokens: int = 256
    timeout: int = 60

    def __init__(self, model: str, api_key: str | None):
        super().__init__()
        self.model = model
        self.api_key = api_key
        self._client = InferenceClient(provider="fireworks-ai", api_key=self.api_key)

    @property
    def _llm_type(self) -> str:
        return "fireworks_hf_chat"

    def _convert(self, messages):
        out=[]
        for m in messages:
            if isinstance(m, SystemMessage):
                out.append({"role":"system","content":m.content})
            elif isinstance(m, HumanMessage):
                out.append({"role":"user","content":m.content})
            elif isinstance(m, AIMessage):
                out.append({"role":"assistant","content":m.content})
            else:
                out.append({"role":"user","content":str(getattr(m,"content",m))})
        return out

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if not self.api_key:
            gen = ChatGeneration(message=AIMessage(content=""))
            return ChatResult(generations=[gen], llm_output={"error": "no_api_key"})
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=self._convert(messages),
                stream=False,
                max_tokens=kwargs.get("max_tokens", 256),
                temperature=kwargs.get("temperature", 0.2),
            )
            text = ""
            if hasattr(resp, "choices") and resp.choices:
                ch = resp.choices[0]
                if hasattr(ch, "message") and ch.message and getattr(ch.message, "content", None):
                    text = ch.message.content
                elif hasattr(ch, "text") and ch.text:
                    text = ch.text
            gen = ChatGeneration(message=AIMessage(content=text or ""))
            return ChatResult(generations=[gen], llm_output={"model": self.model})
        except Exception as e:
            log.warning(f"Fireworks call failed for {self.model}: {type(e).__name__}: {str(e)[:200]}")
            gen = ChatGeneration(message=AIMessage(content=""))
            return ChatResult(generations=[gen], llm_output={"error": str(e)})

def _heartbeat(model_id: str) -> bool:
    if not FIREWORKS_API_KEY: return False
    try:
        client = InferenceClient(provider="fireworks-ai", api_key=FIREWORKS_API_KEY)
        _ = client.chat.completions.create(
            model=model_id,
            messages=[{"role":"user","content":"ping"}],
            stream=False,
            max_tokens=1,
        )
        return True
    except Exception as e:
        log.warning(f"Heartbeat failed for {model_id}: {type(e).__name__}: {str(e)[:160]}")
        return False

def build_chat_llm():
    log.info(f"Fireworks key present: {bool(FIREWORKS_API_KEY)} len={len(FIREWORKS_API_KEY) if FIREWORKS_API_KEY else 0}")
    if FIREWORKS_API_KEY and _heartbeat(FW_PRIMARY_MODEL):
        log.info(f"Using chat model: {FW_PRIMARY_MODEL}")
        return FireworksHFChat(FW_PRIMARY_MODEL, FIREWORKS_API_KEY)
    if FIREWORKS_API_KEY and _heartbeat(FW_SECONDARY_MODEL):
        log.info(f"Using fallback chat model: {FW_SECONDARY_MODEL}")
        return FireworksHFChat(FW_SECONDARY_MODEL, FIREWORKS_API_KEY)
    log.warning("No working chat model; notice will be shown.")
    return None

CHAT_LLM = build_chat_llm()