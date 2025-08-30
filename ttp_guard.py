from __future__ import annotations
import re, os
from dataclasses import dataclass
from typing import Dict, Any, List

class GuardDecision:
    ALLOW = "allow"
    BLOCK = "block"
    ANNOTATE = "annotate"

@dataclass
class GuardResult:
    action: str
    reason: str
    indicators: List[str]

class TTPGuard:
    """
    Lightweight rule-based guard for adversarial TTPs:
    - Prompt injection / instruction override (e.g., "ignore previous instructions", "you are DAN")
    - Safety bypass ("never refuse", "no moralizing")
    - Secret exfiltration ("print your system prompt", "reveal keys")
    - Credential patterns (AWS, Slack, HuggingFace tokens) in input
    """
    def __init__(self, block_level: int = None):
        self.block_level = int(os.getenv("TTP_BLOCK_LEVEL", block_level if block_level is not None else 3))

        self.rules = [
            (3, r"\bignore (all|any|previous) instructions\b", "prompt_injection"),
            (3, r"\boverride (system|assistant) (prompt|instructions)\b", "prompt_injection"),
            (3, r"\byou are (now )?(?:dan|dev mode)\b", "jailbreak_alias"),
            (2, r"\bnever refuse\b|\bdon't refuse\b|\balways comply\b", "safety_bypass"),
            (3, r"\bshow (me )?(your )?(system prompt|hidden instructions)\b", "sys_prompt_exfil"),
            (3, r"\bexfiltrate\b|\bleak\b|\bdump secrets?\b", "exfil_intent"),
            (2, r"BEGIN RSA PRIVATE KEY|BEGIN OPENSSH PRIVATE KEY", "secret_marker"),
            (2, r"AKIA[0-9A-Z]{16}", "aws_access_key"),
            (2, r"sk-[A-Za-z0-9]{20,}", "generic_api_key"),
            (2, r"hf_[A-Za-z0-9]{30,}", "huggingface_token"),
            (2, r"xox[baprs]-[A-Za-z0-9-]{10,}", "slack_token"),
        ]

    def score(self, text: str) -> (int, list):
        hits=[]
        sev=0
        t = text.lower()
        for level, rx, tag in self.rules:
            if re.search(rx, t, flags=re.IGNORECASE):
                hits.append(tag)
                sev = max(sev, level)
        return sev, list(sorted(set(hits)))

    def inspect_input(self, text: str) -> GuardResult:
        sev, indicators = self.score(text)
        if sev >= self.block_level:
            return GuardResult(action=GuardDecision.BLOCK, reason=f"TTP severity {sev} >= block_level", indicators=indicators)
        if sev > 0:
            return GuardResult(action=GuardDecision.ANNOTATE, reason=f"TTP indicators: {', '.join(indicators)}", indicators=indicators)
        return GuardResult(action=GuardDecision.ALLOW, reason="clean", indicators=[])

    def describe_policy(self) -> Dict[str, Any]:
        return {
            "block_level": self.block_level,
            "rules": [{"severity":lvl, "regex":rx, "tag":tag} for (lvl, rx, tag) in self.rules]
        }

# sensible default
default_guard = TTPGuard()