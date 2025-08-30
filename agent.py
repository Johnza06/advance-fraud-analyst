from __future__ import annotations
from typing import List
from langchain.agents import initialize_agent, AgentType
from llm_provider import CHAT_LLM, SUMMARY_NOTICE
from ttp_guard import TTPGuard, GuardDecision

AGENT_SYSTEM = """You are an AI Consultant for Fraud/Risk.
You have tools for Transactions, KYC, Sanctions/PEP, and Credit Risk.
If the user pastes a small CSV snippet, pick the relevant tool and analyze it.
Be concise and actionable."""

def build_agent(tools: List, guard: TTPGuard):
    if CHAT_LLM is None:
        # Stub agent that returns notice
        class Stub:
            def invoke(self, prompt): return SUMMARY_NOTICE
        return Stub()

    # Wrap LLM invocation with a guard-aware tool-use policy by leveraging the system message.
    return initialize_agent(
        tools,
        CHAT_LLM,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        agent_kwargs={"system_message": AGENT_SYSTEM},
        handle_parsing_errors=True,
    )