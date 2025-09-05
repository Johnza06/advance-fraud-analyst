# Agents

This module demonstrates tool schemas and an agent with retry, backoff, and circuit-breaker logic. Routing decisions are logged via Python's `logging` module.

* `tool_schemas.py` defines typed input/output models using Pydantic.
* `example_agent.py` shows a simple agent that retries failed tool calls and opens a circuit after repeated failures.
