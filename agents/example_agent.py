import logging
import random
import time
from typing import Callable

from .tool_schemas import TransactionLookupInput, TransactionLookupOutput

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CircuitBreaker(Exception):
    pass

class ToolRouter:
    """Routes calls to tools and logs the decision."""
    
    def __init__(self, tool: Callable[[TransactionLookupInput], TransactionLookupOutput]):
        self.tool = tool
        self.failures = 0
        self.max_failures = 3
    
    def call(self, inp: TransactionLookupInput) -> TransactionLookupOutput:
        logger.info("Routing to tool with transaction_id=%s", inp.transaction_id)
        if self.failures >= self.max_failures:
            logger.error("Circuit open: too many failures")
            raise CircuitBreaker("circuit open")
        
        for attempt in range(3):
            try:
                return self.tool(inp)
            except Exception as e:
                self.failures += 1
                wait = 2 ** attempt
                logger.warning("Tool failed (%s). retrying in %ss", e, wait)
                time.sleep(wait)
        logger.error("Tool failed after retries")
        raise CircuitBreaker("tool unavailable")

# Example tool implementation
def mock_transaction_lookup(inp: TransactionLookupInput) -> TransactionLookupOutput:
    if random.random() < 0.2:
        raise RuntimeError("random failure")
    return TransactionLookupOutput(status="ok", risk_score=random.random())

if __name__ == "__main__":
    router = ToolRouter(mock_transaction_lookup)
    req = TransactionLookupInput(transaction_id="123")
    try:
        resp = router.call(req)
        logger.info("Tool response: %s", resp)
    except CircuitBreaker:
        logger.error("Call aborted due to circuit breaker")
