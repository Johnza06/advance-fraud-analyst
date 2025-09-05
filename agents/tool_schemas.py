from pydantic import BaseModel

class TransactionLookupInput(BaseModel):
    transaction_id: str

class TransactionLookupOutput(BaseModel):
    status: str
    risk_score: float
