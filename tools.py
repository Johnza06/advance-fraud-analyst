from __future__ import annotations
import io, pandas as pd
from pydantic import BaseModel, Field
from langchain.tools import tool

from modules.transactions import prepare_transactions, detect_transactions
from modules.kyc import prepare_kyc, detect_kyc
from modules.sanctions import prepare_sanctions, detect_sanctions
from modules.credit import prepare_credit, detect_credit

def _csv_text_to_df(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_text))

class TransactionCSVInput(BaseModel):
    csv_text: str = Field(..., description="Transactions CSV text")

@tool("transactions_fraud_tool", args_schema=TransactionCSVInput)
def transactions_fraud_tool(csv_text: str) -> str:
    """Analyze transactions CSV: large/negative amounts, high-risk MCCs, per-customer daily sum >50k. Returns counts + sample."""
    df = _csv_text_to_df(csv_text)
    clean, issues, quality, colmap = prepare_transactions(df)
    flagged, stats = detect_transactions(clean, colmap)
    return f"{stats}\nData quality issues: {len(issues)}\nFirst flagged:\n{flagged.head(5).to_csv(index=False)}"[:2800]

class KYCCSVInput(BaseModel):
    csv_text: str = Field(..., description="KYC CSV text")

@tool("kyc_fraud_tool", args_schema=KYCCSVInput)
def kyc_fraud_tool(csv_text: str) -> str:
    """Analyze KYC CSV: duplicate email/phone, invalid DOBs, suspicious names. Returns counts + sample."""
    df = _csv_text_to_df(csv_text)
    clean, issues, quality, colmap = prepare_kyc(df)
    flagged, stats = detect_kyc(clean, colmap)
    return f"{stats}\nData quality issues: {len(issues)}\nFirst flagged:\n{flagged.head(5).to_csv(index=False)}"[:2800]

class SanctionsCSVInput(BaseModel):
    csv_text: str = Field(..., description="Customers CSV text with 'name' column")

@tool("sanctions_pep_tool", args_schema=SanctionsCSVInput)
def sanctions_pep_tool(csv_text: str) -> str:
    """Check customers against sanctions/PEP list (exact + simple fuzzy). Returns counts + sample."""
    df = _csv_text_to_df(csv_text)
    clean, issues, quality, colmap = prepare_sanctions(df)
    flagged, stats = detect_sanctions(clean, colmap)
    return f"{stats}\nData quality issues: {len(issues)}\nFirst matches:\n{flagged.head(5).to_csv(index=False)}"[:2800]

class CreditCSVInput(BaseModel):
    csv_text: str = Field(..., description="Credit CSV text")

@tool("credit_risk_tool", args_schema=CreditCSVInput, description="Credit risk rules: score<600, utilization>0.8, DTI>0.4, defaults>0, income<30000.")
def credit_risk_tool(csv_text: str) -> str:
    """Score credit risk using simple rules → risk_score, risk_level. Returns counts + sample."""
    df = _csv_text_to_df(csv_text)
    clean, issues, quality, colmap = prepare_credit(df)
    flagged, stats = detect_credit(clean, colmap)
    return f"{stats}\nData quality issues: {len(issues)}\nFirst flagged:\n{flagged.head(5).to_csv(index=False)}"[:2800]

def build_tools():
    return [transactions_fraud_tool, kyc_fraud_tool, sanctions_pep_tool, credit_risk_tool]