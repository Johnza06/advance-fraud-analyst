from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
from mcp import mcp_fetch_sanctions, mcp_fetch_high_risk_mcc
from modules.sanctions import DEMO_SANCTIONS

@dataclass
class ThreatIntel:
    sanctions_df: Optional[pd.DataFrame]
    high_risk_mcc: List[str]

    @staticmethod
    def load() -> "ThreatIntel":
        sanc = mcp_fetch_sanctions()
        mcc = mcp_fetch_high_risk_mcc() or ["HIGH_RISK","GAMBLING","CRYPTO_EXCHANGE","ESCORTS","CASINO"]
        return ThreatIntel(sanctions_df=sanc or DEMO_SANCTIONS, high_risk_mcc=mcc)