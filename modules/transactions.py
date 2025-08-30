from __future__ import annotations
import pandas as pd
from typing import Optional, List, Dict
from validation import _prepare_generic, _nfkc

TX_EXPECTED = {
    "transaction_id":["txn_id","transactionid","id","tx_id"],
    "customer_id":["cust_id","user_id","client_id"],
    "amount":["amt","amount_inr","value"],
    "timestamp":["date","event_time","created_at","tx_time"],
    "merchant_category":["mcc","merchant_cat","category"]
}

def prepare_transactions(df: pd.DataFrame):
    return _prepare_generic(df, TX_EXPECTED)

def detect_transactions(clean_df: pd.DataFrame, colmap: Dict[str,str], high_risk_mcc: Optional[List[str]] = None):
    high_risk = set(["HIGH_RISK","GAMBLING","CRYPTO_EXCHANGE","ESCORTS","CASINO"])
    if high_risk_mcc:
        high_risk.update([_nfkc(x).strip().upper().replace(" ","_") for x in high_risk_mcc])
    if not all(k in colmap for k in ["customer_id","amount"]):
        return pd.DataFrame(), "Required columns missing for detection (need at least customer_id, amount)."
    df = clean_df.copy()
    reasons = []
    amtcol = colmap.get("amount")
    if amtcol:
        reasons.append(df[amtcol] > 10000)    # large
        reasons.append(df[amtcol] < 0)        # negative
    if "merchant_category" in colmap:
        mcc = colmap["merchant_category"]
        high = df[mcc].astype(str).str.upper().str.replace(" ","_", regex=False).isin(high_risk)
        reasons.append(high)
    if all(k in colmap for k in ["customer_id","timestamp","amount"]):
        cid, ts, amt = colmap["customer_id"], colmap["timestamp"], colmap["amount"]
        daily = df.groupby([cid, df[ts].dt.date])[amt].transform("sum")
        reasons.append(daily > 50000)
    mask = None
    for m in reasons:
        mask = m if mask is None else (mask | m)
    flagged = df[mask] if mask is not None else pd.DataFrame()
    if not flagged.empty:
        rr=[]
        for _, row in flagged.iterrows():
            hits=[]
            if amtcol:
                a=row[amtcol]
                if pd.notna(a) and a>10000: hits.append("large_amount")
                if pd.notna(a) and a<0: hits.append("negative_amount")
            if "merchant_category" in colmap:
                val = str(row[colmap["merchant_category"]]).upper().replace(" ","_")
                if val in high_risk: hits.append("mcc_high_risk")
            try:
                if all(k in colmap for k in ["customer_id","timestamp","amount"]):
                    sub = df[(df[colmap["customer_id"]]==row[colmap["customer_id"]]) &
                             (df[colmap["timestamp"]].dt.date==pd.to_datetime(row[colmap["timestamp"]], errors="coerce").date())]
                    if sub[colmap["amount"]].sum() > 50000: hits.append("daily_sum>50k")
            except Exception: pass
            rr.append(", ".join(sorted(set(hits))) or "rule_hit")
        flagged = flagged.assign(risk_reason=rr)
    stats = f"Transactions flagged: {len(flagged)} of {len(df)}."
    return flagged, stats