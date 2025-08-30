from __future__ import annotations
import pandas as pd
from typing import Dict
from validation import _prepare_generic
import numpy as np

KYC_EXPECTED = {
    "customer_id":["cust_id","user_id","client_id"],
    "name":["full_name","customer_name"],
    "email":["email_address","mail"],
    "phone":["phone_number","mobile","contact"],
    "dob":["date_of_birth","birthdate"]
}

def prepare_kyc(df: pd.DataFrame):
    return _prepare_generic(df, KYC_EXPECTED)

def _age_years(dob: pd.Series) -> pd.Series:
    now = pd.Timestamp.utcnow()
    return (now - dob).dt.days / 365.25

def detect_kyc(clean_df: pd.DataFrame, colmap: Dict[str,str]):
    if not all(k in colmap for k in ["customer_id","name"]):
        return pd.DataFrame(), "Required columns missing for KYC (need at least customer_id, name)."
    df = clean_df.copy()
    reasons=[]
    if "email" in colmap:
        dupe_email = df.duplicated(subset=[colmap["email"]], keep=False) & df[colmap["email"]].notna()
        reasons.append(dupe_email)
    if "phone" in colmap:
        dupe_phone = df.duplicated(subset=[colmap["phone"]], keep=False) & df[colmap["phone"]].notna()
        reasons.append(dupe_phone)
    if "dob" in colmap:
        age = _age_years(df[colmap["dob"]])
        invalid = (df[colmap["dob"]].isna()) | (df[colmap["dob"]] > pd.Timestamp.utcnow()) | (age > 120)
        reasons.append(invalid)
    if "name" in colmap:
        name = df[colmap["name"]].astype(str)
        susp = name.str.isupper() | name.str.contains(r"\d") | (name.str.len()<3)
        reasons.append(susp)
    mask=None
    for m in reasons:
        mask = m if mask is None else (mask | m)
    flagged = df[mask] if mask is not None else pd.DataFrame()
    if not flagged.empty:
        flagged = flagged.assign(risk_reason="kyc_rule_hit")
    stats = f"KYC flagged: {len(flagged)} of {len(df)}."
    return flagged, stats