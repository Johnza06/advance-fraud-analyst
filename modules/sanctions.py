from __future__ import annotations
import re, pandas as pd
from typing import Optional, Dict
from validation import _prepare_generic, _standardize_df

SAN_EXPECTED = {"customer_id":["cust_id","user_id","client_id"], "name":["full_name","customer_name"]}

def prepare_sanctions(df: pd.DataFrame):
    return _prepare_generic(df, SAN_EXPECTED)

DEMO_SANCTIONS = pd.DataFrame({"name":["Ivan Petrov","Global Terror Org","Acme Front LLC","John Doe (PEP)","Shadow Brokers"]})

def token_overlap(a: str, b: str) -> int:
    at = set(re.findall(r"[A-Za-z0-9]+", a.lower()))
    bt = set(re.findall(r"[A-Za-z0-9]+", b.lower()))
    return len(at & bt)

def detect_sanctions(clean_df: pd.DataFrame, colmap: Dict[str,str], sanctions_df: Optional[pd.DataFrame]=None):
    if "name" not in colmap:
        return pd.DataFrame(), "Required column missing for Sanctions (need name)."
    df = clean_df.copy()
    sanc = sanctions_df if sanctions_df is not None else DEMO_SANCTIONS.copy()
    sanc = _standardize_df(sanc)
    if "name" not in sanc.columns:
        for c in sanc.columns:
            if "name" in c: sanc = sanc.rename(columns={c:"name"}); break
    sanc_names = sanc["name"].dropna().astype(str).tolist()
    matches=[]
    for idx, row in df.iterrows():
        nm = str(row[colmap["name"]] or "").strip()
        if not nm: continue
        if any(nm.lower()==s.lower() for s in sanc_names):
            matches.append((idx,"exact")); continue
        if any(token_overlap(nm, s) >= 2 for s in sanc_names):
            matches.append((idx,"fuzzy"))
    flagged = df.loc[[i for i,_ in matches]].copy() if matches else pd.DataFrame()
    if not flagged.empty:
        mt = {i:t for i,t in matches}
        flagged = flagged.assign(match_type=[mt.get(i,"") for i in flagged.index])
    stats = f"Sanctions matches: {len(flagged)} of {len(df)}. (Using {'uploaded/MCP' if sanctions_df is not None else 'demo'} list)"
    return flagged, stats