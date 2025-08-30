from __future__ import annotations
import pandas as pd, numpy as np
from typing import Dict
from validation import _prepare_generic

CR_EXPECTED = {
    "customer_id":["cust_id","user_id","client_id"],
    "credit_score":["creditscore","score"],
    "utilization":["util","credit_utilization","utilization_ratio"],
    "dti":["debt_to_income","debt_to_income_ratio"],
    "recent_defaults":["defaults","recentdefaults"],
    "income":["annual_income","salary"]
}

def prepare_credit(df: pd.DataFrame):
    return _prepare_generic(df, CR_EXPECTED)

def detect_credit(clean_df: pd.DataFrame, colmap: Dict[str,str]):
    needed = ["credit_score","utilization","dti","recent_defaults","income"]
    if not any(k in colmap for k in needed):
        return pd.DataFrame(), "Required columns missing for Credit Risk."
    df = clean_df.copy()
    cs  = df[colmap.get("credit_score","credit_score")] if "credit_score" in colmap else pd.Series([np.nan]*len(df))
    util= df[colmap.get("utilization","utilization")] if "utilization" in colmap else pd.Series([np.nan]*len(df))
    dti = df[colmap.get("dti","dti")] if "dti" in colmap else pd.Series([np.nan]*len(df))
    rde = df[colmap.get("recent_defaults","recent_defaults")] if "recent_defaults" in colmap else pd.Series([np.nan]*len(df))
    inc = df[colmap.get("income","income")] if "income" in colmap else pd.Series([np.nan]*len(df))
    out=[]
    for i in range(len(df)):
        hits=0; reasons=[]
        if pd.notna(cs.iloc[i]) and cs.iloc[i] < 600: hits+=1; reasons.append("credit_score<600")
        if pd.notna(util.iloc[i]) and util.iloc[i] > 0.8: hits+=1; reasons.append("utilization>0.8")
        if pd.notna(dti.iloc[i]) and dti.iloc[i] > 0.4: hits+=1; reasons.append("DTI>0.4")
        if pd.notna(rde.iloc[i]) and rde.iloc[i] > 0: hits+=1; reasons.append("recent_defaults>0")
        if pd.notna(inc.iloc[i]) and inc.iloc[i] < 30000: hits+=1; reasons.append("income<30000")
        level = "High" if hits>=3 else ("Medium" if hits==2 else ("Low" if hits==1 else "None"))
        out.append((hits, level, ", ".join(reasons)))
    res = df.assign(
        risk_score=[x[0] for x in out],
        risk_level=[x[1] for x in out],
        risk_reason=[x[2] for x in out]
    )
    flagged = res[res["risk_level"].isin(["High","Medium","Low"]) & (res["risk_level"]!="None")]
    stats = f"Credit Risk flagged: {len(flagged)} of {len(df)}. Distribution: High={(res['risk_level']=='High').sum()}, Medium={(res['risk_level']=='Medium').sum()}, Low={(res['risk_level']=='Low').sum()}."
    return flagged, stats