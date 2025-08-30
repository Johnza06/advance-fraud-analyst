"""
Fraud Detector Analyst — LangChain + (optional) MCP
Advanced “prototype-first” build:
- Chat uses chat-completion models (LangChain ChatHuggingFace).
- AI Summary shows a notice when no inference is connected.

LLM env (serverless friendly):
  HF_TOKEN (or HF_SPACES)
  LC_CHAT_MODEL (default: "Qwen/Qwen2.5-0.5B-Instruct")
  LC_CHAT_MODEL_FALLBACK (default: "mistralai/Mistral-7B-Instruct")

Summary behavior:
  If no working inference/token -> summary fields display:
  "🔌 Please connect to an inference point to generate summary."

Optional MCP:
  ENABLE_MCP=1
  MCP_SANCTIONS_URL, MCP_HIGH_RISK_MCC_URL
  MCP_AUTH_HEADER="Authorization: Bearer <token>"

Run:
  pip install -r requirements.txt
  python app.py
On Spaces:
  Add secret HF_TOKEN (or HF_SPACES). Launch.
"""

from __future__ import annotations

import os, io, re, json, math, unicodedata, logging
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

# LangChain
from langchain.tools import tool
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.schema import HumanMessage, SystemMessage

from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Phone normalization
try:
    import phonenumbers
    HAVE_PHONENUM = True
except Exception:
    HAVE_PHONENUM = False

# ------------------------
# Setup
# ------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fraud-analyst")

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_SPACES")

# Chat models (chat-completions)
DEFAULT_CHAT_MODEL = os.getenv("LC_CHAT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
FALLBACK_CHAT_MODEL = os.getenv("LC_CHAT_MODEL_FALLBACK", "mistralai/Mistral-7B-Instruct")

SUMMARY_NOTICE = "🔌 Please connect to an inference point to generate summary."
CHAT_NOTICE = "🔌 Chat model not configured. Set HF_TOKEN and LC_CHAT_MODEL to enable chat."

# ------------------------
# LLM builders
# ------------------------
def _mk_chat_llm(model_id: str) -> ChatHuggingFace:
    """
    ChatHuggingFace uses HF Inference under the hood.
    Although the backend task is 'text-generation', this wrapper handles chat-style messages.
    """
    base = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=256,
        temperature=0.2,
        repetition_penalty=1.05,
        timeout=60,
    )
    return ChatHuggingFace(llm=base)

def _heartbeat_chat(model_id: str) -> bool:
    try:
        chat = _mk_chat_llm(model_id)
        _ = chat.invoke([HumanMessage(content="ok")])
        return True
    except Exception as e:
        log.warning(f"Heartbeat failed for {model_id}: {str(e)[:160]}")
        return False

def build_chat_llm() -> Optional[ChatHuggingFace]:
    """
    Returns a working ChatHuggingFace or None (if token/permissions missing).
    """
    log.info(f"HF token present: {bool(HF_TOKEN)} len={len(HF_TOKEN) if HF_TOKEN else 0}")
    if HF_TOKEN and _heartbeat_chat(DEFAULT_CHAT_MODEL):
        log.info(f"Using chat model: {DEFAULT_CHAT_MODEL}")
        return _mk_chat_llm(DEFAULT_CHAT_MODEL)
    if HF_TOKEN and _heartbeat_chat(FALLBACK_CHAT_MODEL):
        log.info(f"Using fallback chat model: {FALLBACK_CHAT_MODEL}")
        return _mk_chat_llm(FALLBACK_CHAT_MODEL)
    log.warning("No working chat model; chat will show a notice.")
    return None

CHAT_LLM = build_chat_llm()

# ------------------------
# Normalization helpers
# ------------------------
def _norm_colname(c: str) -> str:
    c = c.strip().lower()
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"[^\w]+", "_", c)
    return c.strip("_")

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _clean_str(x):
    if pd.isna(x): return x
    return _collapse_ws(_nfkc(str(x)))

def _is_email(s: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", s or ""))

def _clean_phone(s: str, default_region: str = "IN"):
    if s is None or str(s).strip() == "":
        return None, "missing_phone"
    raw = re.sub(r"[^\d+]", "", str(s))
    if HAVE_PHONENUM:
        try:
            pn = phonenumbers.parse(raw, default_region)
            if phonenumbers.is_possible_number(pn) and phonenumbers.is_valid_number(pn):
                return phonenumbers.format_number(pn, phonenumbers.PhoneNumberFormat.E164), None
            return raw, "invalid_phone"
        except Exception:
            return raw, "invalid_phone"
    digits = re.sub(r"\D", "", raw)
    return (digits, None) if 8 <= len(digits) <= 15 else (digits, "invalid_phone")

def _parse_datetime(s):
    try:
        return pd.to_datetime(s, errors="coerce", utc=True)
    except Exception:
        return pd.NaT

def _to_numeric(series: pd.Series):
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced, (coerced.isna() & series.notna())

def _read_csv_any(file_obj) -> pd.DataFrame:
    if file_obj is None:
        raise ValueError("No file uploaded.")
    if hasattr(file_obj, "name"):
        p = file_obj.name
        try: return pd.read_csv(p)
        except Exception: return pd.read_csv(p, encoding="latin-1")
    try: return pd.read_csv(file_obj)
    except Exception:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding="latin-1")

def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm_colname(c) for c in df.columns]
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].apply(_clean_str)
    return df

def _prepare_generic(df: pd.DataFrame, expected: Dict[str, List[str]]):
    issues = []
    df0 = _standardize_df(df)

    # Synonym mapping
    colmap = {}
    cols = set(df0.columns)
    for canon, syns in expected.items():
        found = None
        for s in [canon] + syns:
            s = _norm_colname(s)
            if s in cols:
                found = s; break
        if found: colmap[canon] = found

    # Email/phone quality
    for c in list(df0.columns):
        if "email" in c:
            df0[c] = df0[c].apply(lambda x: str(x).lower().strip() if pd.notna(x) else x)
            for idx, v in df0[c].items():
                if pd.isna(v) or str(v).strip()=="":
                    issues.append({"row": idx, "field": c, "issue":"missing_email","value":""})
                elif not _is_email(v):
                    issues.append({"row": idx, "field": c, "issue":"invalid_email","value":str(v)})
        if "phone" in c or "mobile" in c:
            vals = []
            for idx, v in df0[c].items():
                e164, prob = _clean_phone(v)
                vals.append(e164)
                if prob: issues.append({"row": idx, "field": c, "issue":prob, "value":str(v)})
            df0[c] = vals

    # Datetime parsing
    for c in df0.columns:
        if any(k in c for k in ["date","time","timestamp","created_at","updated_at"]):
            parsed = _parse_datetime(df0[c])
            bad = parsed.isna() & df0[c].notna()
            for idx in df0.index[bad]:
                issues.append({"row": int(idx), "field": c, "issue":"unparseable_timestamp", "value":str(df0.loc[idx, c])})
            df0[c] = parsed

    # Numeric coercions for common fields
    for nc in ["amount","credit_score","utilization","dti","recent_defaults","income"]:
        for c in df0.columns:
            if c == nc or c.endswith("_"+nc) or nc in c:
                coerced, badmask = _to_numeric(df0[c])
                for idx in df0.index[badmask]:
                    issues.append({"row": int(idx), "field": c, "issue":"non_numeric", "value":str(df0.loc[idx, c])})
                df0[c] = coerced

    issues_df = pd.DataFrame(issues, columns=["row","field","issue","value"])
    missing = [k for k in expected.keys() if k not in colmap]
    quality_summary = f"Rows={len(df0)}, Cols={len(df0.columns)}; Missing required fields: {missing if missing else 'None'}"
    return df0, issues_df, quality_summary, colmap

# ------------------------
# Modules & Rules
# ------------------------
TX_EXPECTED = {
    "transaction_id":["txn_id","transactionid","id","tx_id"],
    "customer_id":["cust_id","user_id","client_id"],
    "amount":["amt","amount_inr","value"],
    "timestamp":["date","event_time","created_at","tx_time"],
    "merchant_category":["mcc","merchant_cat","category"]
}
def prepare_transactions(df): return _prepare_generic(df, TX_EXPECTED)

def detect_transactions(clean_df, colmap, high_risk_mcc: Optional[List[str]]=None):
    high_risk = set(["HIGH_RISK","GAMBLING","CRYPTO_EXCHANGE","ESCORTS","CASINO"])
    if high_risk_mcc:
        high_risk.update([_nfkc(x).strip().upper().replace(" ","_") for x in high_risk_mcc])
    if not all(k in colmap for k in ["customer_id","amount"]):
        return pd.DataFrame(), "Required columns missing for detection (need at least customer_id, amount)."
    df = clean_df.copy()
    reasons = []
    amtcol = colmap.get("amount")
    if amtcol is not None:
        reasons.append(("large_amount>10k", df[amtcol] > 10000))
        reasons.append(("negative_amount", df[amtcol] < 0))
    if "merchant_category" in colmap:
        mcc = colmap["merchant_category"]
        high = df[mcc].astype(str).str.upper().str.replace(" ","_", regex=False).isin(high_risk)
        reasons.append(("merchant_category_high_risk", high))
    if all(k in colmap for k in ["customer_id","timestamp","amount"]):
        cid, ts, amt = colmap["customer_id"], colmap["timestamp"], colmap["amount"]
        daily = df.groupby([cid, df[ts].dt.date])[amt].transform("sum")
        reasons.append(("daily_sum_per_customer>50k", daily > 50000))
    mask = None
    for _, m in reasons:
        mask = m if mask is None else (mask | m)
    flagged = df[mask] if mask is not None else pd.DataFrame()
    if not flagged.empty:
        rr=[]
        for _, row in flagged.iterrows():
            hits=[]
            a = row[amtcol] if amtcol in flagged.columns else None
            if pd.notna(a) and a>10000: hits.append("large_amount")
            if pd.notna(a) and a<0: hits.append("negative_amount")
            if "merchant_category" in colmap:
                val = str(row[colmap["merchant_category"]]).upper().replace(" ","_")
                if val in high_risk: hits.append("mcc_high_risk")
            # daily sum check reconstructed
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

KYC_EXPECTED = {
    "customer_id":["cust_id","user_id","client_id"],
    "name":["full_name","customer_name"],
    "email":["email_address","mail"],
    "phone":["phone_number","mobile","contact"],
    "dob":["date_of_birth","birthdate"]
}
def prepare_kyc(df): return _prepare_generic(df, KYC_EXPECTED)

def _age_years(dob: pd.Series) -> pd.Series:
    now = pd.Timestamp.utcnow()
    return (now - dob).dt.days / 365.25

def detect_kyc(clean_df, colmap):
    if not all(k in colmap for k in ["customer_id","name"]):
        return pd.DataFrame(), "Required columns missing for KYC (need at least customer_id, name)."
    df = clean_df.copy()
    reasons=[]
    if "email" in colmap:
        dupe_email = df.duplicated(subset=[colmap["email"]], keep=False) & df[colmap["email"]].notna()
        reasons.append(("duplicate_email", dupe_email))
    if "phone" in colmap:
        dupe_phone = df.duplicated(subset=[colmap["phone"]], keep=False) & df[colmap["phone"]].notna()
        reasons.append(("duplicate_phone", dupe_phone))
    if "dob" in colmap:
        age = _age_years(df[colmap["dob"]])
        invalid = (df[colmap["dob"]].isna()) | (df[colmap["dob"]] > pd.Timestamp.utcnow()) | (age > 120)
        reasons.append(("invalid_dob", invalid))
    if "name" in colmap:
        name = df[colmap["name"]].astype(str)
        susp = name.str.isupper() | name.str.contains(r"\d") | (name.str.len()<3)
        reasons.append(("suspicious_name", susp))
    mask = None
    for _, m in reasons:
        mask = m if mask is None else (mask | m)
    flagged = df[mask] if mask is not None else pd.DataFrame()
    if not flagged.empty:
        flagged = flagged.assign(risk_reason="kyc_rule_hit")
    stats = f"KYC flagged: {len(flagged)} of {len(df)}."
    return flagged, stats

SAN_EXPECTED = {"customer_id":["cust_id","user_id","client_id"], "name":["full_name","customer_name"]}
def prepare_sanctions(df): return _prepare_generic(df, SAN_EXPECTED)

DEMO_SANCTIONS = pd.DataFrame({"name":["Ivan Petrov","Global Terror Org","Acme Front LLC","John Doe (PEP)","Shadow Brokers"]})

def token_overlap(a: str, b: str) -> int:
    at = set(re.findall(r"[A-Za-z0-9]+", a.lower()))
    bt = set(re.findall(r"[A-Za-z0-9]+", b.lower()))
    return len(at & bt)

def detect_sanctions(clean_df, colmap, sanctions_df: Optional[pd.DataFrame]=None):
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

CR_EXPECTED = {
    "customer_id":["cust_id","user_id","client_id"],
    "credit_score":["creditscore","score"],
    "utilization":["util","credit_utilization","utilization_ratio"],
    "dti":["debt_to_income","debt_to_income_ratio"],
    "recent_defaults":["defaults","recentdefaults"],
    "income":["annual_income","salary"]
}
def prepare_credit(df): return _prepare_generic(df, CR_EXPECTED)

def detect_credit(clean_df, colmap):
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
    risk_score=[x[0] for x in out]; risk_level=[x[1] for x in out]; reason=[x[2] for x in out]
    res = df.assign(risk_score=risk_score, risk_level=risk_level, risk_reason=reason)
    flagged = res[res["risk_level"].isin(["High","Medium","Low"]) & (res["risk_level"]!="None")]
    stats = f"Credit Risk flagged: {len(flagged)} of {len(df)}. Distribution: High={(res['risk_level']=='High').sum()}, Medium={(res['risk_level']=='Medium').sum()}, Low={(res['risk_level']=='Low').sum()}."
    return flagged, stats

# ------------------------
# Summarizer (notice-first)
# ------------------------
SUMMARY_SYS = "You are a helpful Fraud/Risk analyst. Be concise (<120 words), list key counts, drivers, and data quality caveats."

def summarize_ai(context: str) -> str:
    """
    If chat LLM is available, use it to generate a short summary.
    Otherwise return the prototype notice string.
    """
    if CHAT_LLM is None:
        return SUMMARY_NOTICE
    try:
        out = CHAT_LLM.invoke([SystemMessage(content=SUMMARY_SYS), HumanMessage(content=context[:4000])])
        if hasattr(out, "content"): return out.content
        return str(out)
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            return SUMMARY_NOTICE
        return SUMMARY_NOTICE

# ------------------------
# Optional MCP
# ------------------------
from urllib.request import Request, urlopen
def _mcp_get_json(url: str, auth_header: Optional[str]):
    try:
        req = Request(url)
        if auth_header:
            k, v = auth_header.split(":", 1)
            req.add_header(k.strip(), v.strip())
        with urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        log.warning(f"MCP fetch failed: {e}")
        return None

def mcp_fetch_sanctions() -> Optional[pd.DataFrame]:
    if os.getenv("ENABLE_MCP","0") not in ("1","true","TRUE"): return None
    url = os.getenv("MCP_SANCTIONS_URL")
    if not url: return None
    data = _mcp_get_json(url, os.getenv("MCP_AUTH_HEADER"))
    if not data: return None
    if isinstance(data, list):
        if all(isinstance(x, dict) for x in data):
            rows = [{"name": x.get("name") or x.get("Name")} for x in data if x.get("name") or x.get("Name")]
            return pd.DataFrame(rows) if rows else None
        if all(isinstance(x, str) for x in data):
            return pd.DataFrame({"name": data})
    return None

def mcp_fetch_high_risk_mcc() -> Optional[List[str]]:
    if os.getenv("ENABLE_MCP","0") not in ("1","true","TRUE"): return None
    url = os.getenv("MCP_HIGH_RISK_MCC_URL")
    if not url: return None
    data = _mcp_get_json(url, os.getenv("MCP_AUTH_HEADER"))
    return [str(x) for x in data] if isinstance(data, list) else None

# ------------------------
# Pipelines (per tab)
# ------------------------
def run_transactions(file):
    try:
        df = _read_csv_any(file)
        clean, issues, quality, colmap = prepare_transactions(df)
        mcc = mcp_fetch_high_risk_mcc()
        flagged, stats = detect_transactions(clean, colmap, mcc)
        ctx = f"[Transactions]\n{stats}\nQuality: {quality}\nHead:\n{clean.head(5).to_csv(index=False)}\nFlagged:\n{flagged.head(5).to_csv(index=False)}"
        ai = summarize_ai(ctx)
        return ai, stats, flagged, issues
    except Exception as e:
        return f"Error: {e}", "Validation failed.", pd.DataFrame(), pd.DataFrame()

def run_kyc(file):
    try:
        df = _read_csv_any(file)
        clean, issues, quality, colmap = prepare_kyc(df)
        flagged, stats = detect_kyc(clean, colmap)
        ctx = f"[KYC]\n{stats}\nQuality: {quality}\nHead:\n{clean.head(5).to_csv(index=False)}\nFlagged:\n{flagged.head(5).to_csv(index=False)}"
        ai = summarize_ai(ctx)
        return ai, stats, flagged, issues
    except Exception as e:
        return f"Error: {e}", "Validation failed.", pd.DataFrame(), pd.DataFrame()

def run_sanctions(customers_file, sanctions_file):
    try:
        df = _read_csv_any(customers_file)
        clean, issues, quality, colmap = prepare_sanctions(df)
        sanc_df = mcp_fetch_sanctions()
        if sanc_df is None and sanctions_file is not None:
            sanc_df = _read_csv_any(sanctions_file)
        flagged, stats = detect_sanctions(clean, colmap, sanc_df)
        ctx = f"[Sanctions]\n{stats}\nQuality: {quality}\nHead:\n{clean.head(5).to_csv(index=False)}\nMatches:\n{flagged.head(5).to_csv(index=False)}"
        ai = summarize_ai(ctx)
        return ai, stats, flagged, issues
    except Exception as e:
        return f"Error: {e}", "Validation failed.", pd.DataFrame(), pd.DataFrame()

def run_credit(file):
    try:
        df = _read_csv_any(file)
        clean, issues, quality, colmap = prepare_credit(df)
        flagged, stats = detect_credit(clean, colmap)
        ctx = f"[Credit]\n{stats}\nQuality: {quality}\nHead:\n{clean.head(5).to_csv(index=False)}\nFlagged:\n{flagged.head(5).to_csv(index=False)}"
        ai = summarize_ai(ctx)
        return ai, stats, flagged, issues
    except Exception as e:
        return f"Error: {e}", "Validation failed.", pd.DataFrame(), pd.DataFrame()

# ------------------------
# Tools (CSV text in → concise text out)
# ------------------------
def _csv_text_to_df(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_text))

class TransactionCSVInput(BaseModel):
    csv_text: str = Field(..., description="Transactions CSV text")

@tool("transactions_fraud_tool", args_schema=TransactionCSVInput)
def transactions_fraud_tool(csv_text: str) -> str:
    df = _csv_text_to_df(csv_text)
    clean, issues, quality, colmap = prepare_transactions(df)
    flagged, stats = detect_transactions(clean, colmap)
    return f"{stats}\nDQ issues: {len(issues)}\nFirst flagged:\n{flagged.head(5).to_csv(index=False)}"[:2800]

class KYCCSVInput(BaseModel):
    csv_text: str = Field(..., description="KYC CSV text")

@tool("kyc_fraud_tool", args_schema=KYCCSVInput)
def kyc_fraud_tool(csv_text: str) -> str:
    df = _csv_text_to_df(csv_text)
    clean, issues, quality, colmap = prepare_kyc(df)
    flagged, stats = detect_kyc(clean, colmap)
    return f"{stats}\nDQ issues: {len(issues)}\nFirst flagged:\n{flagged.head(5).to_csv(index=False)}"[:2800]

class SanctionsCSVInput(BaseModel):
    csv_text: str = Field(..., description="Customers CSV text with a 'name' column")

@tool("sanctions_pep_tool", args_schema=SanctionsCSVInput)
def sanctions_pep_tool(csv_text: str) -> str:
    df = _csv_text_to_df(csv_text)
    clean, issues, quality, colmap = prepare_sanctions(df)
    flagged, stats = detect_sanctions(clean, colmap)
    return f"{stats}\nDQ issues: {len(issues)}\nFirst matches:\n{flagged.head(5).to_csv(index=False)}"[:2800]

class CreditCSVInput(BaseModel):
    csv_text: str = Field(..., description="Credit CSV text")

@tool("credit_risk_tool", args_schema=CreditCSVInput)
def credit_risk_tool(csv_text: str) -> str:
    df = _csv_text_to_df(csv_text)
    clean, issues, quality, colmap = prepare_credit(df)
    flagged, stats = detect_credit(clean, colmap)
    return f"{stats}\nDQ issues: {len(issues)}\nFirst flagged:\n{flagged.head(5).to_csv(index=False)}"[:2800]

TOOLS: List[Tool] = [
    transactions_fraud_tool,
    kyc_fraud_tool,
    sanctions_pep_tool,
    credit_risk_tool,
]

# ------------------------
# Agent (chat-completions)
# ------------------------
AGENT_SYSTEM = """You are an AI Consultant for Fraud/Risk.
You have tools for Transactions, KYC, Sanctions/PEP, and Credit Risk.
If the user pastes a small CSV snippet, pick the relevant tool and analyze it.
Be concise and actionable."""

def build_agent():
    if CHAT_LLM is None:
        class Stub:
            def invoke(self, prompt): return CHAT_NOTICE
        return Stub()
    return initialize_agent(
        TOOLS,
        CHAT_LLM,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        agent_kwargs={"system_message": AGENT_SYSTEM},
        handle_parsing_errors=True,
    )

AGENT = build_agent()

def agent_reply(history: List[Dict], user_msg: str):
    try:
        looks_like_csv = ("," in user_msg) and ("\n" in user_msg) and (user_msg.count(",") >= 2)
        prompt = f"CSV snippet detected. Decide tool and analyze:\n\n{user_msg}" if looks_like_csv else user_msg
        res = AGENT.invoke(prompt)
        if isinstance(res, dict) and "output" in res: return res["output"]
        return str(res)
    except Exception as e:
        return f"Agent error: {e}"

# ------------------------
# UI
# ------------------------
with gr.Blocks(title="Fraud Detector Analyst — LangChain + MCP", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Fraud Detector Analyst — LangChain + MCP")
    gr.Markdown(
        "This prototype runs **rules & data checks locally**. "
        "Chat + AI summaries require a remote inference provider (HF Inference)."
    )

    with gr.Tabs():
        with gr.Tab("Transactions"):
            gr.Markdown("Upload a **transactions** CSV.")
            tx_file = gr.File(file_types=[".csv"], label="Transactions CSV", type="binary")
            tx_ai = gr.Textbox(label="AI Summary (requires inference)", value=SUMMARY_NOTICE, lines=6)
            tx_stats = gr.Textbox(label="Stats", lines=3)
            tx_flagged = gr.Dataframe(label="Flagged Transactions")
            tx_issues = gr.Dataframe(label="Data Quality Issues (row, field, issue, value)")
            tx_file.upload(run_transactions, inputs=[tx_file], outputs=[tx_ai, tx_stats, tx_flagged, tx_issues])

        with gr.Tab("KYC"):
            gr.Markdown("Upload a **KYC** CSV.")
            kyc_file = gr.File(file_types=[".csv"], label="KYC CSV", type="binary")
            kyc_ai = gr.Textbox(label="AI Summary (requires inference)", value=SUMMARY_NOTICE, lines=6)
            kyc_stats = gr.Textbox(label="Stats", lines=3)
            kyc_flagged = gr.Dataframe(label="Flagged KYC Rows")
            kyc_issues = gr.Dataframe(label="Data Quality Issues")
            kyc_file.upload(run_kyc, inputs=[kyc_file], outputs=[kyc_ai, kyc_stats, kyc_flagged, kyc_issues])

        with gr.Tab("Sanctions/PEP"):
            gr.Markdown("Upload **customers** CSV (+ optional sanctions CSV).")
            san_customers = gr.File(file_types=[".csv"], label="Customers CSV", type="binary")
            san_list = gr.File(file_types=[".csv"], label="Sanctions/PEP CSV (optional)", type="binary")
            san_ai = gr.Textbox(label="AI Summary (requires inference)", value=SUMMARY_NOTICE, lines=6)
            san_stats = gr.Textbox(label="Stats", lines=3)
            san_flagged = gr.Dataframe(label="Matches")
            san_issues = gr.Dataframe(label="Data Quality Issues")
            san_customers.upload(run_sanctions, inputs=[san_customers, san_list], outputs=[san_ai, san_stats, san_flagged, san_issues])
            san_list.upload(run_sanctions, inputs=[san_customers, san_list], outputs=[san_ai, san_stats, san_flagged, san_issues])

        with gr.Tab("Credit Risk"):
            gr.Markdown("Upload a **credit** CSV.")
            cr_file = gr.File(file_types=[".csv"], label="Credit CSV", type="binary")
            cr_ai = gr.Textbox(label="AI Summary (requires inference)", value=SUMMARY_NOTICE, lines=6)
            cr_stats = gr.Textbox(label="Stats", lines=3)
            cr_flagged = gr.Dataframe(label="Flagged Applicants")
            cr_issues = gr.Dataframe(label="Data Quality Issues")
            cr_file.upload(run_credit, inputs=[cr_file], outputs=[cr_ai, cr_stats, cr_flagged, cr_issues])

        with gr.Tab("AI Consultant (Agent)"):
            gr.Markdown("Paste a small CSV snippet or ask questions. Uses chat-completions when configured.")
            chatbot = gr.Chatbot(type="messages", label="Fraud AI Consultant")
            user_in = gr.Textbox(label="Message or CSV snippet")
            send_btn = gr.Button("Send")
            def _chat_fn(history, msg):
                reply = agent_reply(history, msg)
                history = (history or []) + [{"role":"user","content":msg}, {"role":"assistant","content":reply}]
                return history, ""
            send_btn.click(_chat_fn, inputs=[chatbot, user_in], outputs=[chatbot, user_in])

    gr.Markdown(
        "### ⚙️ Enable inference\n"
        "- Set **HF_TOKEN** (or HF_SPACES on Spaces)\n"
        "- Optional: **LC_CHAT_MODEL** (default Qwen 0.5B Instruct), **LC_CHAT_MODEL_FALLBACK** (default Mistral 7B Instruct)\n"
        "- Optional MCP: `ENABLE_MCP=1`, `MCP_SANCTIONS_URL`, `MCP_HIGH_RISK_MCC_URL`, `MCP_AUTH_HEADER`"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)