from __future__ import annotations
import re, math, unicodedata
import pandas as pd
import numpy as np

try:
    import phonenumbers
    HAVE_PHONENUM = True
except Exception:
    HAVE_PHONENUM = False

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

def _prepare_generic(df: pd.DataFrame, expected: dict[str, list[str]]):
    issues = []
    df0 = _standardize_df(df)

    colmap = {}
    cols = set(df0.columns)
    for canon, syns in expected.items():
        found = None
        for s in [canon] + syns:
            s = _norm_colname(s)
            if s in cols:
                found = s; break
        if found: colmap[canon] = found

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

    for c in df0.columns:
        if any(k in c for k in ["date","time","timestamp","created_at","updated_at"]):
            parsed = _parse_datetime(df0[c])
            bad = parsed.isna() & df0[c].notna()
            for idx in df0.index[bad]:
                issues.append({"row": int(idx), "field": c, "issue":"unparseable_timestamp", "value":str(df0.loc[idx, c])})
            df0[c] = parsed

    for nc in ["amount","credit_score","utilization","dti","recent_defaults","income"]:
        for c in df0.columns:
            if c == nc or c.endswith("_"+nc) or nc in c:
                coerced, badmask = _to_numeric(df0[c])
                for idx in df0.index[badmask]:
                    issues.append({"row": int(idx), "field": c, "issue":"non_numeric", "value":str(df0.loc[idx, c])})
                df0[c] = coerced

    import pandas as pd
    issues_df = pd.DataFrame(issues, columns=["row","field","issue","value"])
    missing = [k for k in expected.keys() if k not in colmap]
    quality_summary = f"Rows={len(df0)}, Cols={len(df0.columns)}; Missing required fields: {missing if missing else 'None'}"
    return df0, issues_df, quality_summary, colmap