from __future__ import annotations
import os, json, logging
from typing import Optional, List
import pandas as pd
from urllib.request import Request, urlopen

log = logging.getLogger("fraud-analyst")

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