from __future__ import annotations
import gradio as gr
import pandas as pd

from llm_provider import CHAT_LLM, SUMMARY_NOTICE
from mcp import mcp_fetch_sanctions, mcp_fetch_high_risk_mcc
from threat_intel import ThreatIntel
from ttp_guard import TTPGuard, GuardDecision, default_guard
from modules.transactions import prepare_transactions, detect_transactions
from modules.kyc import prepare_kyc, detect_kyc
from modules.sanctions import prepare_sanctions, detect_sanctions, DEMO_SANCTIONS
from modules.credit import prepare_credit, detect_credit
from agent import build_agent
from tools import build_tools
from langchain.schema import SystemMessage, HumanMessage

# ---------- Summarizer ----------
SUMMARY_SYS = "You are a helpful Fraud/Risk analyst. Be concise (<120 words), list key counts, drivers, and data quality caveats."

def summarize_ai(context: str) -> str:
    if CHAT_LLM is None:
        return SUMMARY_NOTICE
    # Guard summaries as well (low severity just annotate)
    decision = default_guard.inspect_input(context)
    if decision.action == GuardDecision.BLOCK:
        return f"Blocked by TTP Guard: {decision.reason}"
    try:
        out = CHAT_LLM.invoke([SystemMessage(content=SUMMARY_SYS), HumanMessage(content=context[:4000])])
        return getattr(out, "content", str(out))
    except Exception:
        return SUMMARY_NOTICE

# ---------- TI + Guard singletons ----------
TI = ThreatIntel.load()   # pulls MCP envs if set, else defaults
GUARD = default_guard

# ---------- Pipelines (tabs) ----------
def run_transactions(file):
    try:
        from validation import _read_csv_any
        df = _read_csv_any(file)
        clean, issues, quality, colmap = prepare_transactions(df)
        mcc_list = mcp_fetch_high_risk_mcc() or TI.high_risk_mcc
        flagged, stats = detect_transactions(clean, colmap, mcc_list)
        ctx = f"[Transactions]\n{stats}\nQuality: {quality}\nHead:\n{clean.head(5).to_csv(index=False)}\nFlagged:\n{flagged.head(5).to_csv(index=False)}"
        ai = summarize_ai(ctx)
        return ai, stats, flagged, issues
    except Exception as e:
        return f"Error: {e}", "Validation failed.", pd.DataFrame(), pd.DataFrame()

def run_kyc(file):
    try:
        from validation import _read_csv_any
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
        from validation import _read_csv_any
        df = _read_csv_any(customers_file)
        clean, issues, quality, colmap = prepare_sanctions(df)
        sanc_df = mcp_fetch_sanctions() or ( _read_csv_any(sanctions_file) if sanctions_file else None ) or TI.sanctions_df or DEMO_SANCTIONS
        flagged, stats = detect_sanctions(clean, colmap, sanc_df)
        ctx = f"[Sanctions]\n{stats}\nQuality: {quality}\nHead:\n{clean.head(5).to_csv(index=False)}\nMatches:\n{flagged.head(5).to_csv(index=False)}"
        ai = summarize_ai(ctx)
        return ai, stats, flagged, issues
    except Exception as e:
        return f"Error: {e}", "Validation failed.", pd.DataFrame(), pd.DataFrame()

def run_credit(file):
    try:
        from validation import _read_csv_any
        df = _read_csv_any(file)
        clean, issues, quality, colmap = prepare_credit(df)
        flagged, stats = detect_credit(clean, colmap)
        ctx = f"[Credit]\n{stats}\nQuality: {quality}\nHead:\n{clean.head(5).to_csv(index=False)}\nFlagged:\n{flagged.head(5).to_csv(index=False)}"
        ai = summarize_ai(ctx)
        return ai, stats, flagged, issues
    except Exception as e:
        return f"Error: {e}", "Validation failed.", pd.DataFrame(), pd.DataFrame()

# ---------- Agent & tools ----------
TOOLS = build_tools()
AGENT = build_agent(TOOLS, GUARD)

def agent_reply(history, user_msg: str):
    # Guard the incoming user message before tool routing
    decision = GUARD.inspect_input(user_msg)
    if decision.action == GuardDecision.BLOCK:
        return f"❌ Blocked by TTP Guard: {decision.reason}"
    try:
        looks_like_csv = ("," in user_msg) and ("\n" in user_msg) and (user_msg.count(",") >= 2)
        prompt = f"CSV snippet detected. Decide tool and analyze:\n\n{user_msg}" if looks_like_csv else user_msg
        res = AGENT.invoke(prompt)
        if isinstance(res, dict) and "output" in res:
            return res["output"]
        return str(res)
    except Exception as e:
        return f"Agent error: {e}"

# ---------- UI ----------
with gr.Blocks(title="Fraud Detector Analyst — LangChain + Fireworks + MCP") as demo:
    gr.Markdown("# 🛡️ Fraud Detector Analyst — LangChain + Fireworks + MCP")
    with gr.Tabs():
        with gr.Tab("Transactions"):
            f = gr.File(file_types=[".csv"], label="Transactions CSV", type="binary")
            ai = gr.Textbox(label="AI Summary (requires inference)", value=SUMMARY_NOTICE, lines=6)
            st = gr.Textbox(label="Stats", lines=3)
            flagged = gr.Dataframe(label="Flagged Transactions")
            issues = gr.Dataframe(label="Data Quality Issues (row, field, issue, value)")
            f.upload(run_transactions, inputs=[f], outputs=[ai, st, flagged, issues])

        with gr.Tab("KYC"):
            f = gr.File(file_types=[".csv"], label="KYC CSV", type="binary")
            ai = gr.Textbox(label="AI Summary (requires inference)", value=SUMMARY_NOTICE, lines=6)
            st = gr.Textbox(label="Stats", lines=3)
            flagged = gr.Dataframe(label="Flagged KYC Rows")
            issues = gr.Dataframe(label="Data Quality Issues")
            f.upload(run_kyc, inputs=[f], outputs=[ai, st, flagged, issues])

        with gr.Tab("Sanctions/PEP"):
            cust = gr.File(file_types=[".csv"], label="Customers CSV", type="binary")
            sanc = gr.File(file_types=[".csv"], label="Sanctions/PEP CSV (optional)", type="binary")
            ai = gr.Textbox(label="AI Summary (requires inference)", value=SUMMARY_NOTICE, lines=6)
            st = gr.Textbox(label="Stats", lines=3)
            flagged = gr.Dataframe(label="Matches")
            issues = gr.Dataframe(label="Data Quality Issues")
            cust.upload(run_sanctions, inputs=[cust, sanc], outputs=[ai, st, flagged, issues])
            sanc.upload(run_sanctions, inputs=[cust, sanc], outputs=[ai, st, flagged, issues])

        with gr.Tab("Credit Risk"):
            f = gr.File(file_types=[".csv"], label="Credit CSV", type="binary")
            ai = gr.Textbox(label="AI Summary (requires inference)", value=SUMMARY_NOTICE, lines=6)
            st = gr.Textbox(label="Stats", lines=3)
            flagged = gr.Dataframe(label="Flagged Applicants")
            issues = gr.Dataframe(label="Data Quality Issues")
            f.upload(run_credit, inputs=[f], outputs=[ai, st, flagged, issues])

        with gr.Tab("AI Consultant (Agent)"):
            chatbot = gr.Chatbot(type="messages", label="Fraud AI Consultant")
            user_in = gr.Textbox(label="Message or CSV snippet")
            send_btn = gr.Button("Send")
            def _chat_fn(history, msg):
                reply = agent_reply(history, msg)
                history = (history or []) + [{"role":"user","content":msg}, {"role":"assistant","content":reply}]
                return history, ""
            send_btn.click(_chat_fn, inputs=[chatbot, user_in], outputs=[chatbot, user_in])

        with gr.Tab("Security & TI"):
            gr.Markdown("**TTP Guard policy & latest indicators**")
            gr.JSON(value=GUARD.describe_policy())
            gr.Dataframe(value=TI.sanctions_df.head(10) if TI.sanctions_df is not None else pd.DataFrame({"note":["demo sanctions used"]}),
                         label="Sanctions (sample)")
            gr.Dataframe(value=pd.DataFrame({"high_risk_mcc": TI.high_risk_mcc}),
                         label="High-risk MCC (current)")

    gr.Markdown(
        "### ⚙️ Configure\n"
        "- `FIREWORKS_API_KEY` **or** `HF_TOKEN` (provider routing to Fireworks)\n"
        "- `FW_PRIMARY_MODEL` (default openai/gpt-oss-20b), `FW_SECONDARY_MODEL` (default Qwen/Qwen3-Coder-30B-A3B-Instruct)\n"
        "- MCP (optional): `ENABLE_MCP=1`, `MCP_SANCTIONS_URL`, `MCP_HIGH_RISK_MCC_URL`, `MCP_AUTH_HEADER`\n"
        "- TTP guard thresholds: `TTP_BLOCK_LEVEL` (default 3)\n"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)