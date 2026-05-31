"""Streamlit dashboard — read-only over SQLite.

Phase 7 expands this. For Phase 1 we render a placeholder so `bash run.sh`
boots end-to-end.
"""

from __future__ import annotations

import streamlit as st

from src.config import get_settings

st.set_page_config(page_title="Agentic Trader", layout="wide")

settings = get_settings()

st.title("Agentic Trader — Sandbox")
st.caption("Paper / simulated only. No live-money path exists.")

st.info(
    "Scaffolding only. Phases 2–7 wire up the sandbox, MCP clients, LLM, "
    "agents, scheduler, and dashboard tabs."
)

with st.expander("Active configuration"):
    st.json(
        {
            "broker_backend": settings.broker_backend,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "capital_total": settings.capital_total,
            "split_day_pct": settings.split_day_pct,
            "short_term_trader_path": settings.short_term_trader_path,
            "stock_recommender_path": settings.stock_recommender_path,
        }
    )
