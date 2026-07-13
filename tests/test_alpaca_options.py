"""Tests for the Alpaca options client safety guards + order recording."""

from __future__ import annotations

import pytest

from src.brokers.alpaca_options import AlpacaOptions, OptionsRecorder, OptionsSafetyError


def _client(monkeypatch, *, paper="true", key="k", secret="s"):
    class _S:
        alpaca_api_key_id = key
        alpaca_secret_key = secret
        alpaca_paper = paper
    monkeypatch.setattr("src.brokers.alpaca_options.get_settings", lambda: _S())
    return AlpacaOptions()


def test_refuses_when_not_paper(monkeypatch):
    c = _client(monkeypatch, paper="false")
    with pytest.raises(OptionsSafetyError):
        c._assert_paper()


def test_refuses_without_credentials(monkeypatch):
    c = _client(monkeypatch, key="", secret="")
    with pytest.raises(OptionsSafetyError):
        c._assert_paper()


def test_paper_flag_accepted(monkeypatch):
    c = _client(monkeypatch)
    c._assert_paper()  # should not raise


def test_recorder_marks_accepted_order(tmp_db):
    rec = OptionsRecorder(tmp_db)
    oid = rec.record(sub_account="day", occ_symbol="AAPL250620C00190000", side="buy",
                     qty=1, agent="day", thesis="bullish breakout",
                     resp={"id": "opt-123", "status": "accepted"})
    row = tmp_db.execute("SELECT symbol, status, external_id, venue FROM orders WHERE id=?",
                          (oid,)).fetchone()
    assert row["symbol"] == "AAPL250620C00190000"
    assert row["status"] == "accepted"
    assert row["external_id"] == "opt-123"
    assert row["venue"] == "alpaca_options"


def test_recorder_marks_rejected_when_no_id(tmp_db):
    rec = OptionsRecorder(tmp_db)
    oid = rec.record(sub_account="day", occ_symbol="AAPL250620P00150000", side="buy",
                     qty=1, agent="day", thesis="bearish", resp={"blocked": "x"})
    row = tmp_db.execute("SELECT status, thesis FROM orders WHERE id=?", (oid,)).fetchone()
    assert row["status"] == "rejected"
    assert "option_not_placed" in row["thesis"]


def test_recorder_marks_rejected_on_error(tmp_db):
    rec = OptionsRecorder(tmp_db)
    oid = rec.record(sub_account="day", occ_symbol="TSLA250620C00300000", side="buy",
                     qty=1, agent="day", thesis="", resp=None, error="boom")
    row = tmp_db.execute("SELECT status, thesis FROM orders WHERE id=?", (oid,)).fetchone()
    assert row["status"] == "rejected"
    assert "option_error:boom" in row["thesis"]
