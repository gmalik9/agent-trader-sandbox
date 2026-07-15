from src.brokers.alpaca_paper_broker import AlpacaPaperBroker
from src.brokers.base import OrderRequest

import pytest


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # The Alpaca leg retries transient failures with backoff; don't actually
    # wait during tests.
    import src.brokers.alpaca_paper_broker as ap
    monkeypatch.setattr(ap.time, "sleep", lambda *_: None)


class FakeMCP:
    def __init__(self, fail: bool = False, blocked: bool = False):
        self.fail = fail
        self.blocked = blocked
        self.calls: list[dict] = []
        self.restarts = 0

    def start(self):
        self.restarts += 1

    def stop(self):
        pass

    def get_account(self):
        return {"equity": 50_000.0, "cash": 10_000.0, "account_number": "PA12345"}

    def list_positions(self):
        return [{"symbol": "AAPL", "qty": 5, "entry_price": 100.0, "current_price": 110.0}]

    def place_order(self, *, symbol, qty, side, order_type="market",
                    time_in_force="day", limit_price=None):
        self.calls.append({"symbol": symbol, "qty": qty, "side": side,
                            "order_type": order_type})
        if self.fail:
            raise RuntimeError("simulated alpaca outage")
        if self.blocked:
            return {"blocked": "trading_disabled",
                    "message": "Trading is disabled. Set STOCK_REC_MCP_TRADING_ENABLED='true'"}
        return {"order_id": "abc123", "status": "filled",
                "filled_avg_price": 101.5, "filled_at": "2026-05-31T15:00:00Z"}

    def cancel_order(self, order_id):
        self.calls.append({"cancel": order_id})
        return {"ok": True}

    def close_position(self, symbol, percentage=100):
        self.calls.append({"close": symbol, "pct": percentage})
        return {"order_id": "close-1", "status": "filled"}


def test_alpaca_place_order_records_two_status_updates(tmp_db):
    mcp = FakeMCP()
    b = AlpacaPaperBroker(mcp, conn=tmp_db)
    res = b.place_order(OrderRequest("AAPL", "buy", 5, sub_account="day", agent="test"))
    assert res.status == "filled" and res.external_id == "abc123"
    row = tmp_db.execute("SELECT status, venue, external_id FROM orders WHERE id=?", (res.id,)).fetchone()
    assert row["status"] == "filled" and row["venue"] == "alpaca_paper" and row["external_id"] == "abc123"


def test_alpaca_place_order_records_rejection_on_failure(tmp_db):
    mcp = FakeMCP(fail=True)
    b = AlpacaPaperBroker(mcp, conn=tmp_db)
    res = b.place_order(OrderRequest("AAPL", "buy", 5, sub_account="day", agent="test"))
    assert res.status == "rejected"


def test_alpaca_blocked_response_is_rejected_not_routed(tmp_db):
    # When trading is disabled the MCP returns a blocked dict (no order id).
    # The broker must NOT leave the row as 'routed_external' (silent phantom).
    mcp = FakeMCP(blocked=True)
    b = AlpacaPaperBroker(mcp, conn=tmp_db)
    res = b.place_order(OrderRequest("AAPL", "buy", 5, sub_account="day", agent="test"))
    assert res.status == "rejected"
    assert res.external_id is None
    row = tmp_db.execute("SELECT status, thesis FROM orders WHERE id=?", (res.id,)).fetchone()
    assert row["status"] == "rejected"
    assert "alpaca_not_placed:trading_disabled" in row["thesis"]


class FlakyMCP(FakeMCP):
    """Returns trading_disabled for the first `flaky` calls, then fills."""

    def __init__(self, flaky: int = 2):
        super().__init__()
        self.flaky = flaky
        self.n = 0

    def place_order(self, **kw):
        self.calls.append(kw)
        self.n += 1
        if self.n <= self.flaky:
            return {"blocked": "trading_disabled", "message": "spurious"}
        return {"order_id": "ok-after-retry", "status": "filled",
                "filled_avg_price": 100.0}


def test_alpaca_retries_transient_trading_disabled_then_fills(tmp_db):
    mcp = FlakyMCP(flaky=2)
    b = AlpacaPaperBroker(mcp, conn=tmp_db)
    res = b.place_order(OrderRequest("AAPL", "buy", 5, sub_account="day", agent="test"))
    assert res.status == "filled" and res.external_id == "ok-after-retry"
    assert mcp.n == 3          # two transient blocks retried, third fills
    assert mcp.restarts >= 1   # MCP client was restarted to clear the gate


class NotShortableMCP(FakeMCP):
    def place_order(self, **kw):
        self.calls.append(kw)
        return {"error": 'asset "SOXL" cannot be sold short', "code": 42210000}


def test_alpaca_permanent_reject_not_retried(tmp_db):
    mcp = NotShortableMCP()
    b = AlpacaPaperBroker(mcp, conn=tmp_db)
    res = b.place_order(OrderRequest("SOXL", "sell", 5, sub_account="day", agent="test"))
    assert res.status == "rejected"
    # A permanent venue reject must NOT be retried (single attempt).
    assert len(mcp.calls) == 1


def test_alpaca_get_account_maps_sub_account_name(tmp_db):
    b = AlpacaPaperBroker(FakeMCP(), conn=tmp_db)
    snap = b.get_account("day")
    assert snap.name == "day_alpaca" and snap.venue == "alpaca_paper"
    assert snap.equity == 50_000.0 and snap.cash == 10_000.0


class FlakyCloseMCP(FakeMCP):
    """close_position returns trading_disabled for the first `flaky` calls, then
    accepts — models the spurious gate that also hits closes."""

    def __init__(self, flaky: int = 2):
        super().__init__()
        self.flaky = flaky
        self.closes = 0

    def close_position(self, symbol, percentage=100):
        self.closes += 1
        if self.closes <= self.flaky:
            return {"blocked": "trading_disabled", "message": "spurious"}
        return {"order_id": "close-ok", "status": "accepted"}


def test_alpaca_close_position_retries_transient_then_succeeds(tmp_db):
    # A stop-loss / exit close must NOT be abandoned on a spurious trading_disabled
    # — it should self-heal (restart MCP) and retry, like place_order does.
    mcp = FlakyCloseMCP(flaky=2)
    b = AlpacaPaperBroker(mcp, conn=tmp_db)
    res = b.close_position("AAPL", sub_account="day")
    assert res.status == "accepted" and res.external_id == "close-ok"
    assert mcp.closes == 3         # two transient blocks retried, third accepted
    assert mcp.restarts >= 1       # MCP client restarted to clear the gate
    row = tmp_db.execute("SELECT status FROM orders WHERE id=?", (res.id,)).fetchone()
    assert row["status"] == "accepted"


class OutageCloseMCP(FakeMCP):
    """close_position raises a transport error every time (persistent outage)."""

    def close_position(self, symbol, percentage=100):
        self.closes = getattr(self, "closes", 0) + 1
        raise RuntimeError("simulated close outage")


def test_alpaca_close_position_retries_transport_errors_then_rejects(tmp_db):
    mcp = OutageCloseMCP()
    b = AlpacaPaperBroker(mcp, conn=tmp_db)
    res = b.close_position("AAPL", sub_account="day")
    # Persistent outage → eventually rejected, but only AFTER multiple attempts.
    assert res.status == "rejected"
    assert mcp.closes >= 2         # retried, not a single give-up

