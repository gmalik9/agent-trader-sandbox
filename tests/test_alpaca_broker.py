from src.brokers.alpaca_paper_broker import AlpacaPaperBroker
from src.brokers.base import OrderRequest


class FakeMCP:
    def __init__(self, fail: bool = False, blocked: bool = False):
        self.fail = fail
        self.blocked = blocked
        self.calls: list[dict] = []

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


def test_alpaca_get_account_maps_sub_account_name(tmp_db):
    b = AlpacaPaperBroker(FakeMCP(), conn=tmp_db)
    snap = b.get_account("day")
    assert snap.name == "day_alpaca" and snap.venue == "alpaca_paper"
    assert snap.equity == 50_000.0 and snap.cash == 10_000.0
