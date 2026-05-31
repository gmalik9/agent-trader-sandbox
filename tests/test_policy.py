from src.agents.policy import Decision, validate
from src.brokers.sandbox_broker import SandboxBroker


def test_validate_rejects_blocklist(tmp_db, stub_bars):
    stub_bars.set("TQQQ", o=10, h=11, l=9, c=10)
    b = SandboxBroker(tmp_db, bar_provider=stub_bars)
    d = validate(Decision(symbol="TQQQ", side="buy", qty=10, limit_price=10.0),
                 b, "day")
    assert not d.accepted and d.reject_reason == "blocklist"


def test_validate_rejects_max_symbol_pct(tmp_db, stub_bars):
    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    b = SandboxBroker(tmp_db, bar_provider=stub_bars)
    # 30k @ 30% of 30k equity = 100% → way over the 25% default cap.
    d = validate(Decision(symbol="AAPL", side="buy", qty=300, limit_price=100.0),
                 b, "day", max_symbol_pct=25.0, max_order_usd=1_000_000.0)
    assert not d.accepted and d.reject_reason.startswith("max_symbol_pct")


def test_validate_rejects_qty_zero(tmp_db, stub_bars):
    b = SandboxBroker(tmp_db, bar_provider=stub_bars)
    d = validate(Decision(symbol="AAPL", side="buy", qty=0, limit_price=10.0), b, "day")
    assert not d.accepted and d.reject_reason == "qty<=0"


def test_validate_rejects_short_sell(tmp_db, stub_bars):
    b = SandboxBroker(tmp_db, bar_provider=stub_bars)
    d = validate(Decision(symbol="MSFT", side="sell", qty=10, limit_price=100.0), b, "day")
    assert not d.accepted and d.reject_reason == "no_shorting"
