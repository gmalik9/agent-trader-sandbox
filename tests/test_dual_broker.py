from src.brokers.alpaca_paper_broker import AlpacaPaperBroker
from src.brokers.base import OrderRequest
from src.brokers.dual_broker import DualBroker
from src.brokers.sandbox_broker import SandboxBroker

from tests.test_alpaca_broker import FakeMCP


def _dual(tmp_db, stub_bars, *, fail_secondary: bool = False):
    primary = SandboxBroker(conn=tmp_db, bar_provider=stub_bars)
    secondary = AlpacaPaperBroker(FakeMCP(fail=fail_secondary), conn=tmp_db)
    return DualBroker(primary, secondary, conn=tmp_db)


def test_dual_fanout_creates_two_orders_with_shared_group_id(tmp_db, stub_bars):
    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    d = _dual(tmp_db, stub_bars)
    res = d.place_order(OrderRequest("AAPL", "buy", 5, sub_account="day", agent="test"))

    rows = tmp_db.execute(
        "SELECT venue, status, dual_group_id FROM orders WHERE dual_group_id = ?",
        (res.dual_group_id,),
    ).fetchall()
    venues = {r["venue"] for r in rows}
    assert venues == {"sandbox", "alpaca_paper"}
    assert len(rows) == 2
    assert all(r["dual_group_id"] == res.dual_group_id for r in rows)


def test_secondary_failure_does_not_rollback_primary(tmp_db, stub_bars):
    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    d = _dual(tmp_db, stub_bars, fail_secondary=True)
    res = d.place_order(OrderRequest("AAPL", "buy", 5, sub_account="day", agent="test"))

    # Primary returned successfully…
    assert res.status == "filled" and res.venue == "sandbox"
    # …secondary leg recorded as rejected…
    sec = tmp_db.execute(
        "SELECT status FROM orders WHERE dual_group_id=? AND venue='alpaca_paper'",
        (res.dual_group_id,),
    ).fetchone()
    assert sec["status"] == "rejected"
    # …and a divergence row was written. AlpacaPaperBroker catches MCP errors
    # internally and returns status='rejected', so DualBroker sees a status
    # mismatch (sandbox=filled vs alpaca=rejected) rather than a raw exception.
    div = tmp_db.execute(
        "SELECT kind FROM dual_divergence WHERE dual_group_id=?",
        (res.dual_group_id,),
    ).fetchone()
    assert div["kind"] in ("status", "secondary_error")


def test_dual_reads_default_to_primary(tmp_db, stub_bars):
    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    d = _dual(tmp_db, stub_bars)
    snap = d.get_account("day")
    assert snap.venue == "sandbox" and snap.name == "day"
    snap_sec = d.get_account("day", venue="secondary")
    assert snap_sec.venue == "alpaca_paper" and snap_sec.name == "day_alpaca"


def test_dual_cancel_targets_both_legs(tmp_db, stub_bars):
    stub_bars.set("AAPL", o=100, h=101, l=99, c=100)
    d = _dual(tmp_db, stub_bars)
    # Limit price far below the bar low → sandbox leg stays 'pending'.
    # Secondary fake always returns 'filled', so cancel only flips the sandbox row.
    res = d.place_order(OrderRequest("AAPL", "buy", 5, order_type="limit",
                                       limit_price=1.0, sub_account="day", agent="test"))
    d.cancel_order(res.id)
    rows = tmp_db.execute(
        "SELECT venue, status FROM orders WHERE dual_group_id=?",
        (res.dual_group_id,),
    ).fetchall()
    statuses = {r["venue"]: r["status"] for r in rows}
    assert statuses["sandbox"] == "cancelled"
    assert statuses["alpaca_paper"] == "filled"  # fake MCP returned filled; cancel is a no-op on filled rows
