"""Tests for the smart-money (insider + political) activity client.

These use a stub httpx.Client so no network calls are made.
"""

from __future__ import annotations

from src.smart_money import SmartMoneyClient, _amount_midpoint, _classify_side


class _StubResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


class _StubHttp:
    """Routes GET calls to canned payloads keyed by a substring of the URL."""

    def __init__(self, routes: dict[str, object]):
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url, params=None):
        self.calls.append(url)
        for frag, payload in self.routes.items():
            if frag in url:
                if isinstance(payload, tuple):
                    body, code = payload
                    return _StubResponse(body, code)
                return _StubResponse(payload)
        return _StubResponse([], 200)

    def close(self):
        pass


def test_classify_side_maps_form4_and_ranges():
    assert _classify_side("P-Purchase") == "buy"
    assert _classify_side("S-Sale") == "sell"
    assert _classify_side("Grant") == "buy"
    assert _classify_side("") == "other"


def test_amount_midpoint_of_congressional_range():
    assert _amount_midpoint("$1,001 - $15,000") == 8000.5
    assert _amount_midpoint("$50,000") == 50000.0
    assert _amount_midpoint("") == 0.0


def test_no_key_degrades_gracefully():
    sm = SmartMoneyClient(client=_StubHttp({}))
    assert sm.available is False
    out = sm.symbol_activity("AAPL")
    assert out["available"] is False and out["reason"] == "no_api_key"
    assert sm.market_activity()["available"] is False
    assert sm.person_positions("Pelosi")["available"] is False


def test_symbol_activity_combines_insider_and_political():
    http = _StubHttp({
        "v4/insider-trading": [
            {"symbol": "NVDA", "reportingName": "CEO Jane", "typeOfOwner": "officer",
             "transactionType": "P-Purchase", "securitiesTransacted": "1000", "price": "100",
             "transactionDate": "2025-01-10"},
            {"symbol": "NVDA", "reportingName": "CFO Bob", "typeOfOwner": "officer",
             "transactionType": "S-Sale", "securitiesTransacted": "200", "price": "100",
             "transactionDate": "2025-01-11"},
        ],
        "v4/senate-trading": [
            {"symbol": "NVDA", "representative": "Sen. Smith", "type": "Purchase",
             "amount": "$15,001 - $50,000", "transactionDate": "2025-01-09"},
        ],
        "v4/senate-disclosure": [],
    })
    sm = SmartMoneyClient(fmp_api_key="k", lookback_days=100000, client=http)
    out = sm.symbol_activity("NVDA")
    assert out["available"] is True
    # insider: 1000*100 buy - 200*100 sell = +80,000 net -> bullish
    assert out["insider"]["net_value"] == 80000.0
    assert out["insider"]["bias"] == "bullish"
    # political: one purchase in the 15,001-50,000 range -> positive net
    assert out["political"]["buys"] == 1
    assert out["overall_bias"] == "bullish"


def test_political_absent_without_fmp_key_but_insider_via_finnhub():
    http = _StubHttp({
        "stock/insider-transactions": {"data": [
            {"name": "CEO Jane", "share": 500, "change": 500,
             "transactionPrice": 20, "transactionDate": "2025-01-10"},
        ]},
    })
    sm = SmartMoneyClient(finnhub_api_key="k", lookback_days=100000, client=http)
    assert sm.available is True
    assert sm.political_available is False
    out = sm.symbol_activity("AAPL")
    assert out["insider"]["net_value"] == 10000.0
    assert out["political"]["available"] is False


def test_market_activity_ranks_by_abs_net_flow():
    http = _StubHttp({
        "v4/insider-trading": [
            {"symbol": "AAA", "reportingName": "x", "transactionType": "P-Purchase",
             "securitiesTransacted": "10", "price": "10", "transactionDate": "2025-01-10"},
            {"symbol": "BBB", "reportingName": "y", "transactionType": "S-Sale",
             "securitiesTransacted": "100", "price": "10", "transactionDate": "2025-01-10"},
        ],
        "senate-trading-rss-feed": [],
        "senate-disclosure-rss-feed": [],
    })
    sm = SmartMoneyClient(fmp_api_key="k", lookback_days=100000, client=http)
    out = sm.market_activity(limit=5)
    assert out["available"] is True
    syms = [s["symbol"] for s in out["symbols"]]
    # BBB has |−1000| > AAA |100| so it ranks first
    assert syms[0] == "BBB"


def test_person_positions_aggregates_by_symbol():
    http = _StubHttp({
        "v4/insider-trading": [
            {"symbol": "AAA", "reportingName": "Jane Doe", "transactionType": "P-Purchase",
             "securitiesTransacted": "10", "price": "10", "transactionDate": "2025-01-10"},
            {"symbol": "AAA", "reportingName": "Jane Doe", "transactionType": "S-Sale",
             "securitiesTransacted": "3", "price": "10", "transactionDate": "2025-01-11"},
            {"symbol": "ZZZ", "reportingName": "Someone Else", "transactionType": "P-Purchase",
             "securitiesTransacted": "5", "price": "10", "transactionDate": "2025-01-10"},
        ],
        "senate-trading-rss-feed": [],
        "senate-disclosure-rss-feed": [],
        "v3/profile/AAA": [{"sector": "Technology"}],
    })
    sm = SmartMoneyClient(fmp_api_key="k", lookback_days=100000, client=http)
    out = sm.person_positions("jane doe")
    assert out["transactions"] == 2
    assert out["positions"][0]["symbol"] == "AAA"
    # 10*10 buy - 3*10 sell = +70 net
    assert out["positions"][0]["net_value"] == 70.0
    assert out["by_sector"][0]["sector"] == "Technology"
