"""Broker interface and shared dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]
TIF = Literal["day", "gtc"]


@dataclass
class AccountSnapshot:
    name: str
    venue: str
    equity: float
    cash: float
    positions_value: float


@dataclass
class Position:
    symbol: str
    qty: float
    avg_cost: float
    mark_price: float
    unrealized_pnl: float


@dataclass
class OrderRequest:
    symbol: str
    side: Side
    qty: float
    order_type: OrderType = "market"
    limit_price: float | None = None
    tif: TIF = "day"
    agent: str = "manual"
    thesis: str | None = None
    sub_account: str = "day"  # 'day' | 'long' for routing
    dual_group_id: str | None = None  # set by DualBroker; None for single-broker routes


@dataclass
class OrderResult:
    id: int
    external_id: str | None
    status: str
    fill_price: float | None
    fees: float
    venue: str
    dual_group_id: str | None = None


class BrokerBase(ABC):
    name: str

    @abstractmethod
    def get_account(self, sub_account: str = "day") -> AccountSnapshot: ...

    @abstractmethod
    def list_positions(self, sub_account: str = "day") -> list[Position]: ...

    @abstractmethod
    def place_order(self, req: OrderRequest) -> OrderResult: ...

    @abstractmethod
    def cancel_order(self, order_id: int) -> None: ...

    @abstractmethod
    def close_position(self, symbol: str, sub_account: str = "day",
                       percentage: float = 100.0) -> OrderResult: ...

    @abstractmethod
    def mark_to_market(self, now: datetime, sub_account: str = "day") -> AccountSnapshot: ...

    @abstractmethod
    def equity_curve(self, sub_account: str = "day",
                     since: datetime | None = None) -> pd.DataFrame: ...
