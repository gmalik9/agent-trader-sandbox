"""Tests for `build_broker` — guards the wiring between the factory and each
broker's constructor signature (a mismatch here shipped a runtime crash that
unit tests missed because they built brokers directly)."""

from __future__ import annotations

import src.brokers.factory as factory
from src.brokers.dual_broker import DualBroker
from src.brokers.sandbox_broker import SandboxBroker
from src.brokers.alpaca_paper_broker import AlpacaPaperBroker

from tests.test_alpaca_broker import FakeMCP


def _patch_backend(monkeypatch, backend: str) -> None:
    class _S:
        broker_backend = backend
    monkeypatch.setattr(factory, "get_settings", lambda: _S())


def test_build_sandbox(tmp_db, monkeypatch):
    _patch_backend(monkeypatch, "sandbox")
    broker = factory.build_broker(tmp_db, long_term_client=None)
    assert isinstance(broker, SandboxBroker)


def test_build_alpaca_paper(tmp_db, monkeypatch):
    _patch_backend(monkeypatch, "alpaca_paper")
    broker = factory.build_broker(tmp_db, long_term_client=FakeMCP())
    assert isinstance(broker, AlpacaPaperBroker)


def test_build_dual(tmp_db, monkeypatch):
    _patch_backend(monkeypatch, "dual")
    broker = factory.build_broker(tmp_db, long_term_client=FakeMCP())
    assert isinstance(broker, DualBroker)
    assert isinstance(broker.primary, SandboxBroker)
    assert isinstance(broker.secondary, AlpacaPaperBroker)
    assert broker.conn is tmp_db


def test_dual_without_client_falls_back_to_sandbox(tmp_db, monkeypatch):
    _patch_backend(monkeypatch, "dual")
    broker = factory.build_broker(tmp_db, long_term_client=None)
    assert isinstance(broker, SandboxBroker)


def test_alpaca_without_client_falls_back_to_sandbox(tmp_db, monkeypatch):
    _patch_backend(monkeypatch, "alpaca_paper")
    broker = factory.build_broker(tmp_db, long_term_client=None)
    assert isinstance(broker, SandboxBroker)
