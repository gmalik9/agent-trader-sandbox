-- agentic-trader sandbox schema
-- All timestamps are ISO8601 UTC strings.

CREATE TABLE IF NOT EXISTS accounts (
  id            INTEGER PRIMARY KEY,
  name          TEXT NOT NULL UNIQUE,         -- 'day'|'long'|'master'|'day_alpaca'|'long_alpaca'
  venue         TEXT NOT NULL,                -- 'sandbox'|'alpaca_paper'|'ledger'
  starting_cash REAL NOT NULL,
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
  id             INTEGER PRIMARY KEY,
  account_id     INTEGER NOT NULL REFERENCES accounts(id),
  ts             TEXT NOT NULL,
  symbol         TEXT NOT NULL,
  side           TEXT NOT NULL,                -- 'buy'|'sell'
  qty            REAL NOT NULL,
  order_type     TEXT NOT NULL,                -- 'market'|'limit'
  limit_price    REAL,
  tif            TEXT NOT NULL,                -- 'day'|'gtc'
  status         TEXT NOT NULL,                -- pending|filled|cancelled|rejected|routed_external
  submitted_at   TEXT NOT NULL,
  filled_at      TEXT,
  fill_price     REAL,
  fees           REAL NOT NULL DEFAULT 0,
  external_id    TEXT,
  agent          TEXT NOT NULL,                -- day|long|coordinator|manual
  thesis         TEXT,
  venue          TEXT NOT NULL,                -- sandbox|alpaca_paper
  dual_group_id  TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_dual_group ON orders(dual_group_id);
CREATE INDEX IF NOT EXISTS idx_orders_account_ts ON orders(account_id, ts);

CREATE TABLE IF NOT EXISTS cash_ledger (
  id            INTEGER PRIMARY KEY,
  account_id    INTEGER NOT NULL REFERENCES accounts(id),
  ts            TEXT NOT NULL,
  delta         REAL NOT NULL,
  reason        TEXT NOT NULL,                -- deposit|fill|fee|transfer
  ref_order_id  INTEGER REFERENCES orders(id)
);

CREATE INDEX IF NOT EXISTS idx_cash_account ON cash_ledger(account_id);

CREATE TABLE IF NOT EXISTS positions_snapshot (
  id            INTEGER PRIMARY KEY,
  account_id    INTEGER NOT NULL REFERENCES accounts(id),
  ts            TEXT NOT NULL,
  symbol        TEXT NOT NULL,
  qty           REAL NOT NULL,
  avg_cost      REAL NOT NULL,
  mark_price    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pos_account_ts ON positions_snapshot(account_id, ts);

CREATE TABLE IF NOT EXISTS equity_curve (
  id              INTEGER PRIMARY KEY,
  account_id      INTEGER NOT NULL REFERENCES accounts(id),
  ts              TEXT NOT NULL,
  cash            REAL NOT NULL,
  positions_value REAL NOT NULL,
  equity          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_equity_account_ts ON equity_curve(account_id, ts);

CREATE TABLE IF NOT EXISTS agent_runs (
  id            INTEGER PRIMARY KEY,
  account_id    INTEGER REFERENCES accounts(id),
  ts            TEXT NOT NULL,
  agent         TEXT NOT NULL,                -- day|long|coordinator
  status        TEXT NOT NULL,                -- ok|halted|error
  prompt        TEXT NOT NULL,
  response      TEXT,
  tools_called  TEXT,                          -- JSON
  decisions     TEXT,                          -- JSON
  error         TEXT,
  latency_ms    INTEGER
);

CREATE TABLE IF NOT EXISTS tick_requests (
  id           INTEGER PRIMARY KEY,
  ts           TEXT NOT NULL,
  agent        TEXT NOT NULL,                  -- day|long|coordinator
  requested_by TEXT NOT NULL DEFAULT 'ui',
  consumed_at  TEXT
);

CREATE TABLE IF NOT EXISTS settings (
  key          TEXT PRIMARY KEY,
  value        TEXT NOT NULL,
  updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dual_divergence (
  id            INTEGER PRIMARY KEY,
  dual_group_id TEXT NOT NULL,
  ts            TEXT NOT NULL,
  kind          TEXT NOT NULL,                 -- secondary_error|fill_price|fill_qty|status
  primary_val   TEXT,
  secondary_val TEXT,
  note          TEXT
);

CREATE INDEX IF NOT EXISTS idx_div_group ON dual_divergence(dual_group_id);

-- Per-position trade plan: the stop (and optional take-profit target) the agent
-- set at entry. The intraday stop monitor reads active rows every tick and
-- closes any position whose live price has breached its stop. One active row per
-- (account_id, symbol); superseded when the agent re-enters or the position goes
-- flat.
CREATE TABLE IF NOT EXISTS position_plans (
  id            INTEGER PRIMARY KEY,
  account_id    INTEGER NOT NULL REFERENCES accounts(id),
  symbol        TEXT NOT NULL,
  side          TEXT NOT NULL,                 -- 'buy' (long) | 'sell' (short)
  entry_price   REAL NOT NULL,
  stop_price    REAL NOT NULL,
  target_price  REAL,
  agent         TEXT NOT NULL,                 -- day|long
  active        INTEGER NOT NULL DEFAULT 1,    -- 1=monitored, 0=closed/superseded
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL,
  closed_at     TEXT,
  close_reason  TEXT                            -- stop_hit|target_hit|flat|superseded
);

CREATE INDEX IF NOT EXISTS idx_plans_active ON position_plans(account_id, active);
CREATE UNIQUE INDEX IF NOT EXISTS idx_plans_active_sym
  ON position_plans(account_id, symbol) WHERE active = 1;
