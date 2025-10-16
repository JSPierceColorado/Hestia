"""
Hestia — S&P 500 Dip Buyer (Full Alpaca Automation, no Telegram)

Philosophy:
  • Scan S&P 500 on 15m bars. Buy oversold dips (RSI<=30 and SMA60<SMA240).
  • Place bracket orders with TP/SL immediately; no timeout.
  • Each new position uses 10% of CURRENT buying power (by notional).
  • No max positions, no sector caps.

Notes:
  • No CSV. S&P 500 symbols are fetched dynamically from Wikipedia.
  • Uses Alpaca Market Data v2 + Trading API.
  • Fractional shares enabled via notional sizing (set ALLOW_FRACTIONAL=true).

Env vars:
  ALPACA_API_KEY=...
  ALPACA_SECRET_KEY=...
  ALPACA_PAPER=true|false

  # Risk controls / sizing
  ALLOW_FRACTIONAL=true|false         # if false, uses share qty instead of notional
  NOTIONAL_PCT_PER_TRADE=10           # 10% of current buying power
  TP_PCT=8                            # +8% take-profit
  SL_PCT=5                            # -5% stop-loss

  # Scanner cadence
  SCAN_SECONDS=900                    # 15 minutes
  MANAGE_SECONDS=60                   # 1 minute for housekeeping

  # State path for persistence
  STATE_PATH=/mnt/data/hestia_state.json
"""
from __future__ import annotations
import os, json, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import pandas_ta as ta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() in {"1","true","yes"}

STATE_PATH = Path(os.getenv("STATE_PATH", "/mnt/data/hestia_state.json"))

ALLOW_FRACTIONAL = os.getenv("ALLOW_FRACTIONAL", "true").lower() in {"1","true","yes"}
NOTIONAL_PCT_PER_TRADE = float(os.getenv("NOTIONAL_PCT_PER_TRADE", "10"))
TP_PCT = float(os.getenv("TP_PCT", "8"))
SL_PCT = float(os.getenv("SL_PCT", "5"))

SCAN_SECONDS = int(os.getenv("SCAN_SECONDS", "900"))
MANAGE_SECONDS = int(os.getenv("MANAGE_SECONDS", "60"))

TIMEFRAME = TimeFrame(15, TimeFrameUnit.Minute)
LOOKBACK_DAYS_STOCK = 30

BOT_TAG = "HESTIA"

# ──────────────────────────────────────────────────────────────────────────────
# Clients
# ──────────────────────────────────────────────────────────────────────────────
if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
    raise SystemExit("Missing ALPACA credentials")

stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
trading = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)

# ──────────────────────────────────────────────────────────────────────────────
# Universe
# ──────────────────────────────────────────────────────────────────────────────
FALLBACK_SP500 = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","JPM","AVGO","UNH","XOM","SPY"
]

def load_sp500() -> List[str]:
    """Fetch S&P 500 symbols dynamically (no CSV). Falls back to small static list on error."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        col = [c for c in df.columns if str(c).lower().startswith("symbol")][0]
        syms = (
            df[col]
            .dropna()
            .astype(str)
            .str.upper()
            .str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
            .unique()
            .tolist()
        )
        return sorted(syms)
    except Exception as e:
        print(f"[WARN] load_sp500 fell back to static list: {e}")
        return FALLBACK_SP500

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
def load_state() -> Dict[str,Any]:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text())
    except Exception:
        pass
    return {"positions": {}}  # keyed by symbol

def save_state(state: Dict[str,Any]):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state))

state = load_state()

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
def to_df_from_response(resp, symbol: str) -> pd.DataFrame:
    if hasattr(resp, "data") and resp.data:
        bars_list = resp.data.get(symbol, [])
        rows = [{
            "t": b.timestamp,
            "o": float(b.open),
            "h": float(b.high),
            "l": float(b.low),
            "c": float(b.close),
            "v": float(getattr(b, "volume", np.nan)) if getattr(b, "volume", None) is not None else np.nan,
        } for b in bars_list]
        return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    if hasattr(resp, "df") and isinstance(resp.df, pd.DataFrame) and not resp.df.empty:
        df = resp.df.copy()
        if "symbol" in df.index.names:
            try:
                df = df.xs(symbol, level="symbol")
            except Exception:
                pass
        df = df.reset_index().rename(columns={
            "timestamp": "t", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"
        })
        keep = ["t","o","h","l","c","v"]
        return df[keep].sort_values("t").reset_index(drop=True)
    return pd.DataFrame(columns=["t","o","h","l","c","v"])

def fetch_15m(symbol: str, end: datetime) -> pd.DataFrame:
    start = end - timedelta(days=LOOKBACK_DAYS_STOCK)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        adjustment=Adjustment.SPLIT,
        feed=DataFeed.IEX if ALPACA_PAPER else DataFeed.SIP,
        limit=10000,
    )
    resp = stock_client.get_stock_bars(req)
    return to_df_from_response(resp, symbol)

# ──────────────────────────────────────────────────────────────────────────────
# Indicators & signals
# ──────────────────────────────────────────────────────────────────────────────
def hestia_signal(df: pd.DataFrame) -> bool:
    if len(df) < 240:
        return False
    c = df["c"].astype(float)
    rsi14 = ta.rsi(c, length=14).iloc[-1]
    sma60 = ta.sma(c, length=60).iloc[-1]
    sma240 = ta.sma(c, length=240).iloc[-1]
    return (rsi14 <= 30) and (sma60 < sma240)

# ──────────────────────────────────────────────────────────────────────────────
# Order helpers (10% BP per trade; bracket TP/SL)
# ──────────────────────────────────────────────────────────────────────────────
def get_buying_power() -> float:
    acct = trading.get_account()
    return float(acct.buying_power)

def place_bracket_buy(symbol: str, last_price: float) -> str:
    notional = get_buying_power() * (NOTIONAL_PCT_PER_TRADE / 100.0)
    if notional <= 0:
        raise RuntimeError("No buying power available")

    tp_price = round(last_price * (1 + TP_PCT/100.0), 2)
    sl_price = round(last_price * (1 - SL_PCT/100.0), 2)
    client_oid = f"{BOT_TAG}-{symbol}-{int(datetime.now(timezone.utc).timestamp())}"

    if ALLOW_FRACTIONAL:
        order = trading.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                notional=notional,
                client_order_id=client_oid,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp_price),
                stop_loss=StopLossRequest(stop_price=sl_price),
            )
        )
    else:
        qty = max(1, int(notional // max(last_price, 0.01)))
        order = trading.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                qty=qty,
                client_order_id=client_oid,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp_price),
                stop_loss=StopLossRequest(stop_price=sl_price),
            )
        )
    return order.id

# ──────────────────────────────────────────────────────────────────────────────
# Main loops
# ──────────────────────────────────────────────────────────────────────────────
def scan_and_trade():
    syms = load_sp500()
    now = datetime.now(timezone.utc)
    print(f"[INFO] Hestia scan at {now.isoformat()} — {len(syms)} symbols")
    for sym in syms:
        try:
            df = fetch_15m(sym, now)
            if df.empty or len(df) < 240:
                continue
            if hestia_signal(df):
                last = df["c"].iloc[-1]
                oid = place_bracket_buy(sym, float(last))
                state["positions"][sym] = {
                    "opened_at": now.isoformat(),
                    "entry_px": float(last),
                    "tp_pct": TP_PCT,
                    "sl_pct": SL_PCT,
                    "order_id": oid,
                }
                print(f"[ALERT] HESTIA BUY {sym} @ ~{last:.2f} (oid={oid})")
                save_state(state)
        except Exception as e:
            print(f"[ERROR] {sym} scan/trade failed: {e}")

def manage_positions():
    # With server-side brackets, TP/SL handled by Alpaca. Here we prune closed.
    try:
        positions = {p.symbol: p for p in trading.get_all_positions()}
        for sym in list(state["positions"].keys()):
            if sym not in positions:
                print(f"[INFO] Position closed: {sym}")
                del state["positions"][sym]
        save_state(state)
    except Exception as e:
        print(f"[WARN] manage_positions error: {e}")

def loop_forever():
    next_scan = 0
    while True:
        now_ts = time.time()
        if now_ts >= next_scan:
            scan_and_trade()
            next_scan = now_ts + SCAN_SECONDS
        manage_positions()
        time.sleep(MANAGE_SECONDS)

if __name__ == "__main__":
    loop_forever()
