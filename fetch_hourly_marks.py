"""Fetch hourly option marks for the contracts in a transaction export.

Reads the export, derives the OCC symbol for every contract traded, pulls
hourly bars for each one over its holding period, and writes a JSON file that
portfolio_sharpe.hourly_sharpe() consumes directly.

Run this on your own machine. It will not work from a restricted network.

Providers (all have a free path for a window this recent):

  alpaca      Free "Basic" plan. Option bars at 1Hour, ~2 years of history,
              anything older than 15 minutes. Signup only, no funding.
              Keys: ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY

  marketdata  "Free Forever" plan: 100 requests/day, hourly candles, data at
              least 24h old and less than 1 year old. 40 contracts fits in
              one day's quota.
              Key: MARKETDATA_TOKEN

  databento   $125 signup credit, OPRA OHLCV-1h, 10 years of history. Highest
              fidelity (consolidated NBBO across all 17 venues). Needs the
              databento package: pip install databento
              Key: DATABENTO_API_KEY

Usage:
    export ALPACA_API_KEY_ID=... ALPACA_API_SECRET_KEY=...
    python fetch_hourly_marks.py export.csv --provider alpaca -o marks.json

    python portfolio_sharpe.py export.csv --marks marks.json --capital 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from portfolio_sharpe import Contract, load_legs

USER_AGENT = "portfolio-sharpe/1.0"


def _get_json(url: str, headers: dict[str, str], retries: int = 4) -> dict:
    """GET with backoff on rate limits and transient server errors."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:300]
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    {exc.code}, retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise SystemExit(f"HTTP {exc.code} for {url}\n{body}") from exc
        except urllib.error.URLError as exc:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise SystemExit(f"network error for {url}: {exc.reason}") from exc

    raise SystemExit(f"gave up on {url}")


def _floor_hour(stamp: datetime) -> datetime:
    """Normalize to a UTC hour so providers with different bar stamps align."""
    return stamp.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def fetch_alpaca(
    symbols: list[str], start: date, end: date
) -> dict[str, dict[datetime, float]]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not (key and secret):
        raise SystemExit("set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")

    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    marks: dict[str, dict[datetime, float]] = defaultdict(dict)

    # The bars endpoint takes a comma-separated batch, so chunk rather than
    # issuing one request per contract.
    for batch_start in range(0, len(symbols), 20):
        batch = symbols[batch_start : batch_start + 20]
        page_token = None

        while True:
            params = {
                "symbols": ",".join(batch),
                "timeframe": "1Hour",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": "10000",
            }
            if page_token:
                params["page_token"] = page_token

            url = (
                "https://data.alpaca.markets/v1beta1/options/bars?"
                + urllib.parse.urlencode(params)
            )
            payload = _get_json(url, headers)

            for symbol, bars in (payload.get("bars") or {}).items():
                for bar in bars:
                    stamp = _floor_hour(
                        datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
                    )
                    marks[symbol][stamp] = float(bar["c"])

            page_token = payload.get("next_page_token")
            if not page_token:
                break

        print(f"  batch {batch_start // 20 + 1}: {len(batch)} symbols")

    return marks


def fetch_marketdata(
    symbols: list[str], start: date, end: date
) -> dict[str, dict[datetime, float]]:
    token = os.environ.get("MARKETDATA_TOKEN")
    if not token:
        raise SystemExit("set MARKETDATA_TOKEN")

    headers = {"Authorization": f"Bearer {token}"}
    marks: dict[str, dict[datetime, float]] = defaultdict(dict)

    for index, symbol in enumerate(symbols, 1):
        params = {"from": start.isoformat(), "to": end.isoformat()}
        url = (
            f"https://api.marketdata.app/v1/options/candles/1H/{symbol}/?"
            + urllib.parse.urlencode(params)
        )
        payload = _get_json(url, headers)

        status = payload.get("s")
        if status == "no_data":
            print(f"  [{index}/{len(symbols)}] {symbol}: no data")
            continue
        if status != "ok":
            print(f"  [{index}/{len(symbols)}] {symbol}: {payload}", file=sys.stderr)
            continue

        for epoch, close in zip(payload["t"], payload["c"]):
            stamp = _floor_hour(datetime.fromtimestamp(epoch, tz=timezone.utc))
            marks[symbol][stamp] = float(close)

        print(f"  [{index}/{len(symbols)}] {symbol}: {len(payload['t'])} bars")

    return marks


def fetch_databento(
    symbols: list[str], start: date, end: date
) -> dict[str, dict[datetime, float]]:
    try:
        import databento
    except ImportError:
        raise SystemExit("pip install databento") from None

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("set DATABENTO_API_KEY")

    client = databento.Historical(key)
    marks: dict[str, dict[datetime, float]] = defaultdict(dict)

    store = client.timeseries.get_range(
        dataset="OPRA.PILLAR",
        schema="ohlcv-1h",
        symbols=symbols,
        stype_in="raw_symbol",
        start=start.isoformat(),
        end=end.isoformat(),
    )

    frame = store.to_df()
    for row in frame.itertuples():
        stamp = _floor_hour(row.Index.to_pydatetime())
        # Databento fixed-point prices carry 9 implied decimals.
        marks[str(row.symbol)][stamp] = float(row.close) / 1e9

    return marks


PROVIDERS = {
    "alpaca": fetch_alpaca,
    "marketdata": fetch_marketdata,
    "databento": fetch_databento,
}


def holding_window(legs) -> tuple[date, date]:
    """Widen the trade range by a day on each side so entry/exit hours exist."""
    first = min(leg.trade_date for leg in legs)
    last = max(leg.trade_date for leg in legs)
    return first - timedelta(days=1), last + timedelta(days=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Robinhood transaction export")
    parser.add_argument(
        "--provider", choices=sorted(PROVIDERS), default="alpaca"
    )
    parser.add_argument("-o", "--output", default="marks.json")
    args = parser.parse_args()

    legs, _ = load_legs(args.csv_path)
    if not legs:
        raise SystemExit("no option legs found")

    contracts: dict[str, Contract] = {
        leg.contract.occ_symbol: leg.contract for leg in legs
    }
    symbols = sorted(contracts)
    start, end = holding_window(legs)

    print(f"{len(symbols)} contracts, {start} to {end}, via {args.provider}")
    marks = PROVIDERS[args.provider](symbols, start, end)

    covered = sum(1 for symbol in symbols if marks.get(symbol))
    total_bars = sum(len(v) for v in marks.values())
    print(f"\n{covered}/{len(symbols)} contracts covered, {total_bars} bars")

    if covered < len(symbols):
        print("\nMissing marks for:", file=sys.stderr)
        for symbol in symbols:
            if not marks.get(symbol):
                print(f"  {symbol}  ({contracts[symbol]})", file=sys.stderr)
        print(
            "\nA contract with no bars usually means it never traded during "
            "some hours, or the provider's history does not reach that far "
            "back. hourly_sharpe() requires full coverage.",
            file=sys.stderr,
        )

    serializable = {
        symbol: {stamp.isoformat(): price for stamp, price in sorted(bars.items())}
        for symbol, bars in marks.items()
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
