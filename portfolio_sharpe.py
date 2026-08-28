"""Sharpe ratio analysis for an options portfolio from a Robinhood transaction export.

The export is a transaction log, not an equity curve: it records fills with a
date but no intraday timestamp, and it never states the account balance. Both
facts constrain what can honestly be computed, so this module is explicit about
which numbers come from the data and which require an assumption.

What it does:
  1. Parses option legs out of the CSV.
  2. FIFO-matches opening fills against closing fills, per contract.
  3. Pairs legs into vertical spreads to recover the capital at risk.
  4. Computes Sharpe over the realized-P&L series at a chosen resampling.

Positions still open at the end of the file are reported but excluded from the
realized series, since marking them requires option quotes this file lacks.

Usage:
    python portfolio_sharpe.py "Aug 1, 2026 - Aug 27, 2026.csv"
    python portfolio_sharpe.py trades.csv --risk-free 0.042 --capital 5000
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime

# Trading-period counts used to annualize. The hourly figure assumes a 6.5-hour
# regular session, which is the convention for US equity-option marks.
TRADING_DAYS_PER_YEAR = 252
MARKET_HOURS_PER_DAY = 6.5
TRADING_HOURS_PER_YEAR = TRADING_DAYS_PER_YEAR * MARKET_HOURS_PER_DAY

OPENING_CODES = {"BTO", "STO"}
CLOSING_CODES = {"BTC", "STC"}
LONG_CODES = {"BTO", "BTC"}

# "AMD 8/31/2026 Call $480.00"
CONTRACT_RE = re.compile(
    r"^(?P<underlying>\S+)\s+"
    r"(?P<expiry>\d{1,2}/\d{1,2}/\d{4})\s+"
    r"(?P<right>Call|Put)\s+"
    r"\$(?P<strike>[\d,]+(?:\.\d+)?)$"
)


def parse_money(raw: str) -> float:
    """Parse Robinhood's money format. Parentheses mean a debit."""
    text = raw.strip()
    if not text:
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace("$", "").replace(",", "")
    value = float(text)
    return -value if negative else value


@dataclass(frozen=True)
class Contract:
    underlying: str
    expiry: date
    right: str
    strike: float

    def __str__(self) -> str:
        return (
            f"{self.underlying} {self.expiry:%-m/%-d/%Y} "
            f"{self.right} ${self.strike:,.2f}"
        )

    @property
    def occ_symbol(self) -> str:
        """OCC/OSI symbol, e.g. AMD260831C00480000.

        Root, then YYMMDD expiry, then C or P, then the strike in thousandths
        padded to eight digits. Every options data vendor keys on this.
        """
        thousandths = round(self.strike * 1000)
        return (
            f"{self.underlying}"
            f"{self.expiry:%y%m%d}"
            f"{self.right[0].upper()}"
            f"{thousandths:08d}"
        )


@dataclass
class Leg:
    """A single option fill."""

    trade_date: date
    contract: Contract
    code: str
    quantity: int
    price: float
    amount: float  # signed cash flow, fees included

    @property
    def is_opening(self) -> bool:
        return self.code in OPENING_CODES

    @property
    def is_long(self) -> bool:
        return self.code in LONG_CODES

    @property
    def cash_per_contract(self) -> float:
        return self.amount / self.quantity


@dataclass
class RoundTrip:
    """One contract opened and later closed, matched FIFO."""

    contract: Contract
    quantity: int
    open_date: date
    close_date: date
    open_cash: float
    close_cash: float

    @property
    def pnl(self) -> float:
        return self.open_cash + self.close_cash

    @property
    def holding_days(self) -> int:
        return (self.close_date - self.open_date).days


@dataclass
class Spread:
    """Two same-underlying, same-expiry legs traded together on one date."""

    trade_date: date
    underlying: str
    expiry: date
    right: str
    long_leg: Leg
    short_leg: Leg

    @property
    def width(self) -> float:
        return abs(self.long_leg.contract.strike - self.short_leg.contract.strike)

    @property
    def quantity(self) -> int:
        return min(self.long_leg.quantity, self.short_leg.quantity)

    @property
    def strikes(self) -> frozenset[float]:
        return frozenset(
            {self.long_leg.contract.strike, self.short_leg.contract.strike}
        )

    @property
    def key(self) -> tuple:
        """Identity of the position, independent of when it was traded."""
        return (self.underlying, self.expiry, self.right, self.strikes)

    @property
    def is_opening(self) -> bool:
        return self.long_leg.is_opening

    @property
    def net_cash(self) -> float:
        return self.long_leg.amount + self.short_leg.amount

    @property
    def net_cash_per_unit(self) -> float:
        return self.net_cash / max(self.quantity, 1)

    @property
    def capital_at_risk(self) -> float:
        """Max loss on the vertical, which is what the broker holds against it.

        A debit vertical risks the debit paid. A credit vertical risks the width
        less the credit received.
        """
        if self.net_cash < 0:  # net debit paid
            return abs(self.net_cash)
        return self.width * 100 * self.quantity - self.net_cash

    @property
    def risk_per_unit(self) -> float:
        return self.capital_at_risk / max(self.quantity, 1)


@dataclass
class SpreadTrade:
    """One vertical opened and later closed, matched FIFO at spread level."""

    spread_key: tuple
    quantity: int
    open_date: date
    close_date: date
    open_cash_per_unit: float
    close_cash_per_unit: float
    risk_per_unit: float

    @property
    def pnl(self) -> float:
        return (self.open_cash_per_unit + self.close_cash_per_unit) * self.quantity

    @property
    def capital_at_risk(self) -> float:
        return self.risk_per_unit * self.quantity

    @property
    def return_on_risk(self) -> float:
        return self.pnl / self.capital_at_risk

    @property
    def label(self) -> str:
        underlying, expiry, right, strikes = self.spread_key
        lo, hi = min(strikes), max(strikes)
        return f"{underlying} {expiry:%-m/%-d} {right} {lo:g}/{hi:g}"


def load_legs(path: str) -> tuple[list[Leg], list[dict]]:
    """Read the CSV, returning option legs and any non-option rows."""
    legs: list[Leg] = []
    other: list[dict] = []

    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("Trans Code") or "").strip()
            if not code:
                continue  # trailing blank / disclaimer rows

            if code not in OPENING_CODES | CLOSING_CODES:
                other.append(row)
                continue

            match = CONTRACT_RE.match((row.get("Description") or "").strip())
            if not match:
                other.append(row)
                continue

            legs.append(
                Leg(
                    trade_date=datetime.strptime(
                        row["Activity Date"].strip(), "%m/%d/%Y"
                    ).date(),
                    contract=Contract(
                        underlying=match["underlying"],
                        expiry=datetime.strptime(match["expiry"], "%m/%d/%Y").date(),
                        right=match["right"],
                        strike=float(match["strike"].replace(",", "")),
                    ),
                    code=code,
                    quantity=int(row["Quantity"]),
                    price=parse_money(row["Price"]),
                    amount=parse_money(row["Amount"]),
                )
            )

    # The export is newest-first, and same-day rows are ordered latest-fill
    # first. Reversing recovers execution order, including within a day, which
    # FIFO matching depends on when a position is opened and closed the same
    # day. The stable sort then only guards against an out-of-order export.
    legs.reverse()
    legs.sort(key=lambda leg: leg.trade_date)
    return legs, other


def match_round_trips(legs: list[Leg]) -> tuple[list[RoundTrip], list[Leg]]:
    """FIFO-match closing fills against opening fills, one contract at a time.

    Returns completed round trips and the opening fills left unmatched (still
    open at the end of the file).
    """
    # A contract can be traded long on one day and short on another, so keep the
    # two directions in separate queues.
    open_lots: dict[tuple[Contract, bool], deque[list]] = defaultdict(deque)
    round_trips: list[RoundTrip] = []

    for leg in legs:
        if leg.is_opening:
            key = (leg.contract, leg.is_long)
            open_lots[key].append([leg, leg.quantity])
            continue

        # A BTC closes a short lot; an STC closes a long lot.
        key = (leg.contract, not leg.is_long)
        remaining = leg.quantity
        queue = open_lots[key]

        while remaining > 0 and queue:
            lot, lot_remaining = queue[0]
            take = min(remaining, lot_remaining)

            round_trips.append(
                RoundTrip(
                    contract=leg.contract,
                    quantity=take,
                    open_date=lot.trade_date,
                    close_date=leg.trade_date,
                    open_cash=lot.cash_per_contract * take,
                    close_cash=leg.cash_per_contract * take,
                )
            )

            remaining -= take
            queue[0][1] -= take
            if queue[0][1] == 0:
                queue.popleft()

        if remaining > 0:
            raise ValueError(
                f"{leg.trade_date}: closing {remaining} of {leg.contract} "
                "with no matching open position — the export may start "
                "mid-position."
            )

    still_open = [lot for queue in open_lots.values() for lot, qty in queue if qty]
    return round_trips, still_open


def build_spreads(legs: list[Leg]) -> tuple[list[Spread], list[Leg]]:
    """Pair each date's legs into verticals by underlying, expiry, and right."""
    buckets: dict[tuple, list[Leg]] = defaultdict(list)
    for leg in legs:
        key = (
            leg.trade_date,
            leg.contract.underlying,
            leg.contract.expiry,
            leg.contract.right,
            leg.is_opening,
        )
        buckets[key].append(leg)

    spreads: list[Spread] = []
    unpaired: list[Leg] = []

    for key, group in buckets.items():
        longs = [leg for leg in group if leg.is_long]
        shorts = [leg for leg in group if not leg.is_long]

        for long_leg, short_leg in zip(longs, shorts):
            spreads.append(
                Spread(
                    trade_date=key[0],
                    underlying=key[1],
                    expiry=key[2],
                    right=key[3],
                    long_leg=long_leg,
                    short_leg=short_leg,
                )
            )
        unpaired.extend(longs[len(shorts):] + shorts[len(longs):])

    spreads.sort(key=lambda s: s.trade_date)
    return spreads, unpaired


def match_spread_trades(
    spreads: list[Spread],
) -> tuple[list[SpreadTrade], list[tuple[Spread, int]]]:
    """FIFO-match closing verticals against opening verticals.

    Matching at spread level rather than leg level is what makes the P&L
    series meaningful: the two legs of a vertical are one position, and
    scoring them separately produces a long/short pair whose individual
    swings dwarf the net result.
    """
    queues: dict[tuple, deque[list]] = defaultdict(deque)
    trades: list[SpreadTrade] = []

    for spread in spreads:
        if spread.is_opening:
            queues[spread.key].append([spread, spread.quantity])
            continue

        remaining = spread.quantity
        queue = queues[spread.key]

        while remaining > 0 and queue:
            opener, lot_remaining = queue[0]
            take = min(remaining, lot_remaining)

            trades.append(
                SpreadTrade(
                    spread_key=spread.key,
                    quantity=take,
                    open_date=opener.trade_date,
                    close_date=spread.trade_date,
                    open_cash_per_unit=opener.net_cash_per_unit,
                    close_cash_per_unit=spread.net_cash_per_unit,
                    risk_per_unit=opener.risk_per_unit,
                )
            )

            remaining -= take
            queue[0][1] -= take
            if queue[0][1] == 0:
                queue.popleft()

        if remaining > 0:
            raise ValueError(
                f"{spread.trade_date}: closing {remaining} of "
                f"{spread.underlying} {spread.right} with no matching open "
                "position — the export may start mid-position."
            )

    still_open = [
        (spread, qty) for queue in queues.values() for spread, qty in queue if qty
    ]
    return trades, still_open


def daily_pnl(trades: list[SpreadTrade]) -> dict[date, float]:
    """Realized P&L bucketed by the date the position was closed."""
    series: dict[date, float] = defaultdict(float)
    for trade in trades:
        series[trade.close_date] += trade.pnl
    return dict(series)


def peak_concurrent_risk(
    trades: list[SpreadTrade], still_open: list[tuple[Spread, int]]
) -> float:
    """Largest capital committed on any single day.

    A position occupies capital from its open date through its close date
    inclusive, so a spread opened and closed the same day still counts. This
    is the denominator a margin desk would use.
    """
    by_day: dict[date, float] = defaultdict(float)

    for trade in trades:
        day = trade.open_date
        while day <= trade.close_date:
            by_day[day] += trade.capital_at_risk
            day = date.fromordinal(day.toordinal() + 1)

    for spread, qty in still_open:
        by_day[spread.trade_date] += spread.risk_per_unit * qty

    return max(by_day.values(), default=0.0)


def sharpe(
    returns: list[float],
    periods_per_year: float,
    risk_free_annual: float = 0.0,
) -> dict[str, float]:
    """Annualized Sharpe over a series of per-period simple returns."""
    n = len(returns)
    if n < 2:
        raise ValueError("need at least two periods")

    rf_period = (1 + risk_free_annual) ** (1 / periods_per_year) - 1
    excess = [r - rf_period for r in returns]

    mean = sum(excess) / n
    variance = sum((x - mean) ** 2 for x in excess) / (n - 1)  # sample stdev
    stdev = math.sqrt(variance)

    if stdev == 0:
        raise ValueError("zero volatility — Sharpe undefined")

    per_period = mean / stdev
    annualized = per_period * math.sqrt(periods_per_year)

    # Lo (2002): SE of a Sharpe estimate under iid returns. With a short sample
    # this dominates the interpretation, so it is reported alongside.
    stderr = math.sqrt((1 + 0.5 * per_period**2) / n) * math.sqrt(periods_per_year)

    return {
        "periods": n,
        "mean_period_return": mean,
        "stdev_period_return": stdev,
        "sharpe_per_period": per_period,
        "sharpe_annualized": annualized,
        "stderr_annualized": stderr,
        "ci95_low": annualized - 1.96 * stderr,
        "ci95_high": annualized + 1.96 * stderr,
    }


def format_report(
    legs: list[Leg],
    other_rows: list[dict],
    trades: list[SpreadTrade],
    still_open: list[tuple[Spread, int]],
    spreads: list[Spread],
    capital: float | None,
    risk_free: float,
    marks: dict | None = None,
) -> str:
    out: list[str] = []
    add = out.append

    pnl_by_day = daily_pnl(trades)
    trading_days = sorted(pnl_by_day)
    total_pnl = sum(pnl_by_day.values())

    open_spreads = [s for s in spreads if s.is_opening]
    avg_risk = (
        sum(s.capital_at_risk for s in open_spreads) / len(open_spreads)
        if open_spreads
        else 0.0
    )
    peak_risk = peak_concurrent_risk(trades, still_open)
    basis = capital if capital is not None else peak_risk

    add("=" * 72)
    add("OPTIONS PORTFOLIO — SHARPE ANALYSIS")
    add("=" * 72)
    add("")
    add(f"  Option legs parsed        {len(legs)}")
    add(f"  Vertical spreads opened   {len(open_spreads)}")
    add(f"  Closed round trips        {len(trades)}")
    add(f"  Still open at file end    {sum(q for _, q in still_open)}")
    add(f"  Underlyings               {len({l.contract.underlying for l in legs})}")
    add(f"  Date range                {legs[0].trade_date} to {legs[-1].trade_date}")
    add(f"  Trading days with a close {len(trading_days)}")
    for row in other_rows:
        add(
            f"  Non-option row            {row['Trans Code']} "
            f"{parse_money(row['Amount']):+,.2f} ({row['Description'][:40]})"
        )
    add("")

    add("-" * 72)
    add("CAPITAL BASE")
    add("-" * 72)
    add(f"  Avg risk per spread       ${avg_risk:,.2f}")
    add(f"  Peak concurrent risk      ${peak_risk:,.2f}")
    if capital is not None:
        add(f"  Account equity (given)    ${capital:,.2f}")
    else:
        add("  Account equity            not in file — using peak concurrent risk")
    add(f"  Return denominator        ${basis:,.2f}")
    add("")

    add("-" * 72)
    add("REALIZED P&L")
    add("-" * 72)
    wins = [t for t in trades if t.pnl > 0]
    best = max(trades, key=lambda t: t.pnl)
    worst = min(trades, key=lambda t: t.pnl)
    add(f"  Total realized P&L        ${total_pnl:,.2f}")
    add(f"  Return on basis           {total_pnl / basis:+.2%}")
    add(f"  Win rate                  {len(wins)}/{len(trades)} "
        f"({len(wins) / len(trades):.1%})")
    add(f"  Best trade                ${best.pnl:+,.2f}  ({best.label})")
    add(f"  Worst trade               ${worst.pnl:+,.2f}  ({worst.label})")
    add("")
    add("  Closed positions:")
    for trade in sorted(trades, key=lambda t: (t.close_date, t.label)):
        add(
            f"    {trade.open_date:%m/%d}->{trade.close_date:%m/%d} "
            f"{trade.label:<26} x{trade.quantity}  "
            f"${trade.pnl:>8,.2f}  on ${trade.capital_at_risk:>7,.2f} risk  "
            f"{trade.return_on_risk:>+8.2%}"
        )
    add("")
    add("  Daily realized P&L:")
    for day in trading_days:
        pnl = pnl_by_day[day]
        add(f"    {day}  ${pnl:>10,.2f}   {pnl / basis:>+8.3%}")
    add("")

    add("-" * 72)
    add("SHARPE — DAILY REGIME")
    add("-" * 72)
    add("  Series is realized P&L booked on the close date. Open positions are")
    add("  not marked between open and close, because the export carries no")
    add("  prices for days a position was merely held. That hides interim")
    add("  volatility, so this Sharpe is biased high.")
    add("")
    returns = [pnl_by_day[day] / basis for day in trading_days]
    try:
        stats = sharpe(returns, TRADING_DAYS_PER_YEAR, risk_free)
        add(f"  Periods                   {stats['periods']} days")
        add(f"  Mean daily return         {stats['mean_period_return']:+.4%}")
        add(f"  Daily volatility          {stats['stdev_period_return']:.4%}")
        add(f"  Sharpe (per day)          {stats['sharpe_per_period']:.4f}")
        add(f"  Sharpe (annualized)       {stats['sharpe_annualized']:.2f}")
        add(f"  95% CI                    [{stats['ci95_low']:.2f}, "
            f"{stats['ci95_high']:.2f}]")
    except ValueError as exc:
        add(f"  Not computable: {exc}")
    add("")

    add("-" * 72)
    add("SHARPE — PER ROUND TRIP")
    add("-" * 72)
    trip_returns = [t.return_on_risk for t in trades]
    if len(trip_returns) >= 2:
        # Per-trade Sharpe is left unannualized: trades are irregularly spaced,
        # so there is no honest periods-per-year factor to scale by.
        stats = sharpe(trip_returns, periods_per_year=1, risk_free_annual=0.0)
        add(f"  Trades                    {stats['periods']}")
        add(f"  Mean return on risk       {stats['mean_period_return']:+.2%}")
        add(f"  Stdev                     {stats['stdev_period_return']:.2%}")
        add(f"  Sharpe (per trade)        {stats['sharpe_per_period']:.4f}")
    add("")

    add("-" * 72)
    add("SHARPE — HOURLY REGIME")
    add("-" * 72)

    if marks:
        try:
            stats = hourly_sharpe(
                marks, legs, starting_cash=basis, risk_free_annual=risk_free
            )
            add(f"  Periods                   {stats['periods']} hours")
            add(f"  Mean hourly return        {stats['mean_period_return']:+.5%}")
            add(f"  Hourly volatility         {stats['stdev_period_return']:.5%}")
            add(f"  Sharpe (per hour)         {stats['sharpe_per_period']:.4f}")
            add(f"  Sharpe (annualized)       {stats['sharpe_annualized']:.2f}")
            add(f"  95% CI                    [{stats['ci95_low']:.2f}, "
                f"{stats['ci95_high']:.2f}]")
            add("")
            add("  Fills carry a date but no time, so each is applied at its")
            add("  trade date's close. Total P&L is exact; the hour within the")
            add("  entry and exit days is not recoverable from the export.")
        except (KeyError, ValueError) as exc:
            add(f"  Not computable: {exc}")
        add("")
        add("=" * 72)
        return "\n".join(out)

    add("  NOT COMPUTABLE from this file.")
    add("")
    session_hours = _session_hours(legs[0].trade_date, legs[-1].trade_date)
    contracts = len({str(leg.contract) for leg in legs})
    add("  The export carries Activity Date only — no intraday timestamp on any")
    add(f"  of the {len(legs)} fills — and no account equity series. An hourly")
    add(f"  Sharpe needs a portfolio value at each of the ~{session_hours:.0f} market")
    add(f"  hours in this window, which means marking all {contracts} contracts")
    add("  hour by hour.")
    add("")
    add(f"  Required: hourly NBBO marks for {contracts} contracts over their "
        "holding periods.")
    add("  Free sources for a window this recent: Alpaca (Basic plan),")
    add("  MarketData.app (Free Forever), Databento ($125 signup credit).")
    add("  Run fetch_hourly_marks.py, then pass --marks marks.json.")
    add("")
    add("  For scale, if the daily series above were re-expressed hourly with")
    add("  identical risk-adjusted performance, the annualization factor moves")
    add(f"  from sqrt({TRADING_DAYS_PER_YEAR}) = {math.sqrt(TRADING_DAYS_PER_YEAR):.1f} "
        f"to sqrt({TRADING_HOURS_PER_YEAR:.0f}) = "
        f"{math.sqrt(TRADING_HOURS_PER_YEAR):.1f}.")
    add("  The annualized Sharpe is invariant to that choice only if returns are")
    add("  iid; intraday option P&L is not, so the hourly number will differ.")
    add("")

    if still_open:
        add("-" * 72)
        add("OPEN AT FILE END (excluded from realized series)")
        add("-" * 72)
        for spread, qty in still_open:
            lo, hi = min(spread.strikes), max(spread.strikes)
            add(
                f"  opened {spread.trade_date}  x{qty}  {spread.underlying} "
                f"{spread.expiry:%-m/%-d} {spread.right} {lo:g}/{hi:g}   "
                f"${spread.risk_per_unit * qty:,.2f} at risk"
            )
        add("")

    add("=" * 72)
    return "\n".join(out)


def _session_hours(start: date, end: date) -> float:
    """Approximate market hours between two dates, ignoring holidays."""
    weekdays = sum(
        1
        for ordinal in range(start.toordinal(), end.toordinal() + 1)
        if date.fromordinal(ordinal).weekday() < 5
    )
    return weekdays * MARKET_HOURS_PER_DAY


def hourly_sharpe(
    hourly_marks: dict[str, dict[datetime, float]],
    legs: list[Leg],
    starting_cash: float,
    risk_free_annual: float = 0.0,
) -> dict[str, float]:
    """Sharpe over hourly marks, once per-contract quotes are available.

    Args:
        hourly_marks: {contract string: {timestamp: mid price per share}}
        legs: parsed fills, used to reconstruct position size at each hour.
        starting_cash: account equity at the start of the window.
        risk_free_annual: annual risk-free rate as a decimal.

    This is the path to a real hourly number. It is unused by the CLI above
    because the transaction export supplies no marks.
    """
    timestamps = sorted({ts for marks in hourly_marks.values() for ts in marks})
    if len(timestamps) < 2:
        raise ValueError("need at least two hourly marks")

    # Fills carry a date but no time, so a fill is applied at the close of its
    # trade date: strictly earlier dates are in the position, the fill date
    # itself is not. Cash and position therefore move together, and total P&L
    # is exact. What is approximate is *when* within the entry and exit days
    # the P&L landed — the export cannot say, so a same-day round trip shows
    # up as a single step rather than an intraday path.
    equity_curve: list[float] = []
    missing: list[str] = []

    for stamp in timestamps:
        cash = starting_cash
        position: dict[Contract, int] = defaultdict(int)

        for leg in legs:
            if leg.trade_date >= stamp.date():
                continue
            cash += leg.amount
            position[leg.contract] += leg.quantity if leg.is_long else -leg.quantity

        market_value = 0.0
        for contract, qty in position.items():
            if qty == 0:
                continue
            mark = hourly_marks.get(contract.occ_symbol, {}).get(stamp)
            if mark is None:
                missing.append(f"{contract.occ_symbol} @ {stamp}")
                continue
            market_value += qty * mark * 100

        equity_curve.append(cash + market_value)

    if missing:
        raise KeyError(
            f"{len(missing)} missing marks; first few: {missing[:5]}. "
            "Every held contract needs a mark at every timestamp, or the "
            "equity curve jumps where coverage stops."
        )

    if any(value <= 0 for value in equity_curve):
        raise ValueError(
            "equity went non-positive — check starting_cash; it must cover "
            "the margin these spreads required."
        )

    returns = [
        equity_curve[i] / equity_curve[i - 1] - 1 for i in range(1, len(equity_curve))
    ]
    return sharpe(returns, TRADING_HOURS_PER_YEAR, risk_free_annual)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Robinhood transaction export")
    parser.add_argument(
        "--risk-free",
        type=float,
        default=0.042,
        help="annual risk-free rate as a decimal (default 0.042)",
    )
    parser.add_argument(
        "--marks",
        default=None,
        help="JSON of hourly option marks from fetch_hourly_marks.py; "
        "enables the hourly-regime Sharpe",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="account equity to use as the return denominator; "
        "defaults to peak concurrent capital at risk",
    )
    args = parser.parse_args()

    legs, other_rows = load_legs(args.csv_path)
    if not legs:
        raise SystemExit("no option legs found in the export")

    marks = None
    if args.marks:
        import json

        with open(args.marks, encoding="utf-8") as handle:
            raw = json.load(handle)
        marks = {
            symbol: {
                datetime.fromisoformat(stamp): float(price)
                for stamp, price in bars.items()
            }
            for symbol, bars in raw.items()
        }

    # Leg-level FIFO is run purely as a consistency check: it fails loudly if
    # the export closes a contract it never opened.
    match_round_trips(legs)

    spreads, unpaired = build_spreads(legs)
    trades, still_open = match_spread_trades(spreads)

    print(
        format_report(
            legs=legs,
            other_rows=other_rows,
            trades=trades,
            still_open=still_open,
            spreads=spreads,
            capital=args.capital,
            risk_free=args.risk_free,
            marks=marks,
        )
    )

    if unpaired:
        print(f"\nNote: {len(unpaired)} legs did not pair into a vertical.")


if __name__ == "__main__":
    main()
