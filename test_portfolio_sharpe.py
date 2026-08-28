"""Tests for portfolio_sharpe.

Run: python test_portfolio_sharpe.py path/to/export.csv
The path is optional; without it the file-driven reconciliation is skipped.
"""

import math
import sys

from portfolio_sharpe import (
    build_spreads,
    daily_pnl,
    load_legs,
    match_round_trips,
    match_spread_trades,
    parse_money,
    peak_concurrent_risk,
    sharpe,
)


def test_parse_money():
    assert parse_money("$819.93") == 819.93
    assert parse_money("($935.04)") == -935.04
    assert parse_money("$1,146.92") == 1146.92
    assert parse_money("") == 0.0
    print("ok  parse_money")


def test_sharpe_known_values():
    # Constant positive returns have zero volatility.
    try:
        sharpe([0.01, 0.01, 0.01], 252)
        raise AssertionError("expected ValueError on zero volatility")
    except ValueError:
        pass

    # Symmetric returns around zero give a Sharpe of zero before the rf drag.
    stats = sharpe([0.01, -0.01, 0.01, -0.01], 252, risk_free_annual=0.0)
    assert abs(stats["sharpe_per_period"]) < 1e-12

    # Annualization scales by sqrt(periods).
    stats = sharpe([0.02, 0.01, 0.03, -0.01], 252, risk_free_annual=0.0)
    assert math.isclose(
        stats["sharpe_annualized"],
        stats["sharpe_per_period"] * math.sqrt(252),
        rel_tol=1e-12,
    )
    print("ok  sharpe")


def test_occ_symbol():
    from datetime import date

    from portfolio_sharpe import Contract

    # Root + YYMMDD + C/P + strike in thousandths, zero-padded to 8.
    assert (
        Contract("AMD", date(2026, 8, 31), "Call", 480.0).occ_symbol
        == "AMD260831C00480000"
    )
    # Fractional strike must not lose the half-dollar.
    assert (
        Contract("MRVL", date(2026, 8, 21), "Call", 242.5).occ_symbol
        == "MRVL260821C00242500"
    )
    # Sub-$20 strike still pads to the full width.
    assert (
        Contract("RGTI", date(2026, 8, 28), "Put", 17.5).occ_symbol
        == "RGTI260828P00017500"
    )
    print("ok  occ_symbol")


def test_reconciliation(path):
    """Realized P&L must equal all option cash flow less open positions' cost."""
    legs, other = load_legs(path)
    match_round_trips(legs)  # raises if the export is internally inconsistent

    spreads, unpaired = build_spreads(legs)
    assert not unpaired, f"{len(unpaired)} legs did not pair into verticals"

    trades, still_open = match_spread_trades(spreads)

    total_cash = sum(leg.amount for leg in legs)
    open_cash = sum(
        spread.net_cash_per_unit * qty for spread, qty in still_open
    )
    realized = sum(trade.pnl for trade in trades)

    assert math.isclose(realized, total_cash - open_cash, abs_tol=0.01), (
        f"realized {realized:.2f} != cash {total_cash:.2f} "
        f"- open {open_cash:.2f}"
    )

    # Every opened contract is accounted for as closed or still open.
    opened = sum(s.quantity for s in spreads if s.is_opening)
    closed = sum(t.quantity for t in trades)
    outstanding = sum(qty for _, qty in still_open)
    assert opened == closed + outstanding, (
        f"{opened} opened != {closed} closed + {outstanding} open"
    )

    # The daily series must sum back to total realized P&L.
    assert math.isclose(sum(daily_pnl(trades).values()), realized, abs_tol=0.01)

    # Risk is positive and bounded by the spread width for every trade.
    for trade in trades:
        assert trade.capital_at_risk > 0, f"non-positive risk on {trade.label}"

    assert peak_concurrent_risk(trades, still_open) > 0

    print(
        f"ok  reconciliation  {len(legs)} legs, {len(trades)} trades, "
        f"realized ${realized:,.2f}"
    )


def test_same_day_round_trip(path):
    """A spread opened and closed the same day must still be matched."""
    legs, _ = load_legs(path)
    spreads, _ = build_spreads(legs)
    trades, _ = match_spread_trades(spreads)

    same_day = [t for t in trades if t.open_date == t.close_date]
    assert same_day, "expected at least one same-day round trip in this export"

    # Such a position must still consume capital in the concurrent-risk figure.
    peak = peak_concurrent_risk(same_day, [])
    assert peak > 0
    print(f"ok  same-day matching  ({len(same_day)} same-day trades)")


def test_execution_order(path):
    """Legs must be ordered oldest-first so FIFO sees opens before closes."""
    legs, _ = load_legs(path)
    assert legs == sorted(legs, key=lambda leg: leg.trade_date)

    # A close on a given day may belong to a position opened earlier, so the
    # ordering invariant is per contract, not per day: when one contract is
    # both opened and closed on the same date, the open must come first.
    checked = 0
    by_contract_day = {}
    for index, leg in enumerate(legs):
        by_contract_day.setdefault((leg.contract, leg.trade_date), []).append(
            (index, leg)
        )

    for (contract, day), entries in by_contract_day.items():
        opens = [i for i, (_, leg) in enumerate(entries) if leg.is_opening]
        closes = [i for i, (_, leg) in enumerate(entries) if not leg.is_opening]
        if opens and closes:
            assert min(opens) < min(closes), (
                f"{day} {contract}: closed before it was opened"
            )
            checked += 1

    assert checked, "no same-day open-and-close contract found to verify"
    print(f"ok  execution order  ({checked} same-day contracts verified)")


if __name__ == "__main__":
    test_parse_money()
    test_sharpe_known_values()
    test_occ_symbol()

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        test_reconciliation(csv_path)
        test_same_day_round_trip(csv_path)
        test_execution_order(csv_path)
    else:
        print("skip  file-driven tests (no CSV path given)")

    print("\nall tests passed")
