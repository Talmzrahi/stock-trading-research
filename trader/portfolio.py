# ═══════════════════════════════════════════════════════════════════════
#  Layer 5 — portfolio construction.
#
#  Fixed fraction at entry: each new position gets equity / slots, and is
#  never rebalanced. `slots` is the expected number of concurrent holdings,
#  estimated from how often the signal fired over the trailing year — so
#  the fraction adapts to a tighter cutoff without peeking ahead. Idle
#  cash sits in the benchmark.
# ═══════════════════════════════════════════════════════════════════════


def slot_count(recent_signals, cfg):
    expected = recent_signals * cfg.hold_days / 252 * cfg.slot_mult
    return max(cfg.min_slots, int(round(expected)))


def size_entries(equity, n_entries, slots, spendable, cfg):
    """Dollar amount per new position. Scaled down pro rata when today's
    entries need more than cash + benchmark can fund; never financed."""
    if n_entries == 0:
        return 0.0
    per = equity / slots
    if per * n_entries > spendable:
        per = max(spendable, 0.0) / n_entries
    return per if per >= cfg.min_order else 0.0


def benchmark_trade(cash_after, equity, bench_value, cfg):
    """Keep idle cash in the benchmark. A band of one to two buffers stops
    small fill drift from trading every day. + buy, - sell (dollars)."""
    lo, hi = cfg.cash_buffer * equity, 2 * cfg.cash_buffer * equity
    if cash_after < lo:
        return -min(lo - cash_after, bench_value)
    if cash_after > hi:
        return cash_after - lo
    return 0.0
