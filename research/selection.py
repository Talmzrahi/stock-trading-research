# ═══════════════════════════════════════════════════════════════════════
#  The pre-registered configuration-selection rule (2026-09-15).
#
#  Shared by portfolio_gate.py and stress_test.py so the walk-forward test
#  picks configurations exactly the way the gate did.
#
#  1. Cutoff (no stop). Baseline = the loosest cutoff. A tighter cutoff c
#     qualifies if its CAGR beats the baseline, every cutoff between the
#     baseline and c also beats it, and it is no worse than the baseline in
#     at least 2 of 3 sub-periods. Highest-CAGR qualifier wins, else baseline.
#  2. Stop, at the chosen cutoff. A multiple k qualifies if it beats no-stop,
#     both grid neighbours of k also beat no-stop, and it is no worse in at
#     least 2 of 3 sub-periods. Highest-CAGR qualifier wins, else no stop.
# ═══════════════════════════════════════════════════════════════════════

CUTOFFS = [0.90, 0.95, 0.97, 0.98]
STOPS   = [None, 3, 5, 8, 12]


def no_worse_in_eras(cand, ref, need=2):
    return sum(c >= r for c, r in zip(cand["eras"], ref["eras"])) >= need


def select(S, cutoffs=CUTOFFS, stops=STOPS):
    """S[(cutoff, stop)] = {"cagr": float, "eras": [three sub-period CAGRs]}.
    Returns (cutoff, stop, qualifying_cutoffs, qualifying_stops)."""
    ref = S[(cutoffs[0], None)]
    qual_c = []
    for i, c in enumerate(cutoffs[1:], start=1):
        chain = all(S[(cc, None)]["cagr"] > ref["cagr"] for cc in cutoffs[1:i + 1])
        if chain and no_worse_in_eras(S[(c, None)], ref):
            qual_c.append(c)
    cutoff = max(qual_c, key=lambda c: S[(c, None)]["cagr"]) if qual_c else cutoffs[0]

    ref2 = S[(cutoff, None)]
    ks = stops[1:]
    qual_k = []
    for j, k in enumerate(ks):
        nbrs = [ks[x] for x in (j - 1, j + 1) if 0 <= x < len(ks)]
        if (S[(cutoff, k)]["cagr"] > ref2["cagr"]
                and all(S[(cutoff, n)]["cagr"] > ref2["cagr"] for n in nbrs)
                and no_worse_in_eras(S[(cutoff, k)], ref2)):
            qual_k.append(k)
    stop = max(qual_k, key=lambda k: S[(cutoff, k)]["cagr"]) if qual_k else None
    return cutoff, stop, qual_c, qual_k
