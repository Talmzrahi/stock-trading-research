# ═══════════════════════════════════════════════════════════════════════
#  Firm identity for point-in-time membership, through ticker renames.
#
#  universe_history stores tickers. When a member renames (BK → BNY,
#  ABC → COR, BLL → BALL) the old ticker "leaves" and the new one "joins".
#  Price and earnings data live under the NEW ticker, so all of the firm's
#  pre-rename events were being treated as non-member events and dropped
#  — BNY alone lost 16 years.
#
#  Wikipedia's constituent table carries each firm's SEC CIK from 2014-10;
#  earlier revisions carry only the company name. This re-reads every
#  snapshot revision and stores (as_of, symbol, name, cik, firm), where
#  firm = the CIK; for pre-CIK snapshots, the CIK of the same ticker in the
#  first CIK-bearing snapshot, else of the same normalised company name,
#  else "SYM:<ticker>". Resumable; firm assignment reruns every time.
#
#    python research/universe_ids.py
# ═══════════════════════════════════════════════════════════════════════

import re
import sqlite3
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import universe as U  # noqa: E402  — revision_at, SESSION

DB_FILE  = ROOT / "data" / "research.db"
SUFFIXES = (r"\b(the|inc|incorporated|corp|corporation|co|company|companies|ltd|plc|group|"
            r"holding|holdings|nv|sa|lp|llc|class [a-z])\b")


def norm_name(s):
    s = re.sub(r"\[.*?\]", "", str(s).lower()).replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(re.sub(SUFFIXES, " ", s).split())


def init(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS universe_ids (
        as_of TEXT NOT NULL, symbol TEXT NOT NULL, name TEXT, cik TEXT, firm TEXT,
        PRIMARY KEY (as_of, symbol))""")
    conn.commit()


def parse_revision(revid):
    html = U.SESSION.get("https://en.wikipedia.org/w/index.php", params={"oldid": revid}, timeout=30).text
    best = None
    for t in pd.read_html(StringIO(html)):
        cols = {str(c).strip().lower(): c for c in t.columns}
        sym = next((cols[c] for c in cols if "symbol" in c or "ticker" in c), None)
        if sym is not None and 350 <= len(t) <= 600 and (best is None or len(t) > len(best[0])):
            best = (t, cols, sym)
    if best is None:
        return pd.DataFrame(columns=["symbol", "name", "cik"])
    t, cols, sym = best
    name = next((cols[c] for c in cols if "security" in c or "company" in c), None)
    cik = cols.get("cik")
    df = pd.DataFrame({
        "symbol": (t[sym].astype(str).str.strip().str.upper()
                   .str.replace(".", "-", regex=False).str.replace(r"\[.*\]", "", regex=True)),
        "name": t[name].astype(str) if name is not None else "",
        "cik": pd.to_numeric(t[cik], errors="coerce") if cik is not None else float("nan"),
    })
    df = df[df.symbol.str.len().between(1, 6) & df.symbol.str.replace("-", "").str.isalpha()]
    df["cik"] = df.cik.map(lambda x: f"{int(x):010d}" if pd.notna(x) else None)
    return df.drop_duplicates("symbol")


def fetch_snapshots(conn):
    snaps = [r[0] for r in conn.execute("SELECT DISTINCT as_of FROM universe_history ORDER BY as_of")]
    have = {r[0] for r in conn.execute("SELECT DISTINCT as_of FROM universe_ids")}
    for as_of in snaps:
        if as_of in have:
            continue
        try:
            revid, _ = U.revision_at(f"{as_of}T23:59:59Z")
            df = parse_revision(revid)
        except Exception as e:
            print(f"  {as_of}: {type(e).__name__} {str(e)[:80]}")
            continue
        members = {r[0] for r in conn.execute("SELECT symbol FROM universe_history WHERE as_of=?", (as_of,))}
        df = df[df.symbol.isin(members)]
        conn.executemany("INSERT OR REPLACE INTO universe_ids (as_of, symbol, name, cik) VALUES (?,?,?,?)",
                         [(as_of, r.symbol, r.name, r.cik) for r in df.itertuples()])
        conn.commit()
        print(f"  {as_of}: {len(df)} of {len(members)} members matched, "
              f"{df.cik.notna().sum()} with CIK", flush=True)
        time.sleep(2.0)


def assign_firms(conn):
    ids = pd.read_sql_query("SELECT as_of, symbol, name, cik FROM universe_ids", conn)
    known = ids[ids.cik.notna()].sort_values("as_of")
    # Majority CIK per ticker: a few revisions carry a wrong CIK (Hanesbrands
    # under Avery Dennison's), so no single row is trusted on its own.
    by_symbol = known.groupby("symbol").cik.agg(lambda s: s.value_counts().index[0]).to_dict()
    good = known[known.cik == known.symbol.map(by_symbol)]
    by_name = (good.assign(n=good.name.map(norm_name))
               .drop_duplicates("n", keep="first").set_index("n").cik.to_dict())

    def firm(r):
        return by_symbol.get(r.symbol) or by_name.get(norm_name(r.name)) or f"SYM:{r.symbol}"

    ids["firm"] = [firm(r) for r in ids.itertuples()]
    conn.executemany("UPDATE universe_ids SET firm=? WHERE as_of=? AND symbol=?",
                     zip(ids.firm, ids.as_of, ids.symbol))
    conn.commit()

    symbols = ids.groupby("firm").symbol.agg(lambda s: sorted(set(s)))
    multi = symbols[symbols.map(len) > 1]
    print(f"\n{ids.firm.nunique()} firms; {len(multi)} carry more than one ticker "
          "(renames, plus share classes such as GOOG/GOOGL)")
    print("  e.g. " + "; ".join("/".join(v) for v in multi.head(25)))
    missing = (ids.firm.str.startswith("SYM:")).mean()
    print(f"member rows without a CIK identity: {missing:.1%}")


def main():
    conn = sqlite3.connect(DB_FILE, timeout=60)
    init(conn)
    fetch_snapshots(conn)
    assign_firms(conn)
    conn.close()


if __name__ == "__main__":
    main()
