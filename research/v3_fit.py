# ═══════════════════════════════════════════════════════════════════════
#  Fit the text model and write the artifact the daily run loads.
#
#  Development data only (S&P 500). Everything the live path needs ends up
#  in config/text_model.json + .npz, so the daily run never refits.
#
#  Two kinds of fitting happen here:
#
#    the sentence-level maps  384-number fingerprint -> the part of the
#      reaction the surprise does not explain, and -> what each mood
#      reader usually says about that kind of sentence. Their penalty is
#      chosen by inner cross-validation, because a release's sentences
#      share one label and a fixed penalty overfits.
#    the release-level ridge  surprise + text features -> reaction.
#
#  The ridge is fitted on OUT-OF-FOLD map features, so its coefficients
#  never see maps that were fitted on the same release. The maps shipped
#  for live use are then refitted on everything, which is the usual
#  train-then-deploy step.
#
#    python research/v3_fit.py
# ═══════════════════════════════════════════════════════════════════════

import json
import sqlite3
import subprocess
import sys
import zlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from trader.text.features import mood_features  # noqa: E402
from trader.text.model import TextModel  # noqa: E402
from trader.text.readers import CAP, MOODS  # noqa: E402
from v3_readers_eval import FoldSums, N_FOLDS, out_of_fold, sentence_map  # noqa: E402

V3_DB   = ROOT / "data" / "v3.db"
READERS = ["minilm"] + MOODS
ALPHA   = 10.0          # release-level ridge; the sample is small and the features few


def load(conn):
    """Every release scored by all three readers, with its label."""
    per = {}
    for name in READERS:
        per[name] = {a: np.frombuffer(zlib.decompress(d), dtype=np.float16)
                     .reshape(n, dim).astype(np.float32)[:CAP]
                     for a, n, dim, d in conn.execute(
                         "SELECT accession, n, dim, data FROM reader_out WHERE reader = ?", (name,))}
    common = set.intersection(*(set(v) for v in per.values()))
    lab = pd.read_sql_query("SELECT * FROM labels", conn).drop_duplicates("accession")
    lab = lab[lab.accession.isin(common) & lab.readable.astype(bool)].reset_index(drop=True)
    outputs = [{name: per[name][a] for name in READERS} for a in lab.accession]
    return lab, outputs


def fold_of(dates, n_folds=N_FOLDS):
    """Blocks of whole years, so a fit is never scored on its own years."""
    year = pd.to_datetime(dates).dt.year
    blocks = np.array_split(sorted(year.unique()), n_folds)
    return year.map({y: i for i, blk in enumerate(blocks) for y in blk}).to_numpy()


def fit_final_map(E, y, owner, alpha):
    """The map shipped for live use: one fit on everything."""
    ones = np.zeros(len(E), dtype=int)
    sums = FoldSums(E, y, ones)
    f = sums.fit([0], alpha)
    mean = E.mean(0)
    # recover beta/intercept in the form trader.text.features.linear_map wants
    probe = np.vstack([mean, mean + np.eye(E.shape[1])])
    vals = f(probe)
    return {"beta": vals[1:] - vals[0], "mean": mean, "intercept": float(vals[0])}


def main():
    conn = sqlite3.connect(f"file:{V3_DB}?mode=ro", uri=True, timeout=60)
    lab, outputs = load(conn)
    conn.close()
    if len(lab) < 500:
        raise SystemExit(f"only {len(lab)} releases scored by all readers — score more first")
    print(f"{len(lab):,} development releases scored by {', '.join(READERS)} "
          f"({lab.entry_date.min()} → {lab.entry_date.max()})")

    lo, hi = lab.sue.quantile([0.01, 0.99])
    lab["sue_w"] = lab.sue.clip(lo, hi)
    fold = fold_of(lab.entry_date)
    y = lab.reaction.to_numpy()

    moods = [mood_features(o)[0] for o in outputs]
    feat = pd.DataFrame(moods)
    nets = {m: [np.asarray(f[f"{m}_mean"]) for f in moods] for m in MOODS}

    emb = [o["minilm"] for o in outputs]
    owner = np.repeat(np.arange(len(lab)), [len(e) for e in emb])
    E = np.vstack(emb)

    base = out_of_fold(pd.DataFrame({"fold": fold}), lab[["sue_w", "conviction"]].to_numpy(float),
                       y, ALPHA)
    resid = y - base
    print("Fitting the sentence-level maps …", flush=True)
    feat["minilm_map"], chosen = sentence_map(E, resid[owner], owner, fold, resid)
    maps = {"minilm_map": fit_final_map(E, resid[owner], owner, float(np.median(chosen)))}
    print(f"   minilm map penalty per fold {chosen}, shipped {np.median(chosen):g}")

    from trader.text.readers import mood_net
    for m in MOODS:
        net = np.concatenate([mood_net(o[m]) for o in outputs])
        net_rel = pd.Series(net).groupby(owner).mean().to_numpy()
        habit, ch = sentence_map(E, net, owner, fold, net_rel)
        feat[f"{m}_biascorr"] = net_rel - habit
        maps[f"habit_{m}"] = fit_final_map(E, net, owner, float(np.median(ch)))
        print(f"   {m} habit penalty per fold {ch}, shipped {np.median(ch):g}")

    features = ["sue_w", "conviction"] + sorted(feat.columns)
    X = pd.concat([lab[["sue_w", "conviction"]], feat], axis=1)[features].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    print(f"\n{ok.sum():,} of {len(X):,} releases have every feature")

    oof = out_of_fold(pd.DataFrame({"fold": fold[ok]}), X[ok], y[ok], ALPHA)
    r2 = 1 - ((y[ok] - oof) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    base_oof = out_of_fold(pd.DataFrame({"fold": fold[ok]}),
                           lab.loc[ok, ["sue_w", "conviction"]].to_numpy(float), y[ok], ALPHA)
    r2b = 1 - ((y[ok] - base_oof) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    print(f"out-of-fold R² on the reaction: surprise only {r2b:.4f}, with text {r2:.4f}")

    mu, sd = X[ok].mean(0), X[ok].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([(X[ok] - mu) / sd, np.ones(int(ok.sum()))])
    A = Z.T @ Z + ALPHA * np.eye(Z.shape[1])
    A[-1, -1] -= ALPHA
    beta = np.linalg.solve(A, Z.T @ y[ok])
    print("\ncoefficients (pp of reaction per standard deviation):")
    for name, b in sorted(zip(features, beta[:-1]), key=lambda kv: -abs(kv[1])):
        print(f"   {name:<28}{b * 100:+.3f}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip()
    quantiles = np.quantile(oof, np.linspace(0.0, 1.0, 101))
    print(f"\npredicted reaction across training: median {np.median(oof) * 100:+.2f}pp, "
          f"top 5% above {quantiles[95] * 100:+.2f}pp, top 1% above {quantiles[99] * 100:+.2f}pp")
    TextModel.save(features, mu, sd, beta, maps, (float(lo), float(hi)), quantiles, {
        "fitted": str(date.today()), "commit": commit, "readers": READERS, "cap": CAP,
        "n_releases": int(ok.sum()), "train_start": str(lab.entry_date.min()),
        "train_end": str(lab.entry_date.max()), "alpha": ALPHA,
        "oof_r2_surprise_only": round(float(r2b), 5), "oof_r2_with_text": round(float(r2), 5),
        "target": "announcement reaction: stock minus SPY, last close before the release to "
                  "the first tradable close",
        "note": "shadow mode only until it passes the gate in CLAUDE.md",
    })
    print("\nwrote config/text_model.json and config/text_model.npz")


if __name__ == "__main__":
    main()
