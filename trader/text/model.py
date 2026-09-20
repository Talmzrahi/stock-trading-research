# ═══════════════════════════════════════════════════════════════════════
#  The fitted text model.
#
#  What it predicts: the announcement reaction — the stock's move against
#  SPY from the last close before the release to the first close the
#  system could trade — from the earnings surprise AND the text. Text is
#  therefore only rewarded for what the numbers do not already say.
#
#  What it is used for: the part of that prediction the market has not
#  already delivered. research/v3_signal_probe.py found the gap between
#  prediction and actual move predicts nothing, while the prediction
#  itself showed continuation (+0.95pp top-minus-bottom fifth at 60
#  sessions, p=0.07 on 8,000 development releases). Borderline, which is
#  why this trades in shadow mode only until it passes the gate.
#
#  The artifact is written by research/v3_fit.py and committed:
#    config/text_model.json   feature names, scalars, provenance
#    config/text_model.npz    the coefficients and the sentence-level maps
#
#  Fitting lives in research; this module only loads and predicts, so the
#  daily run never refits anything.
# ═══════════════════════════════════════════════════════════════════════

import json
from pathlib import Path

import numpy as np

ROOT       = Path(__file__).resolve().parent.parent.parent
MODEL_JSON = ROOT / "config" / "text_model.json"
MODEL_NPZ  = ROOT / "config" / "text_model.npz"


class TextModel:
    """Loaded coefficients plus the sentence-level maps the features need."""

    def __init__(self, meta, arrays):
        self.meta = meta
        self.features = list(meta["features"])
        self.sue_winsor = tuple(meta["sue_winsor"])
        self.mu = arrays["mu"]
        self.sd = arrays["sd"]
        self.beta = arrays["beta"]
        self.quantiles = arrays["pred_quantiles"]
        self.maps = {}
        for key in meta["maps"]:
            self.maps[key] = {"beta": arrays[f"{key}_beta"], "mean": arrays[f"{key}_mean"],
                              "intercept": float(arrays[f"{key}_intercept"])}

    # ── loading and saving ────────────────────────────────────────────
    @classmethod
    def load(cls, json_path=MODEL_JSON, npz_path=MODEL_NPZ):
        if not Path(json_path).exists() or not Path(npz_path).exists():
            return None
        meta = json.loads(Path(json_path).read_text(encoding="utf-8"))
        with np.load(npz_path) as z:
            arrays = {k: z[k] for k in z.files}
        return cls(meta, arrays)

    @classmethod
    def save(cls, features, mu, sd, beta, maps, sue_winsor, quantiles, meta,
             json_path=MODEL_JSON, npz_path=MODEL_NPZ):
        arrays = {"mu": np.asarray(mu, dtype=float), "sd": np.asarray(sd, dtype=float),
                  "beta": np.asarray(beta, dtype=float),
                  "pred_quantiles": np.asarray(quantiles, dtype=float)}
        for key, m in maps.items():
            arrays[f"{key}_beta"] = np.asarray(m["beta"], dtype=np.float32)
            arrays[f"{key}_mean"] = np.asarray(m["mean"], dtype=np.float32)
            arrays[f"{key}_intercept"] = np.asarray(float(m["intercept"]))
        Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **arrays)
        payload = {"features": list(features), "maps": sorted(maps), "sue_winsor": list(sue_winsor),
                   **meta}
        Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ── using it ──────────────────────────────────────────────────────
    def winsorise_sue(self, sue):
        lo, hi = self.sue_winsor
        return float(np.clip(sue, lo, hi))

    def row(self, features):
        """Feature dict -> the ordered vector, NaN for anything missing."""
        return np.array([features.get(name, np.nan) for name in self.features], dtype=float)

    def predict(self, rows):
        """Predicted reaction (a return, e.g. 0.012 = +1.2pp) per row.
        NaN where any feature is missing: a half-read release is not scored."""
        X = np.atleast_2d(np.asarray(rows, dtype=float))
        out = np.full(len(X), np.nan)
        ok = np.isfinite(X).all(axis=1)
        if ok.any():
            z = (X[ok] - self.mu) / np.where(self.sd > 0, self.sd, 1.0)
            out[ok] = np.column_stack([z, np.ones(ok.sum())]) @ self.beta
        return out

    def percentile(self, predictions):
        """Where a prediction sits in the training distribution, in [0, 1].

        A live score has no pool of its own to rank against for years, so it
        is ranked against what the model predicted across the training
        releases. That is what makes "the top 5%" mean the same thing on day
        one as it did in the fit."""
        grid = np.linspace(0.0, 1.0, len(self.quantiles))
        p = np.interp(np.asarray(predictions, dtype=float), self.quantiles, grid)
        return np.where(np.isfinite(predictions), p, np.nan)
