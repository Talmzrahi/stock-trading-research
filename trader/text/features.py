# ═══════════════════════════════════════════════════════════════════════
#  Reader output -> the numbers the fitted model expects.
#
#  Per release, from its first 20 new or edited sentences:
#
#    <reader>_mean      average P(positive) − P(negative)
#    <reader>_neg/_pos  share of sentences the reader calls negative/positive
#    spread             how much the mood readers disagree, per sentence
#    <reader>_dev       how far this reader sits from the others
#    minilm_map         a sentence-level map from the 384-number
#                       fingerprint to the part of the reaction the
#                       earnings surprise does not explain — the market
#                       labelling each sentence
#    <reader>_biascorr  this reader's score minus what it usually says
#                       about this kind of sentence
#
#  The last three need maps fitted on training data; they live in the
#  model artifact (trader/text/model.py). The disagreement and
#  bias-corrected features are the owner's idea (2026-09-19) and measured
#  their keep: moods alone R² 0.0805, with disagreement 0.0816, with
#  everything 0.0824.
#
#  Training and the daily run both build features here, so a feature
#  cannot mean one thing when fitted and another when traded.
# ═══════════════════════════════════════════════════════════════════════

import numpy as np

from .readers import MOODS, mood_net


def linear_map(emb, m):
    """Apply a stored sentence-level ridge map."""
    if not len(emb):
        return np.zeros(0)
    return (np.asarray(emb, dtype=float) - np.asarray(m["mean"])) @ np.asarray(m["beta"]) + m["intercept"]


def mood_features(outputs):
    """Everything computable from the mood readers alone."""
    out, nets = {}, {}
    for name in MOODS:
        if name not in outputs:
            continue
        logits = outputs[name]
        net = mood_net(logits)
        nets[name] = net
        out[f"{name}_mean"] = float(net.mean()) if len(net) else 0.0
        if len(logits):
            argmax = logits.argmax(1)
            out[f"{name}_neg"] = float((argmax == 1).mean())
            out[f"{name}_pos"] = float((argmax == 0).mean())
        else:
            out[f"{name}_neg"] = out[f"{name}_pos"] = 0.0

    have = list(nets)
    if len(have) >= 2:
        stacked = np.vstack([nets[n] for n in have])
        out["spread"] = float(stacked.std(0).mean()) if stacked.shape[1] else 0.0
        for j, name in enumerate(have):
            others = np.delete(stacked, j, axis=0).mean(0)
            out[f"{name}_dev"] = float((stacked[j] - others).mean()) if stacked.shape[1] else 0.0
    return out, nets


def map_features(outputs, nets, maps):
    """The features that need maps fitted on training data."""
    out = {}
    emb = outputs.get("minilm")
    if emb is None or maps is None:
        return out
    if "minilm_map" in maps:
        p = linear_map(emb, maps["minilm_map"])
        out["minilm_map"] = float(p.mean()) if len(p) else 0.0
    for name, net in nets.items():
        key = f"{name}_biascorr"
        if f"habit_{name}" in maps:
            habit = linear_map(emb, maps[f"habit_{name}"])
            out[key] = float(net.mean() - habit.mean()) if len(net) else 0.0
    return out


def release_features(outputs, maps=None):
    """All text features for one release."""
    out, nets = mood_features(outputs)
    out.update(map_features(outputs, nets, maps))
    return out
