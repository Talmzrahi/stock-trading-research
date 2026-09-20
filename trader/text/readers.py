# ═══════════════════════════════════════════════════════════════════════
#  The models that read sentences.
#
#  Three free models, loaded once and kept: two financial-mood readers
#  whose disagreement carries information, and one small model whose
#  384-number fingerprint says what a sentence is about. Chosen by
#  measurement, not by reputation — research/design_sentiment_v3.md
#  ("Reader comparison"): DistilRoBERTa-finance beat FinBERT (R² 0.0805
#  vs 0.0793) at half the depth, averaging the two bought nothing, and
#  their disagreement bought more than either alone.
#
#  Mood logits come out RAW, in the fixed order [positive, negative,
#  neutral]: no temperature, no bias, no averaging. v1's adjustments were
#  measured to be cosmetic (rank correlation 0.993-0.999 with untuned
#  output) and averaging destroys the disagreement.
# ═══════════════════════════════════════════════════════════════════════

import numpy as np

MODELS = {
    "minilm":            ("sentence-transformers/all-MiniLM-L6-v2", "embed"),
    "distilroberta_fin": ("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", "mood"),
    "finbert":           ("ProsusAI/finbert", "mood"),
}
MOODS   = [n for n, (_, kind) in MODELS.items() if kind == "mood"]
CAP     = 20      # sentences read per release: the news is at the top
MAX_TOK = 128     # changed sentences: median 34 tokens, 90th percentile 63
BATCH   = 64


def label_order(model):
    """Indices of [positive, negative, neutral] in this model's output."""
    lab = {i: v.lower() for i, v in model.config.id2label.items()}
    pick = lambda word: next(i for i, v in lab.items() if word in v)
    return [pick("pos"), pick("neg"), pick("neu")]


class Readers:
    """The models, loaded on first use. Loading costs ~20s and ~1GB, so a
    daily run builds this once and scores every release through it."""

    def __init__(self, names=None, threads=None):
        self.names = list(names or MODELS)
        self.threads = threads
        self._runners = None

    def _load(self):
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        if self.threads:
            torch.set_num_threads(self.threads)
        self._runners = {}
        for name in self.names:
            repo, kind = MODELS[name]
            tok = AutoTokenizer.from_pretrained(repo)
            if kind == "embed":
                model = AutoModel.from_pretrained(repo).eval()

                def run(texts, tok=tok, model=model):
                    enc = tok(texts, return_tensors="pt", truncation=True,
                              max_length=MAX_TOK, padding=True)
                    with torch.no_grad():
                        hidden = model(**enc).last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                    return torch.nn.functional.normalize(emb, dim=1).numpy()
            else:
                model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
                order = label_order(model)

                def run(texts, tok=tok, model=model, order=order):
                    enc = tok(texts, return_tensors="pt", truncation=True,
                              max_length=MAX_TOK, padding=True)
                    with torch.no_grad():
                        return model(**enc).logits[:, order].numpy()
            self._runners[name] = run

    def score(self, sentences):
        """{reader: array (sentences x dims)} for one release's sentences."""
        if not sentences:
            return {name: np.zeros((0, 384 if name == "minilm" else 3)) for name in self.names}
        if self._runners is None:
            self._load()
        order = np.argsort([len(s) for s in sentences])      # similar lengths pad less
        out = {}
        for name, run in self._runners.items():
            rows = [None] * len(sentences)
            for i in range(0, len(order), BATCH):
                idx = order[i:i + BATCH]
                for j, r in zip(idx, run([sentences[j] for j in idx])):
                    rows[j] = r
            out[name] = np.stack(rows)
        return out


def mood_net(logits):
    """P(positive) − P(negative) per sentence, from raw logits."""
    logits = np.asarray(logits, dtype=float)
    if not len(logits):
        return np.zeros(0)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return p[:, 0] - p[:, 1]
