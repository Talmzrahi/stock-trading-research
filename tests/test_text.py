import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trader.config import Config
from trader.market_calendar import trading_days
from trader.shadow import run_shadow, shadow_events
from trader.text import store
from trader.text.features import mood_features, release_features
from trader.text.model import TextModel
from trader.text.readers import mood_net

N = 120
CFG = Config(hold_days=10, min_slots=5, initial_capital=1000.0, cutoff=0.95)


def logits(*triples):
    """Rows of [positive, negative, neutral] logits."""
    return np.array(triples, dtype=float)


class MoodFeatureTest(unittest.TestCase):
    def test_net_is_positive_minus_negative(self):
        net = mood_net(logits((2.0, 0.0, 0.0), (0.0, 2.0, 0.0)))
        self.assertGreater(net[0], 0.5)
        self.assertLess(net[1], -0.5)

    def test_shares_and_mean(self):
        out, nets = mood_features({"distilroberta_fin": logits((3.0, 0, 0), (0, 3.0, 0),
                                                               (0, 0, 3.0), (3.0, 0, 0))})
        self.assertAlmostEqual(out["distilroberta_fin_pos"], 0.5)
        self.assertAlmostEqual(out["distilroberta_fin_neg"], 0.25)
        self.assertAlmostEqual(out["distilroberta_fin_mean"], nets["distilroberta_fin"].mean())

    def test_disagreement_between_two_readers(self):
        agree = mood_features({"distilroberta_fin": logits((3.0, 0, 0), (3.0, 0, 0)),
                               "finbert": logits((3.0, 0, 0), (3.0, 0, 0))})[0]
        differ = mood_features({"distilroberta_fin": logits((3.0, 0, 0), (3.0, 0, 0)),
                                "finbert": logits((0, 3.0, 0), (0, 3.0, 0))})[0]
        self.assertAlmostEqual(agree["spread"], 0.0, places=6)
        self.assertGreater(differ["spread"], 0.5)
        # the reader that is more positive than the other has a positive deviation
        self.assertGreater(differ["distilroberta_fin_dev"], 0)
        self.assertLess(differ["finbert_dev"], 0)

    def test_empty_release_gives_zeros_not_errors(self):
        out, _ = mood_features({"finbert": np.zeros((0, 3))})
        self.assertEqual(out["finbert_mean"], 0.0)


class ModelArtifactTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.json = Path(self.dir.name) / "m.json"
        self.npz = Path(self.dir.name) / "m.npz"
        features = ["conviction", "finbert_mean", "minilm_map"]
        beta = np.array([0.01, 0.02, 0.0, 0.005])        # last entry is the intercept
        maps = {"minilm_map": {"beta": np.array([1.0, 0.0]), "mean": np.array([0.0, 0.0]),
                               "intercept": 0.5}}
        TextModel.save(features, np.zeros(3), np.ones(3), beta, maps, (-0.1, 0.1),
                       np.linspace(-0.05, 0.05, 101), {"fitted": "2026-09-20"},
                       json_path=self.json, npz_path=self.npz)
        self.model = TextModel.load(self.json, self.npz)

    def tearDown(self):
        self.dir.cleanup()

    def test_round_trip_and_prediction(self):
        row = self.model.row({"conviction": 1.0, "finbert_mean": 2.0, "minilm_map": 3.0})
        self.assertEqual(self.model.predict(row)[0], 0.01 + 0.04 + 0.005)

    def test_missing_feature_is_not_scored(self):
        row = self.model.row({"conviction": 1.0})       # the others are absent
        self.assertTrue(np.isnan(self.model.predict(row)[0]))

    def test_percentile_is_monotone_and_bounded(self):
        p = self.model.percentile([-1.0, -0.05, 0.0, 0.05, 1.0])
        self.assertTrue(np.all(np.diff(p) >= 0))
        self.assertAlmostEqual(p[0], 0.0)
        self.assertAlmostEqual(p[-1], 1.0)
        self.assertAlmostEqual(p[2], 0.5, places=6)

    def test_sue_winsorised_to_training_range(self):
        self.assertEqual(self.model.winsorise_sue(5.0), 0.1)
        self.assertEqual(self.model.winsorise_sue(-5.0), -0.1)

    def test_map_features_use_the_stored_map(self):
        out = release_features({"minilm": np.array([[2.0, 9.0], [4.0, 9.0]]),
                                "finbert": logits((1.0, 0, 0), (1.0, 0, 0))}, self.model.maps)
        self.assertAlmostEqual(out["minilm_map"], 3.5)       # mean of (2, 4) + intercept 0.5


class ScoreStoreTest(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        store.init_scores(self.conn)

    def row(self, key, prediction, percentile, status="scored"):
        return {"event_key": key, "scored_on": "2026-09-21", "symbol": "AAA", "cik": "1",
                "accession": "x", "ann_date": "2026-09-20", "entry_date": "2026-09-21",
                "status": status, "n_sentences": 20, "prediction": prediction,
                "percentile": percentile, "features": {"finbert_mean": 0.1}}

    def test_record_and_read_back(self):
        store.record(self.conn, self.row("AAA|2026-09-20", 0.02, 0.97))
        self.assertEqual(store.percentiles(self.conn), {"AAA|2026-09-20": 0.97})
        self.assertEqual(store.scored_keys(self.conn), {"AAA|2026-09-20"})

    def test_first_write_wins(self):
        store.record(self.conn, self.row("AAA|2026-09-20", 0.02, 0.97))
        store.record(self.conn, self.row("AAA|2026-09-20", -0.09, 0.01))
        self.assertEqual(store.percentiles(self.conn), {"AAA|2026-09-20": 0.97})

    def test_unscored_releases_are_recorded_too(self):
        store.record(self.conn, {**self.row("BBB|2026-09-20", None, None, "no_filing"),
                                 "n_sentences": None, "features": None})
        self.assertEqual(store.percentiles(self.conn), {})
        self.assertEqual(store.scored_keys(self.conn), {"BBB|2026-09-20"})


class ShadowAccountTest(unittest.TestCase):
    def setUp(self):
        self.cal = trading_days("2020-01-01", "2021-12-31")[:N]
        self.closes = pd.DataFrame({"SPY": np.full(N, 100.0), "AAA": np.full(N, 50.0)},
                                   index=self.cal)
        self.events = pd.DataFrame([
            dict(symbol="AAA", key="AAA|1", entry_idx=20, entry_date=self.cal[20], pit=True),
        ])
        self.dir = tempfile.TemporaryDirectory()
        self.db = Path(self.dir.name) / "shadow.db"

    def tearDown(self):
        self.dir.cleanup()

    def test_scores_become_conviction(self):
        ev = shadow_events(self.events, {"AAA|1": 0.99})
        self.assertAlmostEqual(ev.conviction.iloc[0], 0.99)
        self.assertEqual(ev.n_signals.iloc[0], 1)

    def test_unscored_event_has_no_conviction(self):
        ev = shadow_events(self.events, {})
        self.assertTrue(np.isnan(ev.conviction.iloc[0]))

    def run_days(self, scores, days):
        out = []
        for i in days:
            out.append(run_shadow(CFG, self.cal, self.closes, self.events, scores,
                                  self.cal[i], db_path=self.db, log=lambda *_: None))
        return out

    def test_a_high_score_is_traded_on_its_own_books(self):
        last = self.run_days({"AAA|1": 0.99}, [20, 21, 22])[-1]
        self.assertEqual(last["positions"], 1)
        self.assertLess(last["cash"], CFG.initial_capital)
        self.assertTrue(self.db.exists())

    def test_a_low_score_is_left_alone(self):
        last = self.run_days({"AAA|1": 0.10}, [20, 21, 22])[-1]
        self.assertEqual(last["positions"], 0)

    def test_it_does_not_settle_into_the_future(self):
        # the live run's calendar extends past the prices, so exits can be
        # scheduled; the account must stop at the last known close
        closes = self.closes.copy()
        closes.iloc[30:] = np.nan
        run_shadow(CFG, self.cal, closes, self.events, {"AAA|1": 0.99}, self.cal[25],
                   db_path=self.db, log=lambda *_: None)
        conn = sqlite3.connect(self.db)
        try:
            last = conn.execute("SELECT value FROM account WHERE key='last_settled'").fetchone()[0]
        finally:
            conn.close()          # Windows will not delete the temp dir while it is open
        self.assertLessEqual(pd.Timestamp(last), self.cal[29])

    def test_the_account_persists_between_runs(self):
        first = self.run_days({"AAA|1": 0.99}, [20])[0]
        again = self.run_days({"AAA|1": 0.99}, [21, 22])[-1]
        self.assertEqual(first["inception"], again["inception"])
        self.assertEqual(again["positions"], 1)          # not re-entered, just carried


if __name__ == "__main__":
    unittest.main()
