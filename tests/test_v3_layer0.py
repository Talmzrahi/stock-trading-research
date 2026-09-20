import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trader.text.novelty import (classify_company, max_similarity, sentences,  # noqa: E402
                                 template_key, word_pairs)


def release(*paragraphs, rows=()):
    return [{"k": "p", "t": p} for p in paragraphs] + [{"k": "row", "c": list(r)} for r in rows]


class SentencesTest(unittest.TestCase):
    def test_splits_at_sentence_ends(self):
        self.assertEqual(sentences("Revenue rose 5%. Margins fell. EPS was $1.05."),
                         ["Revenue rose 5%.", "Margins fell.", "EPS was $1.05."])

    def test_does_not_split_after_abbreviations_or_initials(self):
        text = "Acme Inc. reported results in the U.S. Market today. John Q. Smith said so."
        self.assertEqual(sentences(text), ["Acme Inc. reported results in the U.S. Market today.",
                                           "John Q. Smith said so."])

    def test_decimals_are_not_boundaries(self):
        self.assertEqual(sentences("Revenue was $2.31 billion. Up 6.7%."),
                         ["Revenue was $2.31 billion.", "Up 6.7%."])


class TemplateKeyTest(unittest.TestCase):
    def test_numbers_dates_and_quarters_masked(self):
        a = template_key("Third quarter revenue was $6.76 billion, up 13%, as of Aug. 28, 2026.")
        b = template_key("Second quarter revenue was $6,620 billion, up (2)%, as of May 29, 2026.")
        self.assertEqual(a, b)

    def test_changed_words_still_differ(self):
        self.assertNotEqual(template_key("Revenue grew 5%."), template_key("Revenue fell 5%."))


class SimilarityTest(unittest.TestCase):
    def test_identical_and_disjoint(self):
        a = word_pairs(template_key("we raised our full year outlook"))
        b = word_pairs(template_key("the board declared a dividend"))
        sim = max_similarity([a, b], [a])
        self.assertAlmostEqual(sim[0], 1.0)
        self.assertAlmostEqual(sim[1], 0.0)


class ClassifyTest(unittest.TestCase):
    def setUp(self):
        boiler = "Acme makes widgets for industrial customers worldwide."
        q1 = release(boiler, "Revenue was $100 million, up 5% in the first quarter.",
                     "We remain confident in our strategy and our people.",
                     rows=[("Revenue", "$100", "$95")])
        q2 = release(boiler, "Revenue was $120 million, up 20% in the second quarter.",
                     "We remain very confident in our long term strategy and our people.",
                     "We are withdrawing guidance because a major customer cancelled orders.",
                     rows=[("Revenue", "$120", "$100")])
        self.recs = classify_company([("q2", "2020-07-30", q2), ("q1", "2020-04-30", q1)])

    def test_first_release_has_no_history(self):
        first = self.recs[0]
        self.assertEqual((first["accession"], first["n_prior"]), ("q1", 0))
        self.assertEqual(first["n_new"], first["n_units"])

    def test_classes_against_history(self):
        got = {u[5]: u[2] for u in self.recs[1]["units"]}
        self.assertEqual(got["Acme makes widgets for industrial customers worldwide."], "boilerplate")
        self.assertEqual(got["Revenue was $120 million, up 20% in the second quarter."], "template")
        self.assertEqual(got["Revenue | $120 | $100"], "template")
        self.assertEqual(got["We remain very confident in our long term strategy and our people."],
                         "edited")
        self.assertEqual(got["We are withdrawing guidance because a major customer cancelled orders."],
                         "new")

    def test_history_is_only_the_last_k(self):
        a = release("Alpha sentence about widgets and gadgets.")
        b = release("Beta sentence about something else entirely.")
        recs = classify_company([("1", "2020-01-01", a), ("2", "2020-04-01", b),
                                 ("3", "2020-07-01", a)], prior_k=1)
        self.assertEqual(recs[2]["n_new"], 1)            # "a" fell out of a 1-release window


if __name__ == "__main__":
    unittest.main()
