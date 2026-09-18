import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
from release_text import blocks, compact_html, exhibit  # noqa: E402


def parse(html):
    return blocks(compact_html(html))


class ExhibitTest(unittest.TestCase):
    def test_takes_the_first_ex99_and_only_its_text(self):
        sub = ("<DOCUMENT>\n<TYPE>8-K\n<TEXT>cover</TEXT>\n</DOCUMENT>\n"
               "<DOCUMENT>\n<TYPE>EX-99.1\n<SEQUENCE>2\n<FILENAME>pr.htm\n"
               "<DESCRIPTION>EX-99.1\n<TEXT>\n<p>Results</p>\n</TEXT>\n</DOCUMENT>")
        kind, name, raw = exhibit(sub)
        self.assertEqual((kind, name), ("EX-99.1", "pr.htm"))
        self.assertEqual(raw.strip(), "<p>Results</p>")        # no SEQUENCE/FILENAME leak

    def test_none_without_an_ex99(self):
        self.assertIsNone(exhibit("<DOCUMENT>\n<TYPE>8-K\n<TEXT>x</TEXT>\n</DOCUMENT>"))


class BlocksTest(unittest.TestCase):
    def test_hard_wrapped_source_stays_one_paragraph(self):
        b = parse("<html><body><p>Revenue rose to $2.31 billion,\nup 21.9% from the\n"
                  "year-ago quarter.</p><p>Second.</p></body></html>")
        self.assertEqual([x["t"] for x in b],
                         ["Revenue rose to $2.31 billion, up 21.9% from the year-ago quarter.",
                          "Second."])

    def test_windows_1252_quotes_and_dashes_are_restored(self):
        b = parse("<html><body><p>The Company&#146;s results &#151; &#147;strong&#148;</p>"
                  "</body></html>")
        self.assertEqual(b[0]["t"], "The Company’s results — “strong”")

    def test_data_table_becomes_rows_with_values_rejoined(self):
        b = parse("<html><body><table>"
                  "<tr><td>Revenue</td><td>$</td><td>6,582</td><td></td><td>$</td><td>5,791</td></tr>"
                  "<tr><td>Other, net</td><td>(8,857</td><td>)</td><td>12</td><td>%</td></tr>"
                  "</table></body></html>")
        self.assertEqual(b, [{"k": "row", "c": ["Revenue", "$6,582", "$5,791"]},
                             {"k": "row", "c": ["Other, net", "(8,857)", "12%"]}])

    def test_layout_table_reads_as_paragraphs(self):
        b = parse("<html><body><table><tr><td>•</td><td><p>Record revenue.</p>"
                  "<p>Raised guidance.</p></td></tr></table></body></html>")
        self.assertEqual([(x["k"], x.get("t")) for x in b],
                         [("p", "Record revenue."), ("p", "Raised guidance.")])

    def test_data_table_nested_in_layout_table(self):
        b = parse("<html><body><table><tr><td><p>Highlights</p>"
                  "<table><tr><td>EPS</td><td>1.05</td><td>0.91</td></tr></table>"
                  "</td></tr></table></body></html>")
        self.assertEqual(b, [{"k": "p", "t": "Highlights"},
                             {"k": "row", "c": ["EPS", "1.05", "0.91"]}])

    def test_long_single_cell_row_is_a_footnote_paragraph(self):
        note = "(1) Non-GAAP figures exclude stock-based compensation, " * 4
        b = parse(f"<html><body><table><tr><td>EPS</td><td>1.05</td></tr>"
                  f"<tr><td>{note}</td></tr></table></body></html>")
        self.assertEqual(b[1], {"k": "p", "t": note.strip()})

    def test_styling_is_stripped_structure_kept(self):
        html = compact_html('<html><body><div style="x"><font size="2">A</font>'
                            '<b>Bold</b><sup>1</sup></div></body></html>')
        self.assertNotIn("style", html)
        self.assertNotIn("font", html)
        self.assertIn("<b>Bold</b>", html)
        self.assertIn("<sup>1</sup>", html)

    def test_plain_text_exhibit(self):
        b = parse("COMPANY REPORTS RESULTS\n\nRevenue grew in the quarter\nacross segments.\n\n"
                  "Revenue      1,200     1,100\nCost           700       650\n")
        self.assertEqual(b, [{"k": "p", "t": "COMPANY REPORTS RESULTS"},
                             {"k": "p", "t": "Revenue grew in the quarter across segments."},
                             {"k": "row", "c": ["Revenue", "1,200", "1,100"]},
                             {"k": "row", "c": ["Cost", "700", "650"]}])


if __name__ == "__main__":
    unittest.main()
