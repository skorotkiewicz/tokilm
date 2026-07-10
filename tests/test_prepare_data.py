import unittest

from tokilm.prepare_data import format_sample, translation_samples

class TranslationSamplesTest(unittest.TestCase):
    def test_formats_both_translation_directions(self):
        samples = translation_samples([{
            "source": "  I   am happy. ",
            "source_lang": "English",
            "tok": "mi pilin pona.",
        }])

        self.assertEqual(samples, [
            format_sample("English: I am happy.", "Toki Pona: mi pilin pona."),
            format_sample("Toki Pona: mi pilin pona.", "English: I am happy."),
        ])

    def test_skips_incomplete_rows_and_caps_valid_pairs(self):
        rows = [
            {"source": "", "source_lang": "English", "tok": "weka"},
            {"source": "Hello!", "source_lang": "English", "tok": "toki!"},
            {"source": "Bye!", "source_lang": "English", "tok": "mi tawa!"},
        ]

        samples = translation_samples(rows, n_samples=1)

        self.assertEqual(len(samples), 2)
        self.assertIn("Hello!", samples[0])
        self.assertNotIn("Bye!", "".join(samples))

if __name__ == "__main__":
    unittest.main()
