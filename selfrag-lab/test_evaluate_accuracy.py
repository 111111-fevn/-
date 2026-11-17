import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluate_accuracy import load_test_cases, score_prediction, summarize_results, write_reports


class AccuracyTests(unittest.TestCase):
    def test_reflection_tokens_do_not_count_as_answer(self):
        result = score_prediction("[Relevant]Paris.[Fully supported][Utility:5]</s>", ["Paris"])
        self.assertTrue(result["exact_match"])
        self.assertTrue(result["answer_inclusion"])

    def test_generated_evidence_cannot_supply_correct_answer(self):
        for prediction in (
            "[Retrieval]<paragraph>Paris is the capital.</paragraph>[Relevant]London.",
            "[Retrieval]<paragraph>Paris is the capital.",
        ):
            with self.subTest(prediction=prediction):
                result = score_prediction(prediction, ["Paris"])
                self.assertFalse(result["exact_match"])
                self.assertFalse(result["answer_inclusion"])

    def test_exact_match_and_inclusion_have_different_meanings(self):
        result = score_prediction("The policy allows returns within 14 days.", ["14"])
        self.assertFalse(result["exact_match"])
        self.assertTrue(result["answer_inclusion"])

    def test_substrings_are_not_english_or_number_answers(self):
        for prediction, reference in (("140", "14"), ("Parisian", "Paris"), ("Cat", "A")):
            with self.subTest(prediction=prediction):
                self.assertFalse(score_prediction(prediction, [reference])["answer_inclusion"])

    def test_multiple_references_and_chinese(self):
        self.assertTrue(score_prediction("FOURTEEN!", ["14", "fourteen"])["exact_match"])
        self.assertTrue(score_prediction("答案是巴黎。", ["巴黎"])["answer_inclusion"])
        self.assertTrue(score_prediction("巴黎。", ["巴黎"])["exact_match"])

    def test_report_counts_regressions_and_round_trips_unicode(self):
        cases = [
            {"id": "1", "question": "首都？", "input": "", "paragraph": "证据", "answers": ["Paris"],
             "before": score_prediction("London", ["Paris"]), "after": score_prediction("Paris", ["Paris"])},
            {"id": "2", "question": "数量？", "input": "", "paragraph": "", "answers": ["14"],
             "before": score_prediction("14", ["14"]), "after": score_prediction("140", ["14"])},
            {"id": "3", "question": "日期？", "input": "", "paragraph": "", "answers": ["Sunday"],
             "before": score_prediction("Monday", ["Sunday"]), "after": score_prediction("Sunday", ["Sunday"])},
        ]
        summary = summarize_results(cases)
        self.assertEqual(summary["exact_match"]["before_correct"], 1)
        self.assertEqual(summary["exact_match"]["after_correct"], 2)
        self.assertEqual(summary["exact_match"]["wrong_to_right"], 2)
        self.assertEqual(summary["exact_match"]["right_to_wrong"], 1)
        self.assertAlmostEqual(summary["exact_match"]["delta_pp"], 100 / 3)
        with tempfile.TemporaryDirectory() as directory:
            write_reports(cases, summary, directory)
            self.assertEqual(json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8")), summary)
            details = [json.loads(line) for line in (Path(directory) / "details.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(details, cases)
            with (Path(directory) / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
                table = list(csv.DictReader(stream))
            self.assertEqual(table[0]["question"], "首都？")
            self.assertEqual(table[1]["after_exact_match"], "False")

    def test_training_output_is_not_accepted_as_gold_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.jsonl"
            path.write_text(json.dumps({"question": "Question", "output": "[Utility:5]"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "answers"):
                load_test_cases(path)

    def test_demo_cases_have_valid_inputs(self):
        cases = load_test_cases(Path(__file__).parent / "examples" / "accuracy_test.jsonl")
        self.assertEqual(len(cases), 10)
        self.assertEqual(cases[0]["answers"], ["14", "fourteen"])


if __name__ == "__main__":
    unittest.main()
