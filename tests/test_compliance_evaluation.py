import unittest
from banking_chatbot.compliance_evaluation import DEFAULT, read_cases, evaluate, summarize
from compliance_fakes import service


class EvaluationTests(unittest.TestCase):
    def test_seed_contracts_and_mock_runner(self):
        cases=read_cases(DEFAULT)
        self.assertEqual(len(cases),24)
        rows=evaluate(cases[:3],service())
        self.assertEqual(len(rows),3)
        self.assertNotIn('answer',rows[0])
        self.assertEqual(summarize(rows)['cases'],3)

    def test_uncertainty_is_not_counted_as_detection(self):
        rows=[dict(expected='block',decision='unverified',elapsed_ms=10),
              dict(expected='allow',decision='unverified',elapsed_ms=20)]
        result=summarize(rows)
        self.assertEqual(result['violation_recall'],0)
        self.assertEqual(result['unverified_rate'],1)
        self.assertEqual(result['compliant_withheld_rate'],1)
        self.assertIsNone(result['block_precision'])
