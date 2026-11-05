import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from compliance_fakes import service, FakeReviewer
from banking_chatbot.compliance_labels import template, adjudicate, dataset_digest
from banking_chatbot.compliance_benchmark import compare, release_gate
from banking_chatbot.compliance_evaluation import read_cases, DEFAULT, summarize
from banking_chatbot.compliance_config import build_compliance_service


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.cases=read_cases(DEFAULT)
        self.digest=dataset_digest(DEFAULT)
        self.a=template(self.cases,self.digest,'alice')
        self.b=template(self.cases,self.digest,'bob')
        for doc in (self.a,self.b):
            for row in doc['labels']: row['decision']='allow'

    def test_agreement_and_disagreement(self):
        self.assertTrue(adjudicate(self.cases,self.digest,[self.a,self.b])['complete'])
        self.b['labels'][0]['decision']='block'
        result=adjudicate(self.cases,self.digest,[self.a,self.b])
        self.assertFalse(result['complete'])
        self.assertNotIn(self.cases[0]['id'],result['labels'])
        self.b['labels'][0]['decision']=None
        self.assertFalse(adjudicate(self.cases,self.digest,[self.a,self.b])['complete'])

    def test_duplicate_identity_changed_dataset_and_case_ids_rejected(self):
        for kind in ('reviewer','dataset','ids'):
            b=json.loads(json.dumps(self.b))
            if kind=='reviewer': b['reviewer']='ALICE'
            if kind=='dataset': b['dataset_sha256']='wrong'
            if kind=='ids': b['labels'][0]['id']='unknown'
            with self.assertRaises(ValueError): adjudicate(self.cases,self.digest,[self.a,b])

    def test_paired_repeat_comparison_does_not_inflate_sample_size(self):
        variants=compare(self.cases[:2],{'one':service(),'two':service()},repeats=2,
                         labels_complete=True,split='held_out',minimum_cases=3,max_p95_ms=10000)
        self.assertEqual(len(variants['one']['results']),4)
        self.assertIn('INSUFFICIENT_UNIQUE_CASES',variants['one']['release_gate']['reasons'])
        self.assertFalse(variants['one']['release_gate']['production_approved'])

    def test_empty_uncertain_and_draft_runs_cannot_pass(self):
        for rows in ([],[dict(id='a',expected='allow',decision='unverified',elapsed_ms=1)]):
            result=release_gate(rows,False,'development',120,None)
            self.assertFalse(result['quality_and_latency_pass'])
            self.assertIn('INDEPENDENT_LABELS_INCOMPLETE',result['reasons'])

    def test_repaired_answers_not_scored_using_original_label(self):
        rows=[dict(id='a',expected='block',decision='allow',repair_attempts=1,elapsed_ms=1)]
        result=summarize(rows)
        self.assertIsNone(result['unsafe_release_rate'])
        self.assertEqual(result['repaired_cases_needing_labels'],1)
        self.assertIn('REPAIRED_OUTPUTS_REQUIRE_INDEPENDENT_LABELS',
                      release_gate(rows,True,'held_out',1,10)['reasons'])

    def test_explicit_repair_off_overrides_environment(self):
        with patch.dict(os.environ,{'COMPLIANCE_REPAIR_MODE':'once'}):
            gate=build_compliance_service(reviewer=FakeReviewer(),allow_test=True,repair_mode='off')
            self.assertIsNone(gate.repairer)

    def test_dry_run_requires_no_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'report.json'
            result=subprocess.run([sys.executable,'-m','banking_chatbot.compliance_benchmark',
                '--output',str(output),'--repair','both'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(output.read_text())
            self.assertFalse(report['live'])
            self.assertEqual(report['planned_evaluations'],40)
            self.assertEqual(report['variants'],{})
