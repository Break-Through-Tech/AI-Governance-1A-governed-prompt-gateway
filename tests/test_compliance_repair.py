import unittest
from unittest.mock import patch
from compliance_fakes import FakeReviewer
from banking_chatbot.compliance_config import build_compliance_service
from banking_chatbot.compliance_contracts import Candidate, ComplianceContext, Reference, ComponentDescriptor, RepairResult, Usage
from banking_chatbot.compliance_repairer import REPAIRABLE


class Repairer:
    descriptor=ComponentDescriptor(component_id='repairer',version='test',rule_ids=REPAIRABLE,test_only=True)
    def __init__(self,text='In this demo, use the app.'):
        self.calls=[]; self.text=text
    def repair(self,request):
        self.calls.append(request)
        return RepairResult(text=self.text,finish_reason='STOP',usage=Usage(input_tokens=3,output_tokens=2,total_tokens=5))


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.context=ComplianceContext(request_id='r',question='How?',references=(Reference(id='a',question='How?',answer='Use the app.'),))
        self.long=Candidate(text='word '*151,finish_reason='STOP')
    def gate(self,repairer,reviewer=None):
        return build_compliance_service(reviewer=reviewer or FakeReviewer(),repairer=repairer,allow_test=True)
    def test_repair_rechecked_and_usage_counted(self):
        repairer=Repairer(); reviewer=FakeReviewer()
        result=self.gate(repairer,reviewer).evaluate(self.long,self.context)
        self.assertEqual(result.decision,'allow')
        self.assertEqual(result.repair_attempts,1)
        self.assertEqual(result.usage.total_tokens,20)
        self.assertEqual(len(reviewer.calls),1)
        self.assertNotIn('word word',repairer.calls[0].model_dump_json())
    def test_no_recursive_repair(self):
        r=Repairer('word '*151)
        result=self.gate(r).evaluate(self.long,self.context)
        self.assertEqual(result.decision,'block'); self.assertEqual(len(r.calls),1)
    def test_repaired_sensitive_text_never_released(self):
        r=Repairer('SSN 123-45-6789')
        result=self.gate(r).evaluate(self.long,self.context)
        self.assertEqual(result.decision,'block')
        self.assertNotIn('123-45-6789',result.model_dump_json())
    def test_hard_failure_and_uncertain_review_skip_repair(self):
        for text,reviewer in [('SSN 123-45-6789',FakeReviewer()),('Short answer',FakeReviewer('uncertain'))]:
            r=Repairer(); result=self.gate(r,reviewer).evaluate(Candidate(text=text,finish_reason='STOP'),self.context)
            self.assertNotEqual(result.decision,'allow'); self.assertEqual(r.calls,[])
    def test_repair_exception_fails_closed(self):
        r=Repairer()
        with patch.object(r,'repair',side_effect=RuntimeError('secret')):
            result=self.gate(r).evaluate(self.long,self.context)
        self.assertEqual(result.decision,'unverified'); self.assertNotIn('secret',result.model_dump_json())
    def test_repair_configuration_changes_cache_identity(self):
        a=build_compliance_service(reviewer=FakeReviewer(),allow_test=True)
        b=self.gate(Repairer())
        self.assertNotEqual(a.fingerprint,b.fingerprint)
