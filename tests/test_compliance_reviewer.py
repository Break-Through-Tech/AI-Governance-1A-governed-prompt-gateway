import unittest
from unittest.mock import MagicMock, patch
import requests
from compliance_fakes import FakeReviewer, service
from banking_chatbot.compliance_reviewer import GeminiReviewer
from banking_chatbot.compliance_contracts import Candidate, ComplianceContext, Reference


class ReviewerTests(unittest.TestCase):
    def setUp(self):
        self.reviewer=GeminiReviewer('test-key')
        self.candidate=Candidate(text='In this demo, use the app.',finish_reason='STOP')
        self.context=ComplianceContext(request_id='x',question='How?',
            references=(Reference(id='faq',question='How?',answer='Use the app.'),))
        self.verdict=FakeReviewer().review(None).model_dump_json()

    def response(self, body, status=200):
        r=MagicMock(); r.__enter__.return_value=r
        r.status_code=status; r.json.return_value=body
        return r

    @patch('banking_chatbot.compliance_reviewer.requests.post')
    def test_rest_contract_and_usage(self,post):
        post.return_value=self.response({'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':self.verdict}]}}],
            'usageMetadata':{'promptTokenCount':20,'candidatesTokenCount':10,'totalTokenCount':30}})
        gate=service(self.reviewer)
        outcome=gate.evaluate(self.candidate,self.context)
        self.assertEqual(outcome.decision,'allow')
        self.assertEqual(outcome.usage.total_tokens,30)
        args,kw=post.call_args
        self.assertNotIn('test-key',args[0])
        self.assertEqual(kw['headers']['x-goog-api-key'],'test-key')
        self.assertFalse(kw['allow_redirects'])
        self.assertEqual(kw['json']['generationConfig']['responseMimeType'],'application/json')
        self.assertNotIn('test-key',str(kw['json']))

    @patch('banking_chatbot.compliance_reviewer.requests.post')
    def test_provider_failures_never_release_candidate(self,post):
        gate=service(self.reviewer)
        bodies=[{}, {'promptFeedback':{'blockReason':'SAFETY'}},
                {'candidates':[{'finishReason':'MAX_TOKENS'}]},
                {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'not json secret'}]}}]}]
        for body in bodies:
            post.return_value=self.response(body)
            result=gate.evaluate(self.candidate,self.context)
            self.assertEqual(result.decision,'unverified')
            self.assertNotIn('secret',result.model_dump_json())
        for status in (302,400,401,403,404,429,503):
            post.return_value=self.response({'secret':'provider body'},status)
            self.assertEqual(gate.evaluate(self.candidate,self.context).decision,'unverified')
        for error in (requests.Timeout('secret'),requests.ConnectionError('secret')):
            post.side_effect=error
            self.assertEqual(gate.evaluate(self.candidate,self.context).decision,'unverified')

    def test_invalid_model_rejected(self):
        with self.assertRaises(ValueError): GeminiReviewer('test','../../other')
