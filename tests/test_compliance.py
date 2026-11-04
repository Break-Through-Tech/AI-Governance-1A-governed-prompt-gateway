import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from compliance_fakes import FakeReviewer, service
from banking_chatbot.compliance_config import build_compliance_service, load_policy
from banking_chatbot.compliance_contracts import Candidate, ComplianceContext, Reference, Claim
from banking_chatbot.compliance_validators import FormatValidator, SensitiveDataValidator, ContactPolicyValidator
from banking_chatbot.chat import chat
from banking_chatbot.cache import SQLiteResponseCache
from banking_chatbot.retrieval import Retriever


class ComplianceTests(unittest.TestCase):
    def setUp(self):
        self.context = ComplianceContext(request_id='test', question='How do I change my PIN?',
            references=(Reference(id='faq', question='Change PIN?', answer='Use the app.'),))
        self.candidate = Candidate(text='In this demo, use the app.', finish_reason='STOP')

    def test_allow_and_block_and_uncertainty_all_origins(self):
        for origin in ('fresh', 'exact_cache', 'semantic_cache'):
            for outcome, expected in [('pass','allow'), ('fail','block'), ('uncertain','unverified')]:
                result = service(FakeReviewer(outcome)).evaluate(self.candidate.model_copy(update={'origin':origin}), self.context)
                self.assertEqual(result.decision, expected)
                if expected != 'allow':
                    self.assertNotIn(self.candidate.text, result.model_dump_json())

    def test_local_failure_prevents_external_call(self):
        for text, finish in [('SSN: 123-45-6789','STOP'), ('Call 800-555-1212','STOP'),
                             ('','STOP'), ('Partial','MAX_TOKENS'), ('word ' * 151,'STOP'),
                             ('Card 4111 1111 1111 1111','STOP')]:
            reviewer = FakeReviewer()
            result = service(reviewer).evaluate(Candidate(text=text,finish_reason=finish),self.context)
            self.assertEqual(result.decision,'block')
            self.assertEqual(reviewer.calls,[])

    def test_sensitive_context_and_missing_evidence_never_sent(self):
        for context in [self.context.model_copy(update={'question':'secret=private-value'}),
                        self.context.model_copy(update={'references':()}),
                        self.context.model_copy(update={'question':'a'*24001})]:
            reviewer=FakeReviewer()
            self.assertEqual(service(reviewer).evaluate(self.candidate,context).decision,'unverified')
            self.assertEqual(reviewer.calls,[])

    def test_exception_and_invalid_coverage_fail_closed(self):
        for raw in [RuntimeError('secret raw provider body'), None]:
            reviewer=FakeReviewer()
            with patch.object(reviewer,'review',side_effect=raw if isinstance(raw,Exception) else None, return_value=raw):
                result=service(reviewer).evaluate(self.candidate,self.context)
                self.assertEqual(result.decision,'unverified')
                self.assertNotIn('provider body',result.model_dump_json())
        reviewer=FakeReviewer()
        valid=reviewer.review(None)
        for invalid in [valid.model_copy(update={'findings':valid.findings[:-1]}),
                        valid.model_copy(update={'claims':(Claim(text='x',assessment='supported',source_ids=('invented',)),),'no_factual_claims':False}),
                        valid.model_copy(update={'claims':(Claim(text='x',assessment='contradicted',source_ids=('faq',)),),'no_factual_claims':False})]:
            with patch.object(reviewer,'review',return_value=invalid):
                self.assertEqual(service(reviewer).evaluate(self.candidate,self.context).decision,'unverified')

    def test_configuration_validation_and_substitution(self):
        with self.assertRaises(ValueError):
            build_compliance_service(reviewer=FakeReviewer())
        with self.assertRaises(ValueError):
            build_compliance_service(reviewer=FakeReviewer(),validators=(FormatValidator(),),allow_test=True)
        with self.assertRaises(ValueError):
            build_compliance_service(reviewer=FakeReviewer(),validators=(FormatValidator(),FormatValidator()),allow_test=True)
        original=service()
        class Replacement(FormatValidator):
            descriptor=FormatValidator.descriptor.model_copy(update={'version':'2'})
        replacement=build_compliance_service(reviewer=FakeReviewer(),allow_test=True,
            validators=(Replacement(),SensitiveDataValidator(),ContactPolicyValidator()))
        self.assertNotEqual(original.fingerprint,replacement.fingerprint)
        self.assertEqual(replacement.evaluate(self.candidate,self.context).decision,'allow')
        external=FakeReviewer(); external.descriptor=external.descriptor.model_copy(update={'execution':'external'})
        with self.assertRaises(ValueError):
            build_compliance_service(reviewer=external,allow_test=True,allow_external=False)

    @patch('banking_chatbot.chat.generate_answer')
    def test_exact_hit_is_reviewed_and_rejected_entry_evicted(self, generate):
        generate.return_value={'status':'generated','answer':self.candidate.text,'finish_reason':'STOP',
            'model':'gemini-2.5-flash-lite','usage':{},'source_ids':[], 'elapsed_ms':1}
        generate.side_effect = lambda q, matches, *args: dict(generate.return_value, source_ids=[m['id'] for m in matches[:5]])
        reviewer=FakeReviewer(); gate=service(reviewer)
        with tempfile.TemporaryDirectory() as tmp:
            cache=SQLiteResponseCache(Path(tmp)/'cache.db')
            args=dict(retriever=Retriever(),use_gemini=True,api_key='test',cache=cache,compliance_service=gate)
            first=chat(self.context.question,[],**args)
            second=chat(self.context.question,[],**args)
            self.assertEqual(second['cache']['status'],'hit')
            self.assertEqual(len(reviewer.calls),2)
            reviewer.outcome='fail'
            rejected=chat(self.context.question,[],**args)
            self.assertEqual(rejected['compliance']['decision'],'block')
            self.assertNotIn('answer',rejected['generation'])
            reviewer.outcome='pass'
            chat(self.context.question,[],**args)
            self.assertEqual(generate.call_count,2)
            self.assertEqual(first['compliance']['decision'],'allow')

    @patch('banking_chatbot.chat.generate_answer')
    def test_blocked_output_not_cached_and_preview_sanitized(self, generate):
        generate.return_value={'status':'generated','answer':'SSN 123-45-6789','finish_reason':'STOP','model':'demo'}
        reviewer=FakeReviewer()
        turn=chat(self.context.question,[],retriever=Retriever(),use_gemini=True,
                  compliance_service=service(reviewer),cache_mode='off')
        self.assertEqual(turn['generation']['status'],'block')
        self.assertNotIn('123-45-6789',json.dumps(turn))
        self.assertEqual(reviewer.calls,[])

    @patch('banking_chatbot.chat.generate_answer')
    def test_semantic_hit_cannot_bypass_reviewer(self, generate):
        from test_semantic_cache import FixedRetriever, MapEmbedder, generation
        retriever = FixedRetriever()
        vectors = {'how do i change my pin': [1, 0],
                   'what is the process to reset my pin': [1, 0]}
        reviewer = FakeReviewer()
        gate = service(reviewer)
        generate.return_value = generation('In this demo, use the app.')
        with tempfile.TemporaryDirectory() as tmp:
            cache = SQLiteResponseCache(Path(tmp) / 'semantic.db', embedder=MapEmbedder(vectors),
                                        semantic_threshold=0.9)
            args = dict(retriever=retriever, use_gemini=True, api_key='test', cache=cache,
                        cache_mode='semantic', compliance_service=gate)
            chat('how do i change my pin', [], **args)
            reviewer.outcome = 'fail'
            turn = chat('what is the process to reset my pin', [], **args)
            self.assertEqual(turn['compliance']['origin'], 'semantic_cache')
            self.assertEqual(turn['compliance']['decision'], 'block')
            self.assertNotIn('answer', turn['generation'])
            self.assertEqual(generate.call_count, 1)

    @patch('banking_chatbot.chat.generate_answer')
    def test_component_change_invalidates_cached_approval(self, generate):
        generate.side_effect = lambda q, matches, h, k, m: {
            'status':'generated', 'answer':'Demo answer', 'finish_reason':'STOP',
            'model':m, 'usage':{}, 'elapsed_ms':1, 'source_ids':[s['id'] for s in matches[:5]]}
        reviewer = FakeReviewer()
        with tempfile.TemporaryDirectory() as tmp:
            cache = SQLiteResponseCache(Path(tmp) / 'version.db')
            args = dict(retriever=Retriever(), use_gemini=True, api_key='test', cache=cache)
            chat(self.context.question, [], compliance_service=service(reviewer), **args)
            reviewer.descriptor = reviewer.descriptor.model_copy(update={'model':'replacement'})
            turn = chat(self.context.question, [], compliance_service=service(reviewer), **args)
            self.assertEqual(turn['cache']['status'], 'miss')
            self.assertEqual(generate.call_count, 2)

    def test_source_previews_remove_identifiers_without_network(self):
        class Sources:
            def search(self, *args, **kwargs):
                return {'matches':[{'id':'a','question':'Profile', 'answer':'SSN 123-45-6789'},
                                   {'id':'b','question':'Contact', 'answer':'Call 800-555-1212'}]}
        turn = chat('q', [], retriever=Sources())
        self.assertNotIn('123-45-6789', json.dumps(turn))
        self.assertNotIn('800-555-1212', json.dumps(turn))
        self.assertIsNone(turn['generation'])

    def test_factory_registry_rejects_unknown_components(self):
        with self.assertRaises(ValueError):
            build_compliance_service(reviewer=FakeReviewer(), allow_test=True, validator_ids=('unknown',))

    def test_no_approved_history_leaks_into_generator(self):
        from banking_chatbot.gemini import recent_conversation
        history=[{'query':'q','generation':{'status':'generated','answer':'unapproved'}}]
        self.assertEqual(recent_conversation(history),[])


if __name__=='__main__': unittest.main()
