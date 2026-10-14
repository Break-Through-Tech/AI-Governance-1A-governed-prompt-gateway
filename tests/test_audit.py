import json
import tempfile
import unittest
from pathlib import Path

from gateway.audit import AuditLog
from gateway.pipeline import process
from gateway.schema import DecisionRecord, GenerationResult


class FailingProvider:
    name = "failing"

    def generate(self, query, system_prompt, model, **_):
        return GenerationResult(status="error", provider=self.name, model=model, message="boom")


class AuditLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = AuditLog(Path(self.tmp.name) / "nested" / "decisions.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_reads_as_empty(self):
        self.assertEqual(list(self.log.read()), [])
        self.assertEqual(len(self.log), 0)

    def test_append_creates_parent_directory_and_one_line_per_record(self):
        self.log.append(DecisionRecord(query="first"))
        self.log.append(DecisionRecord(query="second"))
        lines = self.log.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["query"], "first")
        self.assertEqual(json.loads(lines[1])["query"], "second")

    def test_read_returns_records_oldest_first(self):
        for q in ("a", "b", "c"):
            self.log.append(DecisionRecord(query=q))
        self.assertEqual([r["query"] for r in self.log.read()], ["a", "b", "c"])

    def test_process_appends_full_record(self):
        record = process("How do I reset my PIN?", audit=self.log)
        stored = list(self.log.read())
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["request_id"], record.request_id)
        self.assertEqual(stored[0]["router"]["tier"], record.router.tier)
        self.assertEqual(stored[0]["final_status"], "answered")

    def test_errors_are_logged_too(self):
        process("hello", provider=FailingProvider(), audit=self.log)
        stored = list(self.log.read())
        self.assertEqual(stored[0]["final_status"], "error")
        self.assertIsNone(stored[0]["output_compliance"])

    def test_process_without_audit_writes_nothing(self):
        process("hello")
        self.assertFalse(self.log.path.exists())


if __name__ == "__main__":
    unittest.main()
