import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gateway.env import load_dotenv, parse_dotenv
from gateway.llm import GeminiProvider


class ParseDotenvTests(unittest.TestCase):
    def test_parses_keys_and_strips_quotes_and_comments(self):
        text = '# comment\n\nA=1\nB = "two"\nC=\'three\'\nnot a pair\n=novalue\n'
        self.assertEqual(parse_dotenv(text), {"A": "1", "B": "two", "C": "three"})

    def test_value_may_contain_equals(self):
        self.assertEqual(parse_dotenv("K=a=b"), {"K": "a=b"})


class LoadDotenvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / ".env"

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_is_fine(self):
        self.assertEqual(load_dotenv(self.path), {})

    def test_sets_variables_but_never_overrides_the_shell(self):
        self.path.write_text("GW_TEST_NEW=from-file\nGW_TEST_SET=from-file\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"GW_TEST_SET": "from-shell"}, clear=False):
            os.environ.pop("GW_TEST_NEW", None)
            applied = load_dotenv(self.path)
            self.assertEqual(applied, {"GW_TEST_NEW": "from-file"})
            self.assertEqual(os.environ["GW_TEST_NEW"], "from-file")
            self.assertEqual(os.environ["GW_TEST_SET"], "from-shell")


class ProviderUsesDotenvTests(unittest.TestCase):
    def test_provider_reads_key_from_dotenv_when_shell_has_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text('GEMINI_API_KEY="file-key"\n', encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GEMINI_API_KEY", None)
                with mock.patch("gateway.llm.load_dotenv", lambda: load_dotenv(path)):
                    self.assertEqual(GeminiProvider().api_key, "file-key")

    def test_explicit_key_skips_dotenv(self):
        with mock.patch("gateway.llm.load_dotenv", side_effect=AssertionError("should not load")):
            self.assertEqual(GeminiProvider(api_key="explicit").api_key, "explicit")


if __name__ == "__main__":
    unittest.main()
