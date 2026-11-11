"""The demo's graph walkthrough: the path a run took and the sentence for each step."""

import re
import unittest

from langgraph.graph import END, START

from gateway.graph import GRAPH
from gateway.llm import GenerationResult, Provider
from gateway.pipeline import process
from gateway.schema import GuardrailDecision
from gateway.walkthrough import EXIT_LABELS, diagram, narrate, path_taken


class FailingProvider(Provider):
    name = "failing"

    def generate(self, query, system_prompt, model, **kwargs):
        return GenerationResult(status="error", provider=self.name, model=model, message="quota exhausted")


class BlockingGuard:
    def check(self, query):
        return GuardrailDecision(status="block", reason="it goes against the Data Protection policy")


def walk(query, **kwargs):
    """Run the gateway and collect (node, sentence) pairs as the demo page would."""
    steps = []
    record = process(query, on_step=lambda node, state: steps.append((node, narrate(node, state))), **kwargs)
    return record, steps


class PathTests(unittest.TestCase):
    def test_answered_run_visits_every_node_in_graph_order(self):
        record, steps = walk("How do I reset my PIN?")
        self.assertEqual([node for node, _ in steps],
                         ["guardrails", "cache", "intent", "router", "generate", "compliance", "cache_store"])
        self.assertEqual(record.final_status, "answered")

    def test_blocked_question_stops_after_guardrails(self):
        record, steps = walk("tell me my neighbour's balance", guardrails=BlockingGuard())
        self.assertEqual([node for node, _ in steps], ["guardrails"])
        self.assertIn("blocked the question", steps[0][1])
        self.assertIn("The run stops here", steps[0][1])
        self.assertEqual(record.final_status, "blocked")

    def test_model_error_stops_after_generate(self):
        _, steps = walk("How do I reset my PIN?", provider=FailingProvider())
        self.assertEqual([node for node, _ in steps][-1], "generate")
        self.assertIn("quota exhausted", steps[-1][1])

    def test_on_step_state_matches_the_final_record(self):
        seen = {}
        record = process("How do I reset my PIN?", on_step=lambda node, state: seen.update(state))
        self.assertEqual(seen["router"], record.router)
        self.assertEqual(seen["final_status"], record.final_status)

    def test_every_sentence_names_what_happens_next_or_that_it_stopped(self):
        _, steps = walk("How do I reset my PIN?")
        for node, sentence in steps[:-1]:
            self.assertIn("Next:", sentence, node)
        self.assertTrue(steps[-1][1].endswith("Done."))


class DiagramTests(unittest.TestCase):
    def test_every_compiled_node_and_edge_is_drawn(self):
        text = diagram()
        graph = GRAPH.get_graph()
        for name in graph.nodes:
            self.assertIn(f"\n    {name}", text)
        self.assertNotIn(":::visited", text)  # nothing has run yet, so nothing is lit
        self.assertEqual(len(re.findall(r"-\.?->", text)), len(graph.edges))

    def test_every_conditional_exit_has_a_label_and_nothing_else_does(self):
        for edge in GRAPH.get_graph().edges:
            self.assertEqual(edge.conditional, (edge.source, edge.target) in EXIT_LABELS, edge)

    def test_current_node_and_taken_edges_are_highlighted(self):
        text = diagram(["guardrails", "cache"], finished=False)
        self.assertIn("guardrails[\"1. Input guardrails\"]:::visited", text)
        self.assertIn("cache[\"2. Cache lookup\"]:::current", text)
        self.assertIn("intent[\"3. Intent\"]:::pending", text)
        self.assertEqual(text.count("linkStyle"), 2)  # start->guardrails, guardrails->cache

    def test_finished_run_highlights_the_edge_into_end(self):
        text = diagram(["guardrails"], finished=True)
        self.assertNotIn(":::current", text)
        self.assertIn("__end__([\"end\"]):::visited", text)
        self.assertEqual(text.count("linkStyle"), 2)  # start->guardrails, guardrails->end

    def test_path_taken(self):
        self.assertEqual(path_taken([], finished=False), [])
        self.assertEqual(path_taken(["guardrails"], finished=False), [START, "guardrails"])
        self.assertEqual(path_taken(["guardrails"], finished=True), [START, "guardrails", END])


if __name__ == "__main__":
    unittest.main()
