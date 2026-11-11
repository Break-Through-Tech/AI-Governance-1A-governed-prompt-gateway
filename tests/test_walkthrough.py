"""The demo's graph walkthrough: the path a run took and the sentence for each step."""

import re
import unittest

from langgraph.graph import END, START

from gateway.graph import GRAPH
from gateway.llm import GenerationResult, Provider
from gateway.pipeline import process
from gateway.schema import GuardrailDecision
from gateway.walkthrough import EXIT_LABELS, OUTCOMES, TITLES, diagram, narrate, path_taken


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
    def test_every_compiled_node_is_drawn_once_with_its_title(self):
        text = diagram()
        for name in GRAPH.get_graph().nodes:
            if name != END:  # END is drawn once per exit, see below
                self.assertEqual(text.count(f">{TITLES[name]}<"), 1, name)
        self.assertNotIn('class="visited"', text)  # nothing has run yet, so nothing is lit

    def test_every_compiled_edge_is_drawn_once(self):
        text = diagram()
        graph = GRAPH.get_graph()
        self.assertEqual(text.count("<line "), len(graph.edges))
        for edge in graph.edges:
            if edge.target == END:  # one outcome box per exit (box titles are the bold texts)
                self.assertEqual(len(re.findall(f'font-weight="600"[^>]*>{OUTCOMES[edge.source]}<', text)), 1, edge)

    def test_every_conditional_exit_has_a_label_and_nothing_else_does(self):
        for edge in GRAPH.get_graph().edges:
            self.assertEqual(edge.conditional, (edge.source, edge.target) in EXIT_LABELS, edge)

    def test_current_node_and_taken_edges_are_highlighted(self):
        text = diagram(["guardrails", "cache"], finished=False)
        self.assertEqual(text.count('class="visited"'), 2)  # start, guardrails
        self.assertEqual(text.count('class="current"'), 1)  # cache
        self.assertEqual(text.count('marker-end="url(#lit)"'), 2)  # start->guardrails, guardrails->cache
        self.assertNotIn('class="reached"', text)

    def test_finished_run_fills_in_the_outcome_it_reached(self):
        text = diagram(["guardrails"], finished=True)
        self.assertNotIn('class="current"', text)
        self.assertEqual(text.count('class="reached"'), 1)
        self.assertRegex(text, r'class="reached"[^>]*/>\s*<text[^>]*>blocked<')
        self.assertEqual(text.count('marker-end="url(#lit)"'), 2)  # start->guardrails, guardrails->blocked

    def test_diagram_is_well_formed_svg(self):
        import xml.dom.minidom
        for visited, finished in ([], False), (["guardrails", "cache"], False), (["guardrails"], True):
            xml.dom.minidom.parseString(diagram(visited, finished=finished))

    def test_path_taken(self):
        self.assertEqual(path_taken([], finished=False), [])
        self.assertEqual(path_taken(["guardrails"], finished=False), [START, "guardrails"])
        self.assertEqual(path_taken(["guardrails"], finished=True), [START, "guardrails", END])


if __name__ == "__main__":
    unittest.main()
