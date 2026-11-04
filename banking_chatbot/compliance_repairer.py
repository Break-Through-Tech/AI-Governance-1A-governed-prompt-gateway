"""Regenerate from evidence without sending the rejected draft or judge prose."""
from .compliance_contracts import ComponentDescriptor, RepairResult, Usage
from .gemini import DEFAULT_MODEL, SUPPORTED_MODELS, generate_answer

REPAIRABLE = ('RESPONSE_FORMAT', 'SUPPORTED_FACTS', 'FICTIONAL_CONTEXT')


class GeminiRepairer:
    def __init__(self, api_key, model=DEFAULT_MODEL):
        if model not in SUPPORTED_MODELS:
            raise ValueError('Unsupported repair model')
        self._key, self._model = api_key, model
        self.descriptor = ComponentDescriptor(component_id='repairer', version='1',
            rule_ids=REPAIRABLE, execution='external', model=model,
            prompt_version='evidence-regeneration-v1')

    def repair(self, request):
        if not set(request.rule_ids) <= set(REPAIRABLE):
            raise ValueError('Nonrepairable violation')
        # The existing generation policy already enforces concise fictional grounding.
        # A new answer is requested; neither a failed draft nor evaluator instructions
        # are available to this adapter.
        answer = generate_answer(request.context.question,
            [r.model_dump() for r in request.context.references], [], self._key, self._model)
        return RepairResult(text=answer['answer'], finish_reason=answer['finish_reason'],
                            usage=Usage(**answer['usage']))
