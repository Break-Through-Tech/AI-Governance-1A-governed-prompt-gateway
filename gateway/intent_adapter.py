"""Stage 3 adapter: Team C's Banking77 classifier behind ``classify_intent``.

Their module (``banking_chatbot.intent``) is a small pipeline of its own: it
predicts the intent, picks a model tier and a prompt, counts tokens with
LiteLLM and calls the model. The gateway already has a router (stage 4) and a
provider (stage 5), so this adapter takes the part that is stage 3's job, the
predicted label and its confidence, and records their tier suggestion without
acting on it. Whether the router should follow that suggestion is a team
decision, not something an adapter should settle.

Two things have to exist before the classifier can run, and neither is in the
repo: the ``litellm`` package (their module imports it) and the trained model
file, which is git-ignored. Until both are present the stage reports intent
``unknown`` and says exactly what is missing. It switches on by itself after.
"""

from __future__ import annotations

from .schema import IntentDecision

MODEL_FILE = "banking_chatbot/intent/models/banking77_classifier.joblib"

# Result of the one import attempt: (classifier module, prompts module) or a reason string.
_IMPORT: tuple | str | None = None


def _modules() -> tuple | str:
    global _IMPORT
    if _IMPORT is None:
        try:
            from banking_chatbot.intent import classifier, prompts
            _IMPORT = (classifier, prompts)
        except ImportError as exc:
            _IMPORT = f"Team C's classifier could not be imported: {exc.name or exc} is not installed"
    return _IMPORT


def classify(query: str) -> IntentDecision:
    modules = _modules()
    if isinstance(modules, str):
        return IntentDecision(reason=modules)
    classifier, prompts = modules
    if classifier.MODEL_PIPELINE is None or classifier.LABEL_NAMES is None:
        # Their code would return a hardcoded ("general_banking_faq", 0.70) here. Recording a
        # made-up confidence in an audit log would be wrong, so say what is missing instead.
        return IntentDecision(reason=f"Team C's classifier is merged but its model file is missing: {MODEL_FILE}")

    label, confidence = classifier.predict_intent(query)
    suggestion = prompts.get_prompt_and_routing_tier(label)["model_tier"]
    return IntentDecision(
        label=label,
        confidence=round(float(confidence), 4),
        suggested_tier=suggestion,
        reason=f"Banking77 classifier (Team C); it suggests the {suggestion} tier",
    )
