# banking_chatbot/intent/classifier.py

import os
import re
import warnings
import joblib
import numpy as np
from typing import Dict, Any
from sklearn.exceptions import InconsistentVersionWarning
from .prompts import get_prompt_and_routing_tier, MODEL_PRICING

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "banking77_classifier.joblib")

if os.path.exists(MODEL_PATH):
    artifact = joblib.load(MODEL_PATH)
    MODEL_PIPELINE = artifact["pipeline"]
    LABEL_NAMES = artifact["label_names"]
else:
    MODEL_PIPELINE = None
    LABEL_NAMES = None


def clean_text(text: str) -> str:
    # Basic text normalization for vectorizer matching
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict_intent(query_text: str) -> tuple[str, float]:
    # Run text through scikit-learn pipeline to get class probabilities
    if MODEL_PIPELINE and LABEL_NAMES:
        cleaned_query = clean_text(query_text)
        probabilities = MODEL_PIPELINE.predict_proba([cleaned_query])[0]
        best_idx = np.argmax(probabilities)
        
        predicted_label = LABEL_NAMES[best_idx]
        confidence = float(probabilities[best_idx])
        
        # Confidence Threshold Fallback:
        # If the model is guessing with low certainty, route to a safe default 
        # instead of trusting an incorrect fine-grained label.
        CONFIDENCE_THRESHOLD = 0.20
        if confidence < CONFIDENCE_THRESHOLD:
            return "general_banking_faq", confidence
            
        return predicted_label, confidence
    
    return "general_banking_faq", 0.70


def estimate_llm_cost_and_route(state: Dict[str, Any]) -> Dict[str, Any]:
    # LangGraph node: predicts intent, picks model tier, and estimates token cost
    query = state.get("query_masked") or state.get("query_raw", "")
    
    # 1. Get intent prediction & confidence score (with threshold check)
    intent_label, confidence = predict_intent(query)
    
    # 2. Look up prompt template and target model tier based on intent complexity
    rules = get_prompt_and_routing_tier(intent_label)
    tier = rules["model_tier"]
    pricing = MODEL_PRICING[tier]
    
    # 3. Estimate tokens (rough heuristic: ~4 chars per token)
    estimated_input_tokens = max(10, len(query) // 4)
    estimated_output_tokens = 150  # average banking response length
    
    # 4. Calculate projected cost in USD
    input_cost = (estimated_input_tokens / 1000.0) * pricing["input_cost_per_1k"]
    output_cost = (estimated_output_tokens / 1000.0) * pricing["output_cost_per_1k"]
    total_projected_cost = input_cost + output_cost
    
    # 5. Populate state for downstream graph steps
    state["intent_label"] = intent_label
    state["intent_confidence"] = confidence
    state["system_prompt"] = rules["prompt"]
    state["model_tier"] = tier
    state["model_name"] = pricing["model_name"]
    state["estimated_input_tokens"] = estimated_input_tokens
    state["estimated_output_tokens"] = estimated_output_tokens
    state["projected_cost_usd"] = round(total_projected_cost, 6)
    state["cache_ttl_seconds"] = rules["ttl_seconds"]
    state["cacheable_intent"] = rules["cacheable"]
    
    return state