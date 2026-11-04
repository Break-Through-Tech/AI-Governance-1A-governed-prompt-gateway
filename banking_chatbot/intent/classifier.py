""" 
    Handles text cleaning, loads the scikit-learn Banking77 model 
    pipeline, applies confidence thresholds, and calculates 
    projected token counts and USD inference costs (which
    should be toyed around w/ later)
"""

import os
import re
import warnings
import joblib
import numpy as np
from typing import Dict, Any
from sklearn.exceptions import InconsistentVersionWarning
from litellm import token_counter, cost_per_token
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
    # Run text through intent model
    if MODEL_PIPELINE and LABEL_NAMES:
        cleaned_query = clean_text(query_text)
        probabilities = MODEL_PIPELINE.predict_proba([cleaned_query])[0]
        best_idx = np.argmax(probabilities)
        
        predicted_label = LABEL_NAMES[best_idx]
        confidence = float(probabilities[best_idx])
        
        # Confidence Threshold Fallback (0.20):
        # If the model is guessing with low certainty, route to safe default 
        # instead of trusting an incorrect fine-grained label
        CONFIDENCE_THRESHOLD = 0.20
        if confidence < CONFIDENCE_THRESHOLD:
            return "general_banking_faq", confidence
            
        return predicted_label, confidence
    
    return "general_banking_faq", 0.70


def estimate_llm_cost_and_route(state: dict) -> dict:
    # Supports both teammate's "query" key and local test keys gracefully
    query = state.get("query", state.get("query_raw", ""))
    
    # Predict intent using your scikit-learn model
    intent_label, confidence = predict_intent(query)
    
    # Get the assigned model tier, system prompt, and expected output tokens from prompts.py
    routing_info = get_prompt_and_routing_tier(intent_label)
    tier_name = routing_info["model_tier"]
    pricing = MODEL_PRICING[tier_name]
    model_name = pricing["model_name"]
    system_prompt = routing_info["prompt"]
    estimated_output_tokens = routing_info["expected_output_tokens"]
    
    # Token counting via LiteLLM
    # Combine the system prompt and the user query into the standard message structure
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    try:
        # Ask LiteLLM's universal tokenizer engine for the exact token count
        estimated_input_tokens = token_counter(model=model_name, messages=messages)
    except Exception:
        # Safe offline fallback if run without proper network/registry access
        full_payload = f"{system_prompt}\nUser: {query}"
        estimated_input_tokens = max(10, len(full_payload) // 4)
        print("Could not get LiteLLM token counter to work, using heuristic fallback.")

    # Calculate precise dollar costs using LiteLLM's pricing engine
    try:
        input_cost, output_cost = cost_per_token(
            model=model_name,
            prompt_tokens=estimated_input_tokens,
            completion_tokens=estimated_output_tokens
        )
        total_cost = input_cost + output_cost
    except Exception:
        # Fallback math if model string isn't in LiteLLM registry yet
        input_cost = (estimated_input_tokens / 1000.0) * pricing["input_cost_per_1k"]
        output_cost = (estimated_output_tokens / 1000.0) * pricing["output_cost_per_1k"]
        total_cost = input_cost + output_cost

    # Build the structured "intent" block expected by teammate's Streamlit UI & Pipeline
    intent_block = {
        "status": "pass",
        "label": intent_label,
        "confidence": confidence,
        "prompt_template": system_prompt[:40] + "..." if len(system_prompt) > 40 else system_prompt,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "projected_cost_usd": round(total_cost, 6)
    }

    # Write everything back to the shared state dictionary
    state.update({
        "intent": intent_block,  # <--- Crucial dictionary block for teammate's UI (Stage 3)
        "intent_label": intent_label,
        "intent_confidence": confidence,
        "model_tier": tier_name,
        "model_name": model_name,
        "system_prompt": system_prompt,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "projected_cost_usd": round(total_cost, 6)
    })
    
    return state