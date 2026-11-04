"""
    Configures LLM pricing tiers, maps Banking77 intent categories 
    into complexities, and defines system prompts and caching 
    rules for model routing
"""
# Cost per 1K tokens for current active model routing tiers (per 1M converted to per 1K)
# Currently hardcoded testing values, and all models are 3.1 flash lite because it was
# less finnicky to test w/
MODEL_PRICING = {
    "fast-cheap": {
        "model_name": "gemini/gemini-3.1-flash-lite",  
        "input_cost_per_1k": 0.00025,
        "output_cost_per_1k": 0.00150
    },
    "standard-balanced": {
        "model_name": "gemini/gemini-3.1-flash-lite",
        "input_cost_per_1k": 0.00075,
        "output_cost_per_1k": 0.00375
    },
    "high-reasoning": {
        "model_name": "gemini/gemini-3.1-flash-lite",
        "input_cost_per_1k": 0.0030,
        "output_cost_per_1k": 0.0150
    }
}

# setup default output tokens and tier mapping policy
INTENT_TOKEN_POLICY = {
    "branch_operating_hours": {"tier": "fast-cheap", "expected_output_tokens": 60},
    "passcode_forgotten": {"tier": "standard-balanced", "expected_output_tokens": 180},
    "direct_debit_dispute": {"tier": "high-reasoning", "expected_output_tokens": 350},
}


# INTENT MAPPING w/ banking77

LOW_COMPLEXITY_INTENTS = {
    "branch_operating_hours",
    "exchange_rate",
    "card_delivery_estimate",
    "supported_cards_and_currencies",
    "general_banking_faq",
    "atm_support",
    "contactless_not_working",
    "getting_virtual_card",
    "get_disposable_virtual_card",
    "verify_source_of_funds",
    "top_up_limits",
    "card_about_to_expire",
    "apple_pay_or_google_pay"
}

MEDIUM_COMPLEXITY_INTENTS = {
    "account_management_faq",
    "transfer_timing",
    "card_linking",
    "passcode_forgotten",
    "change_pin",
    "top_up_reverted",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "balance_not_updated_after_bank_transfer",
    "transfer_fee_charged",
    "transfer_into_account",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_acceptance",
    "pin_blocked",
    "edit_personal_details"
}



def get_prompt_and_routing_tier(intent_label: str) -> dict:
    # Match intent label to its appropriate model tier, system prompt, and expected output tokens
    
    # Check if the specific intent has an explicit policy override
    policy = INTENT_TOKEN_POLICY.get(intent_label, {})
    expected_tokens = policy.get("expected_output_tokens")

    if intent_label in LOW_COMPLEXITY_INTENTS:
        return {
            "prompt": "You are a helpful customer service assistant for a digital bank. Provide clear, concise answers.",
            "model_tier": "fast-cheap",
            "expected_output_tokens": expected_tokens if expected_tokens is not None else 60
        }
    elif intent_label in MEDIUM_COMPLEXITY_INTENTS:
        return {
            "prompt": "You are a secure banking assistant helping with account settings and transfers. Provide step-by-step guidance.",
            "model_tier": "standard-balanced",
            "expected_output_tokens": expected_tokens if expected_tokens is not None else 180
        }
    
    # Fallback for higher security issues
    return {
        "prompt": "You are a senior compliance officer assisting with sensitive financial disputes and fraud claims.",
        "model_tier": "high-reasoning",
        "expected_output_tokens": expected_tokens if expected_tokens is not None else 350
    }