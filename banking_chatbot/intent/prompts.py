"""
    Configures LLM pricing tiers, maps Banking77 intent categories 
    into complexities, and defines system prompts and caching 
    rules for model routing
"""
# Cost per 1K tokens for current active model routing tiers (per 1M converted to per 1K)
MODEL_PRICING = {
    "fast-cheap": {
        "model_name": "gemini-3.8-flash",
        "input_cost_per_1k": 0.00075,   # $0.75 per 1M tokens
        "output_cost_per_1k": 0.00375   # $3.75 per 1M tokens
    },
    "standard-balanced": {
        "model_name": "gemini-3.1-pro-preview",
        "input_cost_per_1k": 0.0020,    # $2.00 per 1M tokens (up to 200k context)
        "output_cost_per_1k": 0.0120    # $12.00 per 1M tokens
    },
    "high-reasoning": {
        "model_name": "claude-3-5-sonnet",
        "input_cost_per_1k": 0.0030,    # $3.00 per 1M tokens
        "output_cost_per_1k": 0.0150    # $15.00 per 1M tokens
    }
}

# INTENT MAPPING w/ banking77

# Low complexity: simple read-only info -> routed to gemini-3.8-flash
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

# Medium complexity: standard account actions & transfers -> routed to gemini-3.1-pro-preview
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

# Fallback: unlisted intents (fraud, disputes, security) route to claude-3-5-sonnet


def get_prompt_and_routing_tier(intent_label: str) -> dict:
    # Match intent label to its appropriate model tier and system prompt
    if intent_label in LOW_COMPLEXITY_INTENTS:
        return {
            "prompt": "You are a helpful customer service assistant for a digital bank. Provide clear, concise answers.",
            "model_tier": "fast-cheap"
        }
    elif intent_label in MEDIUM_COMPLEXITY_INTENTS:
        return {
            "prompt": "You are a secure banking assistant helping with account settings and transfers. Provide step-by-step guidance.",
            "model_tier": "standard-balanced"
        }
    
    # High-stakes fallback for security / fraud cases
    return {
        "prompt": "You are a senior compliance officer assisting with sensitive financial disputes and fraud claims.",
        "model_tier": "high-reasoning"
    }