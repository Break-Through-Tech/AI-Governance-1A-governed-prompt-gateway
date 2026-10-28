# ==============================================================================
# INTENT POLICY & MODEL ROUTING CONFIGURATION
# ==============================================================================

# Cost per 1K tokens for model routing tiers
MODEL_PRICING = {
    "fast-cheap": {
        "model_name": "gemini-1.5-flash",
        "input_cost_per_1k": 0.000075,
        "output_cost_per_1k": 0.0003
    },
    "standard-balanced": {
        "model_name": "gemini-1.5-pro",
        "input_cost_per_1k": 0.00125,
        "output_cost_per_1k": 0.0050
    },
    "high-reasoning": {
        "model_name": "claude-3-5-sonnet",
        "input_cost_per_1k": 0.0030,
        "output_cost_per_1k": 0.0150
    }
}

# ------------------------------------------------------------------------------
# INTENT COMPLEXITY MAPS (Banking77 Categories)
# ------------------------------------------------------------------------------

# Low complexity: simple read-only info -> routed to gemini-1.5-flash
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

# Medium complexity: standard account actions & transfers -> routed to gemini-1.5-pro
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
    # Match intent label to its appropriate model tier and prompt template
    if intent_label in LOW_COMPLEXITY_INTENTS:
        return {
            "prompt": "You are a helpful customer service assistant for a digital bank. Provide clear, concise answers.",
            "ttl_seconds": 3600,
            "cacheable": True,
            "category": "CACHEABLE_SHORT_TTL",
            "model_tier": "fast-cheap"
        }
    elif intent_label in MEDIUM_COMPLEXITY_INTENTS:
        return {
            "prompt": "You are a secure banking assistant helping with account settings and transfers. Provide step-by-step guidance.",
            "ttl_seconds": 86400,
            "cacheable": True,
            "category": "CACHEABLE_MEDIUM_TTL",
            "model_tier": "standard-balanced"
        }
    
    # High-stakes fallback for security / fraud cases
    return {
        "prompt": "You are a senior compliance officer assisting with sensitive financial disputes and fraud claims.",
        "ttl_seconds": 86400,
        "cacheable": False,
        "category": "SECURE_LONG_TTL",
        "model_tier": "high-reasoning"
    }