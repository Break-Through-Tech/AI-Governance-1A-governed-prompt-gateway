# ==============================================================================
# LOCAL STANDALONE TEST SCRIPT FOR INTENT CLASSIFICATION & COST ROUTING
# ==============================================================================
import sys
import os

# Ensure project root is in path when running from tests/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from banking_chatbot.intent.workflow import build_gateway_graph

app = build_gateway_graph()

# Diverse test suite covering different intents, typos, and complexities
test_queries = [
    # 1. Low Complexity / Fast-Cheap Tier (Expected: gemini-1.5-flash)
    "What are your branch operating hours on Saturday?",
    "What's the current exchange rate from USD to EUR?",
    "When will my new debit card arrive?",
    "Do you support contactless payments?",
    "Hi, what is a savings account?",

    # 2. High Complexity / Security Tier (Expected: claude-3-5-sonnet)
    "I see a charge on my statement from a store I've never visited.",
    "My physical credit card was stolen at the airport yesterday.",
    "How do I dispute a fraudulent wire transfer?",
    "Why was I charged an extra overdraft fee on my account?",
    "I forgot my mobile banking passcode and need to reset it.",

    # 3. Typos, Noise, & Missing Spaces (Robustness checks)
    "hw do i resset my passwrd for the appp",
    "forgotpasswordhowdoireset",
    "charged $10.50 extra fee on my card???!!!",
    "whrs the nearest atm around here??",
]

print("\n" + "=" * 140)
print(f"{'QUERY':<42} | {'PREDICTED INTENT':<26} | {'CONF':<6} | {'MODEL':<18} | {'TOK':<5} | {'COST'}")
print("=" * 140)

for query in test_queries:
    initial_state = {
        "query_raw": query,
        "query_masked": query,
        "guardrail_status": "ALLOW"
    }
    
    # Invoke LangGraph workflow
    result_state = app.invoke(initial_state)
    
    total_tokens = (
        result_state.get('estimated_input_tokens', 0) + 
        result_state.get('estimated_output_tokens', 0)
    )
    
    confidence = result_state.get('intent_confidence', 0.0)
    
    print(
        f"{query[:40]:<42} | "
        f"{str(result_state.get('intent_label')):<26} | "
        f"{confidence:<6.2f} | "
        f"{str(result_state.get('model_name')):<18} | "
        f"{total_tokens:<5} | "
        f"${result_state.get('projected_cost_usd', 0.0):.5f}"
    )

print("=" * 140 + "\n")