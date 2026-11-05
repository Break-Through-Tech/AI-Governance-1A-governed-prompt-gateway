import json
from banking_chatbot.intent.classifier import process_intent_stage

print("=" * 70)
print("  BANKING CHATBOT - INTENT CLASSIFIER INTERACTIVE TEST")
print("  Type your custom query and press Enter. (Type 'exit' or 'q' to quit)")
print("=" * 70)

while True:
    try:
        user_input = input("\nEnter prompt > ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "q", "quit"]:
            print("Exiting test session.")
            break

        # Simulate pipeline state passing through the stage
        state = {"query_raw": user_input, "query_masked": user_input}
        updated_state = process_intent_stage(state)

        # Print structured inspection output
        print("-" * 50)
        print(f"Predicted Intent : {updated_state.get('intent_label')}")
        print(f"Confidence Score : {updated_state.get('intent_confidence'):.4f}")
        print(f"Category Group   : {updated_state.get('intent_category')}")
        print(f"Cacheable        : {updated_state.get('cacheable_intent')}")
        print(f"Cache TTL        : {updated_state.get('cache_ttl_seconds')} seconds")
        print(f"System Prompt    :\n\"{updated_state.get('system_prompt')}\"")
        print("-" * 50)

    except KeyboardInterrupt:
        print("\nSession interrupted.")
        break