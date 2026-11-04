from banking_chatbot.intent.workflow import build_gateway_graph
# testing for the "whole thing", feel free to modify for integration testing if thats easier

def test_pipeline():
    # Create LangGraph workflow
    app = build_gateway_graph()
    
    # Test queries across different complexity tiers
    test_queries = [
        "What are your branch opening hours?",
        "How do I reset my forgotten passcode?",
        "I need to report a fraudulent unauthorized transaction on my account."
    ]
    
    for query in test_queries:
        print(f"\nRunning Query: '{query}'")
        initial_state = {"query_raw": query}
        
        # Invoke the graph pipeline (classifier -> execution -> feedback)
        final_state = app.invoke(initial_state)
        
        print(f"-> Assigned Intent: {final_state.get('intent_label')} (Confidence: {final_state.get('intent_confidence'):.2f})")
        print(f"-> Model Tier Used: {final_state.get('model_tier')} ({final_state.get('model_name')})")
        print(f"-> Projected Cost:  ${final_state.get('projected_cost_usd'):.6f}")
        print(f"-> Actual Cost:     ${final_state.get('actual_cost_usd', 0.0):.6f}")

if __name__ == "__main__":
    test_pipeline()