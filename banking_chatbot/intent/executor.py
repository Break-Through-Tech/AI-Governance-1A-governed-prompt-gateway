""" 
    Executes the LLM call via LiteLLM, captures actual token usage, 
    and compares against estimates for the feedback loop.
"""

import time
from litellm import completion, completion_cost
from litellm.exceptions import ServiceUnavailableError

def execute_llm_call(state: dict) -> dict:
    model_name = state.get("model_name", "gemini/gemini-1.5-flash")
    system_prompt = state.get("system_prompt", "")
    query = state.get("query_raw", "")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    # Some retry info
    response = None
    max_retries = 3
    retry_delay = 2.0 
    
    # Retry loop
    for attempt in range(max_retries + 1):
        try:
            response = completion(
                model=model_name,
                messages=messages
            )
            break
        except Exception as e:
            error_str = str(e).lower()
            is_transient = "503" in error_str or "unavailable" in error_str or isinstance(e, ServiceUnavailableError)
            
            if attempt == max_retries or not is_transient:
                raise e
            
            print(f"\n503/Unavailable error. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2  # Double the wait time for each try
    
    # Extract actual metrics from the response provider metadata
    response_message = response.choices[0].message.content
    actual_input_tokens = response.usage.prompt_tokens
    actual_output_tokens = response.usage.completion_tokens
    
    # Calculate true exact cost using LiteLLM
    actual_cost = completion_cost(completion_response=response)
    
    # Feedback loop log
    # Compare estimates against reality
    est_input = state.get("estimated_input_tokens", 0)
    est_output = state.get("estimated_output_tokens", 0)
    est_cost = state.get("projected_cost_usd", 0.0)
    
    print(f"\n--- [FEEDBACK LOOP AUDIT] ---")
    print(f"Intent: {state.get('intent_label')}")
    print(f"Input Tokens  -> Estimated: {est_input} | Actual: {actual_input_tokens}")
    print(f"Output Tokens -> Estimated: {est_output} | Actual: {actual_output_tokens}")
    print(f"Cost (USD)    -> Estimated: ${est_cost:.6f} | Actual: ${float(actual_cost):.6f}")
    print(f"-----------------------------\n")
    
    # 5. Save results to state
    state.update({
        "response_content": response_message,
        "actual_input_tokens": actual_input_tokens,
        "actual_output_tokens": actual_output_tokens,
        "actual_cost_usd": float(actual_cost)
    })
    
    return state