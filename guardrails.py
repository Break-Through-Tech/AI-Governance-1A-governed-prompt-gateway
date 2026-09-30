from langchain_openai import ChatOpenAI
from langchain.agents import create_chat_agent
from langchain.middleware import PIIMiddleware
import re

"""
This is the guardrails module for detecting prompt injection attacks using a deterministic appraoch 


Verbs and targets used for detecting prompt injection attacks. they hold the regular expressions for common verbs and targets that might indicate an attempt to override instructions or manipulate the system.
prompt injection attacks. we can use the libary re in python to match these patterns within user inputs.

"""

VERBS = r"(?:ignore|disregard|forget|clear|override|bypass|drop|reset)"
TARGETS = r"(?:previous|prior|above|existing|system|original|initial|all)\s+(?:instructions?|prompts?|rules?|directives?|guidelines?|context)"


# this is a dictionary containing regular expression patterns for detecting various types of prompt injection attacks.
# if a user input matches any of these patterns, it may indicate a prompt injection attempt. therfore it will print the type of violation detected.
INJECTION_PATTERNS= { 

 "INSTRUCTIONS OVERRIDE":rf"\b{VERBS}\b(?:\s+\w+){{0,3}}\s+{TARGETS}",
 # Matches: "you are now in developer mode", "you are now a hacker", "act as DAN"
"ROLE_HIJACK": r"(?i)\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|from\s+now\s+on\s+you\s+are)\b\s+(?:an?\s+)?(?:unrestricted|unfiltered|jailbroken|evil|dan|developer|root|admin)",
    
 # Matches: "enter developer mode", "switch to debug mode", "enable god mode"
 "MODE_SWITCH": r"(?i)\b(?:enter|enable|switch\s+to|activate)\b\s+(?:developer|debug|test|god|unrestricted|maintenance)\s+mode\b",
    
 # Matches: "print system prompt", "show your initial instructions", "reveal prompt above"
 "PROMPT_EXTRACTION": r"(?i)\b(?:print|show|display|reveal|repeat|output)\b(?:\s+\w+){0,3}\s+(?:system\s+prompt|initial\s+instructions?|hidden\s+prompt|rules\s+above)"

}
"""
this will be where the llm will evaluate the rest of the prompt and decided to block the mesesage or let it pass on to the llm part of the 
llm. 

"""




