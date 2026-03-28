pubmedqa_none = """# INSTRUCTION:
Answer the question given in *QUESTION* based on your own knowledge.
The answer must be one of the "yes," "no" or "maybe", and it should be wrapped within <ans></ans>.

# QUESTION: 
<<question>>

# RESPONSE:
"""

pubmedqa_cot_0shot = """# INSTRUCTION:
Answer the question given in *QUESTION* based on your own knowledge.
Follow these steps to arrive at your conclusion:
1. Explain the reasoning behind your answer. 
2. Determine the answer, which must be one of the "yes," "no" or "maybe". And the answer should be placed between <ans></ans>.

# QUESTION: 
<<question>>

# RESPONSE:
"""

pubmedqa_cot_1shot = """# INSTRUCTION:
Answer the question given in *QUESTION* based on your own knowledge.
Follow these steps to arrive at your conclusion:
1. Explain the reasoning behind your answer. 
2. Determine the answer, which must be one of the "yes," "no" or "maybe". And the answer should be placed between <ans></ans>.

# EXAMPLE: 
- Question: Does regular consumption of A reduce the risk of B disease?
- Response: 
1. Reasoning: It is well-established that regular A consumption has the potential to reduce C levels, and C is a significant risk factor for B. So it can be assumed that A reduces the risk of B disease by lowering C levels.
2. <ans>yes</ans>

# QUESTION: 
<<question>>

# RESPONSE:
"""

pubmedqa_rag_0shot = """# INSTRUCTION:
Answer the question given in *QUESTION* based on your own knowledge and supplemented by the context derived from a medical knowledge graph given in *CONTEXT*.

Follow these steps to arrive at your conclusion:
1. Explain the reasoning behind your answer.
2. Determine the answer, which must be one of the "yes," "no" or "maybe". And the answer should be placed between <ans></ans>.

Please note:
1. If you have sufficient domain knowledge or are confident in the answer, respond based on your own knowledge without relying on the CONTEXT. If you are unfamiliar with the question or uncertain about the answer, refer to the information provided in *CONTEXT* to guide your decision.
2. Disregard any irrelevant information.

# RESPONSE FORMAT
1. Analysis and reasons for the answer.
2. <ans>yes or no or maybe</ans>

# QUESTION:
<<question>>

# CONTEXT:
<<context>>

# QUESTION:
<<question>>

# RESPONSE:
"""

pubmedqa_rag_1shot = """# INSTRUCTION:
Answer the question given in *QUESTION* based on your own knowledge and supplemented by the context derived from a medical knowledge graph given in *CONTEXT*.

Follow these steps to arrive at your conclusion:
1. Explain the reasoning behind your answer.
2. Determine the answer, which must be one of the "yes," "no" or "maybe". And the answer should be placed between <ans></ans>.

Please note:
1. If you have sufficient domain knowledge or are confident in the answer, respond based on your own knowledge without relying on the CONTEXT. If you are unfamiliar with the question or uncertain about the answer, refer to the information provided in *CONTEXT* to guide your decision.
2. Disregard any irrelevant information.

# RESPONSE FORMAT
1. Analysis and reasons for the answer.
2. <ans>yes or no or maybe</ans>

# EXAMPLE:
- Question: Does regular consumption of A reduce the risk of B disease?
- Context:
... Triple: [C, risk_factor_of, B] ...
- Response:
1. Reasoning: It is well-established that regular A consumption has the potential to reduce C levels. Although there is no direct evidence linking A to B, the triple provided in the additional context mentions that C is a risk factor for B. So it can be assumed that A reduces the risk of B disease by lowering C levels.
2. <ans>yes</ans>

# QUESTION:
<<question>>

# CONTEXT:
<<context>>

# QUESTION:
<<question>>

# RESPONSE:
"""

bioasq_none = """# INSTRUCTION:
Answer the question given in *QUESTION* based on your own knowledge.
The answer must be one of the "yes" or "no", and it should be wrapped within <ans></ans>.

# QUESTION: 
<<question>>

# RESPONSE:
"""

bioasq_cot_0shot = """# INSTRUCTION:
Answer the question given in *QUESTION* based on your own knowledge.
Follow these steps to arrive at your conclusion:
1. Explain the reasoning behind your answer. 
2. Determine the answer, which must be one of the "yes" or "no". And the answer should be placed between <ans></ans>.

# QUESTION: 
<<question>>

# RESPONSE:
"""

bioasq_cot_1shot = """# INSTRUCTION:
Answer the question given in *QUESTION* based on your own knowledge.
Follow these steps to arrive at your conclusion:
1. Explain the reasoning behind your answer. 
2. Determine the answer, which must be one of the "yes" or "no". And the answer should be placed between <ans></ans>.

# EXAMPLE: 
- Question: Does regular consumption of A reduce the risk of B disease?
- Response: 
1. Reasoning: It is well-established that regular A consumption has the potential to reduce C levels, and C is a significant risk factor for B. So it can be assumed that A reduces the risk of B disease by lowering C levels.
2. <ans>yes</ans>

# QUESTION: 
<<question>>

# RESPONSE:
"""

bioasq_rag_0shot = """# INSTRUCTION:
Answer the question given in *QUESTION* based on your own knowledge and supplemented by the context derived from a medical knowledge graph given in *CONTEXT*.
Follow these steps to arrive at your conclusion:
1. Explain the reasoning behind your answer.
2. Determine the answer, which must be one of the "yes" or "no". And the answer should be placed between <ans></ans>.

# RESPONSE FORMAT
1. Analysis and reasons for the answer.
2. <ans>yes or no</ans>

# QUESTION:
<<question>>

# CONTEXT:
<<context>>

# QUESTION:
<<question>>

# RESPONSE:
"""

bioasq_rag_1shot = """# INSTRUCTION:
Answer the question given in *QUESTION* based on your own knowledge and supplemented by the context derived from a medical knowledge graph given in *CONTEXT*.
Follow these steps to arrive at your conclusion:
1. Explain the reasoning behind your answer.
2. Determine the answer, which must be one of the "yes" or "no". And the answer should be placed between <ans></ans>.

# RESPONSE FORMAT
1. Analysis and reasons for the answer.
2. <ans>yes or no</ans>

# EXAMPLE:
- Question: Does regular consumption of A reduce the risk of B disease?
- Context:
... Triple: [C, risk_factor_of, B] ...
- Response:
1. Reasoning: It is well-established that regular A consumption has the potential to reduce C levels. Although there is no direct evidence linking A to B, the triple provided in the additional context mentions that C is a risk factor for B. So it can be assumed that A reduces the risk of B disease by lowering C levels.
2. <ans>yes</ans>

# QUESTION:
<<question>>

# CONTEXT:
<<context>>

# QUESTION:
<<question>>

# RESPONSE:
"""