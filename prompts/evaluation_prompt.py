evaluation_prompt_structured = """# INSTRUCTION  
--------------------
A corpus from a medical paper is provided in the *INPUT_TEXT*, and the triples extracted from the text are provided in the *EXTRACTED_TRIPLES*. Your task is to evaluate whether each triple is "Correct" or "Incorrect" based on the text.  

# EVALUATION_CRITERIA
--------------------
- Correct: The relationship expressed in the triple is clearly supported by the information in the original text.
- Incorrect: The relationship expressed in the triple is not supported by the information in the original text.

# INPUT_TEXT
--------------------
<<text>>

# EXTRACTED_TRIPLES
--------------------
Each triple is structured as: head|relation|tail
<<triples>>

# RESPONSE_FORMAT 
--------------------
Evaluate each extracted triple ONE BY ONE based on the *EVALUATION_CRITERIA* and categorize it as "Correct" or "Incorrect." 
The evaluation results must follow the exact order of the input triples, and must adhere to the following JSON schema:
```json
{
  "Evaluations": [
    {"Result": "Correct or Incorrect"},
    {"Result": "Correct or Incorrect"},
    ...
  ]
}
```
"""

recall_evaluation_prompt_structured = """
# INSTRUCTION
--------------------
You are given a set of ground-truth triples in *GROUND_TRUTH_TRIPLES* and a set of model-extracted triples in *EXTRACTED_TRIPLES*.
Your task is to determine, for each ground-truth triple, whether its meaning is covered by the extracted triples.

This evaluation is recall-oriented: mark a ground-truth triple as "Covered" when an extracted triple conveys essentially the same biomedical meaning, even if the wording is not identical.

# MATCHING_RULES
--------------------
Mark a ground-truth triple as "Covered" if at least one extracted triple satisfies all of the following:
1. The head and tail entities refer to the same or essentially equivalent biomedical concepts.
2. The relation expresses the same core meaning.
   Exact relation wording is not required. More general or approximate relations are acceptable as long as they preserve the main meaning.

For example:
   - [ leukocyte telomere length | associated_with | hypertension ] ≈ [ high blood pressure | linked_to | LTL ]
   - [ retinal vessels | used_by_algorithms_to_predict | age ] ≈ [ retinal vessels | associated_with | age ]
   - [ CKD | presents_in_retina_as | sparse capillaries ] ≈ [ chronic kidney disease | associated_with | retinal capillary rarefaction ]
   - [ rs7412 | located_in | APOE ] ≈ [ rs7412 | is_a_variant_in_the_gene | APOE ] 
   - [ estimated glomerular filtration rate | negatively_associated_with | chronic kidney disease progression ] ≈ [ eGFR | inversely_related_to | CKD progression ]
   - [ C-reactive protein | elevated_in | rheumatoid arthritis ] ≈ [ CRP | is_higher_among_patients_with | rheumatoid arthritis ]

# EVALUATION_CRITERIA
--------------------
- Covered: At least one extracted triple matches the ground-truth triple under the *MATCHING_RULES*.
- Not Covered: No extracted triple matches the ground-truth triple under the *MATCHING_RULES*.

# GROUND_TRUTH_TRIPLES
--------------------
Each triple is structured as: [ head | relation | tail ]
<<gt_triples>>

# EXTRACTED_TRIPLES
--------------------
Each triple is structured as: [ head | relation | tail ]
<<extracted_triples>>

# RESPONSE_FORMAT
--------------------
Evaluate each ground-truth triple ONE BY ONE and categorize it as "Covered" or "Not Covered." 
The evaluation results must follow the exact order of the ground-truth triples, and must adhere to the following JSON schema:
```json
{
  "Evaluations": [
    {"Result": "Covered | Not Covered"},
    {"Result": "Covered | Not Covered"},
    ...
  ]
}
```
"""