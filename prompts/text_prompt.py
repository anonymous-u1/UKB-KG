entity_filter_prompt_structured = """## INSTRUCTION:
You are given a list of candidate entities extracted from a scientific article. 
Your task is to identify and retain only valid medical or biomedical entities.

Remove entities that are:
- Non-medical concepts or generic words without biomedical meaning
- Study design elements (e.g., "analysis", "method", "result")
- Statistical terms (e.g., "p-value", "confidence interval")

Note: 
Only select entities from the original list. Do not modify them. 

## CANDIDATE ENTITIES:
```
<<entities>>
```
"""

get_triple_prompt_structured = """# INSTRUCTION
--------------------
You are a researcher skilled in summarizing scientific findings into concise and informative triples to construct a medical knowledge graph.
The corpus is provided in *TEXT* with medical entities marked between <ent> and </ent>. Your task is to extract meaningful medical triples from *TEXT*.
1. Triple Structure: (Entity1, Relation, Entity2)
 - Entity1 and Entity2 are the medical entities annotated with <ent> and </ent> in the text.
 - Relation represents the semantic relationship between Entity1 and Entity2, inferred from the context.
2. Guidelines for Extraction:
 - Focus on key findings reported in the research. Exclude relationships that do not represent meaningful biomedical knowledge, such as:
   - abbreviation definitions (e.g., [A, abbreviation_of, B]),
   - vague or non-informative relations (e.g., [diabetes, associated_with, diseases], [BMI, is, risk factors], [risk factors, yields, risk]),
   - expressions that do not convey a clear biomedical fact (e.g., [loss, of, noxious heat sensation]).
 - Ensure that the extracted triples reflect factual statements supported by the text, rather than speculative or hypothetical statements.
 - Occasionally, an entity that should remain a single term may be incorrectly split into multiple entities in the text. In such cases, merge them into the correct complete entity when extracting triples. For example, "<ent>breast</ent> <ent>cancer</ent>" should be extracted as "breast cancer".
 - Prefer full names of entities rather than abbreviations whenever possible.
 - If two triples express essentially the same meaning (e.g., due to different relation wording or reversed entity order), extract only one of them.
 
# EXAMPLES
--------------------
- Example text:
```
<ent>smoking</ent> has been widely recognized as a leading cause of higher risk of <ent>lung cancer</ent>.... This study suggests <ent>smoking</ent> leads to lower <ent>longevity</ent> by 1~3 year
```
- Example response:
```json
{
  "Triples": [
    {"Entity1": "smoking", "Relation": "risk_factor_of", "Entity2": "lung cancer"},
    {"Entity1": "smoking", "Relation": "decreases", "Entity2": "longevity"}
  ]
}
```

# TEXT
-------------------- 
Here is the text for extraction. Please extract the medical triples and output them in the specified format:
```
<<text>>
```
"""

get_triple_0shot_prompt_structured = """# INSTRUCTION
--------------------
You are a researcher skilled in summarizing scientific findings into concise and informative triples to construct a medical knowledge graph.
The corpus is provided in *TEXT* with medical entities marked between <ent> and </ent>. Your task is to extract meaningful medical triples from *TEXT*.
1. Triple Structure: (Entity1, Relation, Entity2)
 - Entity1 and Entity2 are the medical entities annotated with <ent> and </ent> in the text.
 - Relation represents the semantic relationship between Entity1 and Entity2, inferred from the context.
2. Guidelines for Extraction:
 - Focus on key findings reported in the research. Exclude relationships that do not represent meaningful biomedical knowledge, such as:
   - abbreviation definitions (e.g., [A, abbreviation_of, B]),
   - vague or non-informative relations (e.g., [diabetes, associated_with, diseases], [BMI, is, risk factors], [risk factors, yields, risk]),
   - expressions that do not convey a clear biomedical fact (e.g., [loss, of, noxious heat sensation]).
 - Ensure that the extracted triples reflect factual statements supported by the text, rather than speculative or hypothetical statements.
 - Occasionally, an entity that should remain a single term may be incorrectly split into multiple entities in the text. In such cases, merge them into the correct complete entity when extracting triples. For example, "<ent>breast</ent> <ent>cancer</ent>" should be extracted as "breast cancer".
 - Prefer full names of entities rather than abbreviations whenever possible.
 - If two triples express essentially the same meaning (e.g., due to different relation wording or reversed entity order), extract only one of them.

# TEXT
-------------------- 
Here is the text for extraction. Please extract the medical triples and output them in the specified format:
```
<<text>>
```
"""

get_triple_5shot_prompt_structured = """# INSTRUCTION
--------------------
You are a researcher skilled in summarizing scientific findings into concise and informative triples to construct a medical knowledge graph.
The corpus is provided in *TEXT* with medical entities marked between <ent> and </ent>. Your task is to extract meaningful medical triples from *TEXT*.
1. Triple Structure: (Entity1, Relation, Entity2)
 - Entity1 and Entity2 are the medical entities annotated with <ent> and </ent> in the text.
 - Relation represents the semantic relationship between Entity1 and Entity2, inferred from the context.
2. Guidelines for Extraction:
 - Focus on key findings reported in the research. Exclude relationships that do not represent meaningful biomedical knowledge, such as:
   - abbreviation definitions (e.g., [A, abbreviation_of, B]),
   - vague or non-informative relations (e.g., [diabetes, associated_with, diseases], [BMI, is, risk factors], [risk factors, yields, risk]),
   - expressions that do not convey a clear biomedical fact (e.g., [loss, of, noxious heat sensation]).
 - Ensure that the extracted triples reflect factual statements supported by the text, rather than speculative or hypothetical statements.
 - Occasionally, an entity that should remain a single term may be incorrectly split into multiple entities in the text. In such cases, merge them into the correct complete entity when extracting triples. For example, "<ent>breast</ent> <ent>cancer</ent>" should be extracted as "breast cancer".
 - Prefer full names of entities rather than abbreviations whenever possible.
 - If two triples express essentially the same meaning (e.g., due to different relation wording or reversed entity order), extract only one of them.

# EXAMPLES
--------------------
1. Example 1
- Example text:
```
<ent>smoking</ent> has been widely recognized as a leading cause of higher risk of <ent>lung cancer</ent>.... This study suggests <ent>smoking</ent> leads to lower <ent>longevity</ent> by 1~3 year
```
- Example response:
```json
{
  "Triples": [
    {"Entity1": "smoking", "Relation": "risk_factor_of", "Entity2": "lung cancer"},
    {"Entity1": "smoking", "Relation": "decreases", "Entity2": "longevity"}
  ]
}
```
2. Example 2
- Example text:
```
<ent>Metformin</ent> has been shown to improve <ent>insulin sensitivity</ent> and reduce <ent>blood glucose</ent> levels.
```
- Example response:
```json
{
  "Triples": [
    {"Entity1": "metformin", "Relation": "improves", "Entity2": "insulin sensitivity"},
    {"Entity1": "metformin", "Relation": "reduces", "Entity2": "blood glucose"}
  ]
}
```
3. Example 3
- Example text:
```
We examined the relationship between <ent>vitamin D</ent>, <ent>obesity</ent>, and <ent>depression</ent> in a cohort study.
```
- Example response:
```json
{
  "Triples": []
}
```
4. Example 4
- Example text:
```
<ent>BRCA1 gene mutation</ent> is a major genetic cause of <ent>breast cancer</ent> and <ent>ovarian cancer</ent>.
```
- Example response:
```json
{
  "Triples": [
    {"Entity1": "BRCA1 gene mutation", "Relation": "increases_risk_of", "Entity2": "breast cancer"},
    {"Entity1": "BRCA1 gene mutation", "Relation": "increases_risk_of", "Entity2": "ovarian cancer"}
  ]
}
```
5. Example 5
- Example text:
```
High intake of <ent>omega-3 fatty acids</ent> may protect against <ent>coronary heart disease</ent> by lowering <ent>triglyceride</ent> levels.
```
- Example response:
```json
{
  "Triples": [
    {"Entity1": "omega-3 fatty acids", "Relation": "protects_against", "Entity2": "coronary heart disease"},
    {"Entity1": "omega-3 fatty acids", "Relation": "reduces", "Entity2": "triglyceride"}
  ]
}
```

# TEXT
-------------------- 
Here is the text for extraction. Please extract the medical triples and output them in the specified format:
```
<<text>>
```
"""

get_triple_wo_ner_prompt_structured = """# INSTRUCTION
--------------------
You are a researcher skilled in summarizing scientific findings into concise and informative triples to construct a medical knowledge graph.
The corpus is provided in *TEXT*. Your task is to extract meaningful medical triples from *TEXT*.
1. Triple Structure: (Entity1, Relation, Entity2)
 - Entity1 and Entity2 are the medical entities in the text.
 - Relation represents the semantic relationship between Entity1 and Entity2, inferred from the context.
2. Guidelines for Extraction:
 - Focus on key findings reported in the research. Exclude relationships that do not represent meaningful biomedical knowledge, such as:
   - abbreviation definitions (e.g., [A, abbreviation_of, B]),
   - vague or non-informative relations (e.g., [diabetes, associated_with, diseases], [BMI, is, risk factors], [risk factors, yields, risk]),
   - expressions that do not convey a clear biomedical fact (e.g., [loss, of, noxious heat sensation]).
 - Ensure that the extracted triples reflect factual statements supported by the text, rather than speculative or hypothetical statements.
 - Prefer full names of entities rather than abbreviations whenever possible.
 - If two triples express essentially the same meaning (e.g., due to different relation wording or reversed entity order), extract only one of them.

# EXAMPLES
--------------------
- Example text:
```
smoking has been widely recognized as a leading cause of higher risk of lung cancer.... This study suggests smoking leads to lower longevity by 1~3 year
```
- Example response:
```json
{
  "Triples": [
    {"Entity1": "smoking", "Relation": "risk_factor_of", "Entity2": "lung cancer"},
    {"Entity1": "smoking", "Relation": "decreases", "Entity2": "longevity"}
  ]
}
```

# TEXT
-------------------- 
Here is the text for extraction. Please extract the medical triples and output them in the specified format:
```
<<text>>
```
"""

triple_filter_prompt_structured = """# INSTRUCTION
--------------------
You are given a set of triples extracted from a scientific article.
Your task is to filter out low-quality triples and keep only valuable biomedical knowledge triples.

# FILTERING RULES
--------------------
1. Remove semantically unclear or poorly defined triples.
   Filter out triples whose meaning is unclear, incomplete, or not interpretable as a valid biomedical relation.
   For example: [ CKD | of_uncertain | aetiology ], [ hypertension | highlights_in_models | retinal vessels ]

2. Remove low-value or non-informative triples. 
   Discard triples that do NOT convey meaningful biomedical knowledge or research findings, including but not limited to:
   - Trivial or obvious statements
   - Abbreviation or definition relations
   - Statements lacking scientific or biomedical significance
   - Methodological or meta-level statements (not biomedical findings)
   For example: [ age | interrelated_with | risk factors ], [ health data | contributes_to | mental healthcare ], [admixture | affects_reliability_of | methods], [post-trauma stress disorder | is | PTSD], [familial kidney failure | occurs_in | families], [genomic variants | regulate | genes]

3. Deduplicate semantically equivalent triples.
   If multiple triples express essentially the same meaning: keep only ONE representative triple and remove the others.

# OUTPUT REQUIREMENTS
--------------------
- Output ONLY the filtered triples
- Include all valid triples that should be retained
- Do NOT modify the wording of triples

# CANDIDATE TRIPLES
--------------------
Each triple is structured as: [ head | relation | tail ]

<<triples>>
"""

triple_refine_prompt_structured = """# INSTRUCTION
--------------------
You are given a set of triples extracted from a scientific article.
Your task is to refine low-quality or non-standard triples while preserving their factual accuracy and original meaning as much as possible.

# REFINE RULES
--------------------
1. Refine relations:
   If a relation is overly complex, overly specific, overly long, or otherwise non-standard, rewrite it into a more concise and canonical form.
   
2. Refine entities：
   - Entities must be biomedical entities.
   - If an entity is overly long or overly specific with redundant information, remove redundant descriptive details (such as statistical values) that are not necessary for preserving the correctness of the triple.
   - Prefer full names over abbreviations when you are confident about the expansion and the full name fits naturally in context.
   
3. Preserve accuracy:
   - Ensure that the refined triple remains factually correct. Do not overstate, strengthen, or exaggerate the original relation or entities beyond the evidence.
   - Only make refinements that improve clarity, normalization, or biomedical correctness.
   
4. Remove unrecoverable triples:
   - If a triple is too unclear, malformed, or semantically invalid to be reliably refined, remove it.
   
# REFINE EXAMPLES:
--------------------
- [ retinal vessels | used_by_algorithms_to_predict | age ] -> [ retinal vessels | associated_with | age ]
- [ CKD | presents_in_retina_as | sparse capillaries ] -> [ chronic kidney disease | associated_with | retinal capillary rarefaction ]
- [ rs7412 | is_a_variant_in_the_gene | APOE ] -> [ rs7412 | located_in | APOE ]
- [ metformin | leads_to_lower | HbA1c levels ] -> [ metformin | decreases | HbA1c ]
- [ CRP | is_higher_among_patients_with | rheumatoid arthritis ] -> [ C-reactive protein | elevated_in | rheumatoid arthritis ]
- [ APOE4 | increases_the_risk_for_developing | Alzheimer’s disease ] -> [ APOE4 | increases_risk_of | Alzheimer's disease ]

# OUTPUT REQUIREMENTS
--------------------
- Keep acceptable triples unchanged. Refine only when necessary.
- For refined triples, ensure the revised version is accurate and standardized.
- Remove triples that are unacceptable and cannot be reliably refined.

# TRIPLES
--------------------
Each triple is structured as: [ head | relation | tail ]. 

<<triples>>
"""