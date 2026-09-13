#!/bin/bash
# Single source of truth for one KG build. Copy this file, edit the copy, and
# pass it with --config to run_extraction.sh / run_postprocess.sh.
#
# Everything a run produces lives under $SAVE_ROOT/$RUN_NAME.
# Ablations are just a different RUN_NAME plus the relevant setting below.

# ------------------------------------------------------------------ identity
RUN_NAME="sample"
SAVE_ROOT="save"

# ----------------------------------------------------------------------- LLM
LLM="gpt-5"
EFFORT="minimal"

# Shards run concurrently within a stage. Articles are processed independently,
# so this only affects wall-clock time, never the extracted triples. 
# Keep it under your API rate limit.
NUM_SHARDS=4

# ------------------------------------------------------------------ input data
PAPER_LIST="PMC_Data/pmcid-ukb-xml-sample.txt"
XML_DIR="PMC_Data/xml"

# Optional: metadata CSV used only as a title fallback when an XML fails to
# parse. Leave empty to skip the fallback. See README > Data Requirements.
TITLE_FILE=""

# Required by run_postprocess.sh only. Publication metadata used for the
# per-edge confidence score. See README > Data Requirements.
PUBLICATION_METADATA="PMC_Data/pubmed-ukb.csv"

# Markdown tables parsed out of the XML, and the two selected subsets.
TABLE_DIR="PMC_Data/tables"
RELATION_TABLE_DIR="PMC_Data/rel_tables"
BASELINE_TABLE_DIR="PMC_Data/baseline_tables"

# ------------------------------------------------------------------- prompts
# Names of module-level variables in prompts/. Swapping these is how the
# few-shot ablations are run, e.g. GET_TRIPLE_PROMPT=get_triple_0shot_prompt_structured
ENTITY_FILTER_PROMPT="entity_filter_prompt_structured"
GET_TRIPLE_PROMPT="get_triple_prompt_structured"
FILTER_PROMPT="triple_filter_prompt_structured"
REFINE_PROMPT="triple_refine_prompt_structured"
TEXT_EVAL_PROMPT="evaluation_prompt_structured"
SELECT_TABLE_PROMPT="select_table_prompt_structured"
TABLE_TRIPLE_PROMPT="extract_triples_from_tables_structured"
TABLE_EVAL_PROMPT="evaluation_table_prompt_structured"
BASELINE_PROMPT="extract_baseline_from_tables_structured"

# -------------------------------------------------------------------- models
SPACY_MODEL="en_core_sci_scibert"
ENTITY_TYPE_MODEL="entity_type_tag/save/save_35w_256"

# ----------------------------------------------------------- post-processing
# Similarity floor for accepting a UMLS canonical name.
UMLS_THRESHOLD=0.98

# BIOS v3 core data, licensed separately. See README > Data Requirements.
BIOS_CONCEPTS="BIOS/bios_v3_release/CoreData/ConceptTerms.txt"
BIOS_TRIPLETS="BIOS/bios_v3_release/CoreData/RelationTriplets.txt"

# Set to 0 to build the KG from literature triples only.
ENABLE_BIOS_COMPLETION=1

# Reference year for the recency term of the confidence score.
CURRENT_YEAR=2025

# ------------------------------------------------------------------ ablation
# 1 skips NER and the entity-filter call, sending raw article text to the
# triple prompt. Pair with GET_TRIPLE_PROMPT=get_triple_wo_ner_prompt_structured.
ABLATION_WO_NER=0

# Judge model for the Evaluation section (a stronger model than $LLM).
EVAL_LLM="gpt-5.4"
EVAL_EFFORT="medium"
