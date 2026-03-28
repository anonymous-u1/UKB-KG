## UKB-KG Construction
To facilitate monitoring of intermediate outputs, the pipeline is modularized into multiple sequential steps.

### LLM Settings
Please configure your OpenAI `base_url` and `api_key` in `llm/openai_chat.py`. We provide functions for calling GPT-5 series models, supporting both structured and unstructured outputs.

### Triple Extraction
From text:
- Extract Triples: `extract_from_text/01_extract.sh`

From table:
- Parse tables: `extract_from_table/01_parse_tables.py`
- Select tables containing extractable biomedical relationships: `extract_from_table/02_select_tables.sh`
- Extract Triples: `extract_from_table/03_extract_from_tables.sh` with `MODE='triples'`.

### Triple Refinement
- Triple Filtering and Revision: Run `extract_from_text/02_filter_or_refine.sh` sequentially with `MODE=llm_filter`, `refine`, and `rule_filter`. (Apply the same procedure to table-derived triples.)
- Triple Verification: `extract_from_text/03_verify_or_eval.sh` with `MODE='verify'` (Self-verification; same for table-derived triples.)
- Entity Alignment: `Proprocess/01_umls_link.py`
- KG Fusion: `Proprocess/02_bios_completion.py`
- Entity Typing and processing: `Proprocess/03_process.py`
- Convert to CSV format: `Proprocess/04_json_to_csv.py`
- Add attributes: `Proprocess/05_add_attributes.py`

### Evaluation
- LLM-based evaluation: `extract_from_text/03_verify_or_eval.sh` with `MODE='eval'`. (e.g., using a strong LLM such as GPT-5.4; same for table-derived triples)
- Metric calculation: `extract_from_text/calc_eval_scores.py`

### Baseline Extraction
- Select tables containing study cohort demographics: `extract_from_table/02_select_tables.sh` (Skip if already executed during triple extraction.)
- Extract baseline information: `extract_from_table/03_extract_from_tables.sh` with `MODE='baseline'`.

Note that some minor manual processing may be required during the workflow if needed (e.g., merging triples extracted from text and tables).