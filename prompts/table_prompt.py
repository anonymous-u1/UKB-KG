select_table_prompt_structured = """# INSTRUCTION
--------------------
You will be provided with several tables from a scientific article. Each table includes a title, body, and possibly footnotes.
Your task is to identify two types of tables based on their content:
1. Biomedical Relationship Tables
2. Baseline / Cohort Demographic Tables

# CATEGORY DEFINITIONS
--------------------
## 1. Biomedical Relationship Tables
Select tables that describe meaningful biomedical, genetic, or clinical relationships, which can be used to extract knowledge graph triples.

Examples of such relationships include:
- Associations or correlations between biomedical entities
- Causal or effect-based relationships
- Treatment or intervention outcomes
- Risk or protective factors
- Biomarker, gene, protein, or metabolite relationships with diseases or phenotypes
- Genetic or genomic relationships, including:
  - SNP–Gene / Gene–Phenotype / SNP–Phenotype associations, Chromosome–Locus–Trait mappings, eQTL / GWAS / TWAS findings, Omics-derived associations

Exclude tables that:
- describe demographics
- summarize datasets or metadata
- describe procedural, methodological or administrative information
- summarize computational methods, algorithms, or models rather than biological relationships

## 2. Baseline / Cohort Demographic Tables
Select tables that describe study cohort demographics or baseline characteristics, including: cohort name, sample size, age distribution, gender distribution, racial distribution, educational attainment, employment status.

# IMPORTANT RULES
--------------------
- Return a list of table IDs for each category: ["<Table ID 1>", ..., "<Table ID N>"]
- If a table does not match either category, ignore the table and do not include it in any category.
- If a table has no content (e.g., the table body is empty or missing), ignore the table and do not include it in any category.
- If no tables match a category, return an empty list `[]` for that category.

# TABLES
--------------------
Below are the tables to analyze. Identify which belong to each category and return the result in the JSON schema.
```markdown
<<tables>>
``` 
"""

extract_triples_from_tables_structured = """# INSTRUCTION
--------------------
You are an expert biomedical researcher specializing in extracting structured knowledge from scientific tables.

You will be provided with one or more tables (including title, body, and possibly footnotes).  
Your task is to extract **high-quality biomedical knowledge triples** for constructing a medical knowledge graph.

Triple form: (Entity1, Relation, Entity2)

# EXTRACTION RULES:
--------------------
1. Extract only meaningful biomedical relationships (e.g., gene–phenotype, variant–trait, biomarker–disease).
2. Entities must be biomedical (e.g., disease, phenotype, gene, variant, biomarker). Prefer full names over abbreviations. Do NOT include numbers, statistical values, or any redundant descriptive information in the entity. 
3. Prefer concise and standardized relations (e.g., associated_with, located_in, increases_risk_of, or similar canonical forms) and avoid vague or overly long relations, while ensuring the relation is directly supported by the table and not overstated beyond the original evidence.
4. Ignore non-relational content: baseline/demographic tables, definitions, methods, or purely statistical descriptions.
5. Do not include trivial or obvious knowledge.
6. Do not extract triples that are semantically equivalent or redundant.
7. If no valid biomedical relationships exist, return {"Triples": []}.

# EXAMPLES
--------------------
- Example table:

**SNP Characteristics and Associated Phenotypes in the Study**

| SNP | Gene | Biomarkers |
|------|-------|------|
| rs7412 | APOE | LDL-C |
| rs1801133 | MTHFR | Homocysteine |

- Example response:

```json
{
  "Triples": [
    {"Entity1": "rs7412", "Relation": "located_in", "Entity2": "APOE"},
    {"Entity1": "rs7412", "Relation": "associated_with", "Entity2": "low-density lipoprotein cholesterol"},
    {"Entity1": "rs1801133", "Relation": "located_in", "Entity2": "MTHFR"},
    {"Entity1": "rs1801133", "Relation": "associated_with", "Entity2": "Homocysteine"}
  ]
}
```

- Example table:

**cis-eQTL Associations in Lung Tissue**

| SNP | Gene | Tissue | p-value |
|------|-------|------|------|
| rs743590 | SULT1A1 | Lung | 1e-8 |
| rs10948723 | GSTA1 | Lung | 0.3 |

- Example response:

```json
{
  "Triples": [
    {"Entity1": "rs743590", "Relation": "cis_eQTL_of", "Entity2": "SULT1A1"},
    {"Entity1": "rs743590", "Relation": "associated_with", "Entity2": "lung tissue"}
  ]
}
```

# TABLES
--------------------
Below are the tables to analyze. Extract only clear, scientifically meaningful medical triples according to the rules above:
```markdown
<<tables>>
```
"""

extract_baseline_from_tables_structured = """# INSTRUCTION
--------------------
You will be provided with one or more tables (each including a title, table body, and possibly footnotes).  
Your task is to extract **cohort-level demographic and baseline characteristics** in a **structured and standardized format**.

1. What to extract:
For each identifiable cohort or subgroup described in the table, extract the following fields:
- **cohort_description**: A short description of what this cohort represents or is used for in the study (e.g., "Baseline characteristics of lung cancer patients").
- **cohort_name**: Name of the cohort or data source (e.g., UK Biobank, FinnGen, GLGC).
- **sample_size**: total sample size of this cohort or subgroup (N).
- **mean_age**: `X (SD = Y)`, or `X` if SD not available.
- **age_distribution**: `<category> (N, %) | <category> (N, %) | ...`
- **gender_distribution**: `Males (N, %) | Females (N, %)`
- **racial_distribution**: `<race/ethnicity> (N, %) | <race/ethnicity> (N, %) | ...`
- **educational_attainment**: `<category> (N, %) | <category> (N, %) | ...`
- **employment_status**: `<category> (N, %) | <category> (N, %) | ...` 

2. Multiple cohorts or subgroups:
- If a table describes **multiple cohorts or subgroups** (e.g., cases vs controls, coffee drinkers vs non-drinkers, multiple GWAS datasets):
- Extract **each cohort separately**, and clearly distinguish them using **cohort_description**.

3. Output rules:
- If only one of N or % is available, report the available value. 
- If a specific field is not mentioned for a cohort/subgroup, set that field to `"None"`.
- If no baseline information can be extracted from the table(s), return an empty list `[]`.

# RESPONSE_FORMAT
--------------------
If cohort information is found, output:
```json
{
  "Cohorts": [
    {
      "cohort_description": "<string>",
      "cohort_name": "<string or None>",
      "sample_size": "<string or None>",
      "mean_age": "<string or None>",
      "age_distribution": "<string or None>",
      "gender_distribution": "<string or None>",
      "racial_distribution": "<string or None>",
      "educational_attainment": "<string or None>",
      "employment_status": "<string or None>"
    }
  ]
}
```
If no cohort information is found:
```json
{
  "Cohorts": []
}
```

# EXAMPLE
--------------------
- Example tables:

### Table 1:

**Baseline characteristics of participants with and without incident dementia.**

| Incident dementia (n = 35,557) | No incident dementia (n = 369,259) |
<the remaining table content is omitted>

- Example response:

```json
{
  "Cohorts": [
    {
      "cohort_description": "Baseline characteristics of participants with incident dementia.",
      "cohort_name": "None",
      "sample_size": "35,557",
      "mean_age": "60 (SD = 15)",
      "age_distribution": "<50 (9844, 27%) | 50–54 (5615, 15%) | 55–59 (6212, 17%) | 60–64 (8146, 22%) | ≥65 (5740, 16%)",
      "gender_distribution": "Males (46%) | Females (53%)",
      "racial_distribution": "European (96%) | Non-European (4%)",
      "educational_attainment": "College or above (45%) | High school (38%) | Below high school (17%)",
      "employment_status": "Employed (62%) | Unemployed (18%) | Retired (20%)"
    },
    {
      "cohort_description": "Baseline characteristics of participants without incident dementia.",
      ...
    }
  ]
}
```

# TABLES
--------------------
Below are the tables to analyze. Extract baseline information according to the rules above:
```markdown
<<tables>>
```
"""