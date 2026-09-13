"""Where the disease-prediction stage finds its inputs.

The KG paths point at the artifacts produced by run_postprocess.sh, so this
module is the single place where the two halves of the repository meet. Change
a path here, or override it in the environment, rather than editing the
numbered scripts:

    UKB_KG_ROOT   save root used by the KG build      (default: save)
    UKB_KG_RUN    RUN_NAME of the KG build            (default: sample)
    UKB_DP_DATA   cohort data directory               (default:
                  disease_prediction/data)
    UKB_KGE_DIR   KGE toolkit working directory       (default: disease_prediction)
    UKB_BIOBERT   BioBERT checkpoint                  (default: model/biobert_v1.2)

disease_prediction/data ships the synthetic cohort so that the pipeline runs
out of the box; replace those arrays in place with the real UK Biobank ones,
keeping the file names and shapes, to reproduce the paper. See
disease_prediction/data/README.md.
"""

import os

# --------------------------------------------------------------- KG artifacts
KG_ROOT = os.environ.get("UKB_KG_ROOT", "save")
KG_RUN = os.environ.get("UKB_KG_RUN", "sample")
KG_DIR = os.path.join(KG_ROOT, KG_RUN)

# Final triples, output of run_postprocess.sh stage 'csv'.
TRIPLES_JSON = os.path.join(KG_DIR, "merged", "10_triples.json")

# Final edge list with confidence scores, output of stage 'attributes'.
TRIPLES_CSV = os.path.join(KG_DIR, "csv", "triples_with_attributes.csv")

# Derived here, in 01_relation_mapping.py / 03_get_entity_emb.py.
MAPPED_TRIPLES_JSON = TRIPLES_JSON.replace(".json", "_mapped_12r.json")
RELATION_MAPPING_JSON = TRIPLES_JSON.replace(".json", "_mapping_12r.json")
MAPPED_TRIPLES_TXT = MAPPED_TRIPLES_JSON.replace(".json", ".txt")
NODE_EMB_NPZ = os.path.join(KG_DIR, "csv", "node_emb.npz")

# ------------------------------------------------------------- cohort  data
DATA_DIR = os.environ.get("UKB_DP_DATA", "disease_prediction/data")
RAW_DIR = os.path.join(DATA_DIR, "data_raw")

PHECODES_NPY = os.path.join(DATA_DIR, "phecodes.npy")
PHECODE_DEF_CSV = os.path.join(DATA_DIR, "phecode_def_abbr.csv")
PHECODE_NODE_MAP_CSV = os.path.join(DATA_DIR, "phecode_node_map.csv")

# --------------------------------------------------------------- KGE toolkits
# The KGE trainers themselves are external; only their data/model directories
# are referenced here.
KGE_DIR = os.environ.get("UKB_KGE_DIR", "disease_prediction")
KGE_DATASET = os.environ.get("UKB_KGE_DATASET", "ukb_gpt-5-minimal_12r")
KGE_DATA_DIR = os.path.join(KGE_DIR, "KGE-HAKE", "data", KGE_DATASET)
ENTITIES_DICT = os.path.join(KGE_DATA_DIR, "entities.dict")
RELATIONS_DICT = os.path.join(KGE_DATA_DIR, "relations.dict")


def kge_entity_embedding(model_name):
    """Entity embedding matrix written by the KGE trainer for `model_name`."""
    return os.path.join(
        KGE_DIR, "KGE-RotatE", "models", f"{model_name}_{KGE_DATASET}", "entity_embedding.npy"
    )


def ene_output_dir(model_name):
    """Where 05_to_ene.py writes the per-subject KG-embedding features."""
    return os.path.join(DATA_DIR, f"data_gpt-5-minimal_12r_{model_name}")


# ------------------------------------------------------------------- models
BIOBERT_PATH = os.environ.get("UKB_BIOBERT", "model/biobert_v1.2")
