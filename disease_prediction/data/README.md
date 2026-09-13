## Synthetic Demo Data for the Disease-Prediction Experiments

The disease-prediction experiments in the paper are run on individual-level UK Biobank cohort data. UK Biobank data are released to approved researchers under a Material Transfer Agreement and cannot be redistributed, so the real cohort arrays are not part of this repository.

To keep the code inspectable and runnable end to end, this directory ships a small **synthetic** cohort with the same file layout, array shapes, dtypes and value semantics as the real inputs. It is intended to let a reader follow the code workflow and check that the pipeline executes and reproduces its own outputs. It is **not** a statistical surrogate for UK Biobank, and no result computed from it carries scientific meaning.

The script used to generate this synthetic data reads no UK Biobank file, and no real record, summary statistic or fitted parameter is used anywhere in it.

Approved researchers reproducing the paper should replace the arrays in this directory with the real cohort ones, keeping the file names and shapes below; every script and default path in `disease_prediction/` then works unchanged. To keep the two side by side instead, put the real arrays elsewhere and set `UKB_DP_DATA` to that directory (see `disease_prediction/config.py`).

### Files

`N` is the number of synthetic subjects; the column spaces are the real ones.


| File                                                             | Shape       | dtype     | Content                                                                               |
| ---------------------------------------------------------------- | ----------- | --------- | ------------------------------------------------------------------------------------- |
| `data_raw/features_{train,val,test}.npy`                         | `(N, 1560)` | `float64` | Baseline diagnosis history: `1` if the subject already has the Phecode                |
| `data_raw/labels_{train,val,test}.npy`                           | `(N, 1560)` | `float64` | Incident diagnoses to be predicted: `1` if the subject later develops the Phecode     |
| `data_gpt-5-minimal_12r_TransE/avr_ene_12r_{train,val,test}.npy` | `(N, 1000)` | `float32` | KG-embedding features: the mean TransE embedding of the subject's historical Phecodes |
| `synthetic_data_summary.json`                                    | –           | –         | Generation parameters and per-split summary counts                                    |


Column `j` of the `features` and `labels` matrices corresponds to Phecode `phecodes.npy[j]`; `phecode_def_abbr.csv` gives the Phecode descriptions. Both matrices are binary and share the same column space, so `features` and `labels` have identical shapes. The embedding dimension of `1000` is the `hidden_dim` of the KGE runs.

Split sizes are `train = 500`, `val = 200`, `test = 20`.

### Fidelity to the real inputs

Matched by construction: file names and directory layout, number of Phecode columns, embedding dimension, dtypes, binary `{0, 1}` encoding, and the identity `features.shape == labels.shape`.


Deliberately **not** matched:

- **Cohort size.** 900 subjects in total.
- **Incident-code density.**
- **Per-Phecode prevalences.** These are drawn from a parametric prior, not taken from the cohort. 



### Running the pipeline on this data

Training, from the repository root:

```bash
python -u -m disease_prediction.train \
    -s save/demo_transe_ene \
    --data_type ene \
    --train_ene_path disease_prediction/data/data_gpt-5-minimal_12r_TransE/avr_ene_12r_train.npy \
    --val_ene_path   disease_prediction/data/data_gpt-5-minimal_12r_TransE/avr_ene_12r_val.npy \
    --train_label_path disease_prediction/data/data_raw/labels_train.npy \
    --val_label_path   disease_prediction/data/data_raw/labels_val.npy \
    -hs 150 -bs 128 -lr 0.0001
```

Evaluation:

```bash
python -u -m disease_prediction.test \
    --save_dir save/demo_transe_ene \
    --test_name test_bin_2 \
    --data_type ene \
    --val_ene_path  disease_prediction/data/data_gpt-5-minimal_12r_TransE/avr_ene_12r_val.npy \
    --test_ene_path disease_prediction/data/data_gpt-5-minimal_12r_TransE/avr_ene_12r_test.npy \
    --train_label_path disease_prediction/data/data_raw/labels_train.npy \
    --val_label_path   disease_prediction/data/data_raw/labels_val.npy \
    --test_label_path  disease_prediction/data/data_raw/labels_test.npy \
    -hs 150 -bs 128 --bin_choice 2 \
    --sample_idx_dir disease_prediction/data/data_raw \
    --min_val_pos 10 --pooled_mode rate --seed 1234
```

Swap `--data_type ene` and the `--*_ene_path` arguments for `--data_type binary` and `--*_feature_path` pointing at `data_raw/features_*.npy` to run the binary baseline. Both complete in well under a minute on a single GPU, train.py stopping early at around epoch 180.

`test.py` writes per-label metrics to `metrics.csv`, prevalence-binned means to `mean.csv`, and the ROC/PR curves and sampled prediction matrices as `.npy` files. The 1:10 patient sampling indices are built on first use, cached in `--sample_idx_dir`, and reused by later runs so that all models are compared on the same sampled subjects.

### Limitations to be aware of

- **The metrics are meaningless.** With 500 training subjects and a 5-layer network over 1,000–1,560 input dimensions, both the absolute values and the ranking of the feature types are noise. Do not read the demo `mean.csv` as evidence for or against the paper's conclusions.
- **The two high-prevalence bins are empty.** `test.py --bin_choice 2` bins labels by their absolute positive count in the training split, with edges at 10, 100 and 500. On 500 subjects only the `0-10` and `10-100` bins are populated and `mean.csv` reports `NaN` for `100-500` and `500+`.