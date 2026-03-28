import pandas as pd
import numpy as np


save_type = 'neo4j'
triple_csv = "save/save_sample/csv/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_processed.csv"
source_file = "PMC_Data/pubmed-ukb.csv"
output_file = triple_csv.replace('.csv', '_neo4j.csv')
triple_df = pd.read_csv(triple_csv)
source_df = pd.read_csv(source_file)

# -----------------------------
# Preprocess
# -----------------------------
source_df['PMID'] = source_df['PMID'].fillna(0).astype(int).astype(str)
source_df['PMID'] = source_df['PMID'].replace('0', np.nan)
source_df.rename(columns={'PMC': 'PMCID'}, inplace=True)
attribute_cols = ['PMCID', 'PMID', 'DP', 'TI', 'TA', 'KW', 'URL', 'JIF', 'UKB', 'CT']
cols_from_source = ['PMID', 'DP', 'TI', 'TA', 'KW', 'URL', 'JIF', 'UKB', 'CT']
source_df = source_df[attribute_cols]

# check PMCID formatting
if not triple_df['PMCID'].map(lambda x: isinstance(x, str)).all():
    raise AssertionError("triple_df PMCID column contains non-string values!")

if (triple_df['PMCID'] != triple_df['PMCID'].str.strip()).any():
    raise AssertionError("triple_df PMCID contains leading/trailing spaces!")

if not source_df['PMCID'].map(lambda x: isinstance(x, str)).all():
    raise AssertionError("source_df PMCID column contains non-string values!")

if (source_df['PMCID'] != source_df['PMCID'].str.strip()).any():
    raise AssertionError("source_df PMCID contains leading/trailing spaces!")

if source_df['PMCID'].duplicated().any():
    raise AssertionError("source_df contains duplicated PMCID rows!")

# -----------------------------
# Confidence Score
# -----------------------------
# Process DP → extract year
def process_dp(dp_value):
    if pd.isna(dp_value) or dp_value == '':
        return np.nan
    year = dp_value[:4]
    return int(year) if year.isdigit() else np.nan

source_df['DP_y'] = source_df['DP'].apply(process_dp)

# Fill missing values
source_df['DP_y'] = source_df['DP_y'].fillna(source_df['DP_y'].median()).astype(int)
source_df['CT'] = source_df['CT'].fillna(source_df['CT'].mean())
source_df['JIF'] = source_df['JIF'].fillna(source_df['JIF'].mean())

# Scoring functions
def exponential_decay(t, T, lambda_):
    return np.exp(-lambda_ * (T - t))

def logarithmic_scaling(c, c_max):
    return np.log1p(c) / np.log1p(c_max)

def exponential_decay_impact(i, alpha):
    return 1 - np.exp(-alpha * i)

def calculate_ukb_score(v):
    return int(v == 1)

# Parameters
current_year = 2025
lambda_ = 0.1
alpha = 0.1

w_dp = 0.3
w_ct = 0.3
w_if = 0.3
w_ukb = 0.1

# Compute Scores (vectorized)
source_df['S_DP'] = exponential_decay(source_df['DP_y'], current_year, lambda_)
source_df['S_CT'] = logarithmic_scaling(source_df['CT'], source_df['CT'].max())
source_df['S_IF'] = exponential_decay_impact(source_df['JIF'], alpha)
source_df['S_UKB'] = source_df['UKB'].apply(calculate_ukb_score)

# Confidence score
source_df['Confidence_Score'] = (
        w_dp * source_df['S_DP'] +
        w_ct * source_df['S_CT'] +
        w_if * source_df['S_IF'] +
        w_ukb * source_df['S_UKB']
)

source_df = source_df.sort_values(by='Confidence_Score', ascending=False)
print("Scoring Done!")

# -----------------------------
# Merge and process
# -----------------------------
merged = triple_df.merge(source_df, on='PMCID', how='left')

median_conf = merged.loc[merged['PMCID'] != 'none', 'Confidence_Score'].median()
merged.loc[merged['PMCID'] == 'none', 'Confidence_Score'] = median_conf

merged['RELID'] = merged['RELID'].astype(str)
numeric_cols = ['JIF', 'UKB', 'CT', 'DP_y', 'S_DP', 'S_CT', 'S_IF', 'S_UKB', 'Confidence_Score', 'Frequency_Score']
non_numeric_cols = [c for c in merged.columns if c not in numeric_cols]
merged[non_numeric_cols] = (
    merged[non_numeric_cols]
        .replace(np.nan, "none")
        .replace("", "none")
)

merged = merged[['HeadName', 'HeadType', 'HeadCID', 'TailName', 'TailType', 'TailCID', 'RelName', 'RelType',
              'RELID', 'PMCID', 'Confidence_Score', 'Frequency_Score']]
merged.to_csv(output_file, index=False)

print(f"✅ Done! Merged file saved to: {output_file}")