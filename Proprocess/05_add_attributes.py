"""Attach a per-edge confidence score derived from its source publication.

Confidence_Score is a weighted sum of four normalized signals about the article
a triple came from:

  S_DP   recency          exponential decay on publication year
  S_CT   citation count   log-normalized
  S_IF   journal impact   saturating exponential
  S_UKB  UK Biobank flag  1 if the paper is a registered UK Biobank study

Edges with no source article (the BIOS-derived ones) and articles missing from
the metadata CSV are assigned the median confidence of the scored edges.
"""

import argparse

import numpy as np
import pandas as pd

# Columns read from the publication metadata CSV. See the README for how it is
# assembled and which of the sources are licensed.
SOURCE_COLS = ["PMCID", "PMID", "DP", "TI", "TA", "KW", "URL", "JIF", "UKB", "CT"]

OUTPUT_COLS = [
    "HeadName",
    "HeadType",
    "HeadCID",
    "TailName",
    "TailType",
    "TailCID",
    "RelName",
    "RelType",
    "RELID",
    "PMCID",
    "Confidence_Score",
    "Frequency_Score",
]

NUMERIC_COLS = [
    "JIF",
    "UKB",
    "CT",
    "DP_y",
    "S_DP",
    "S_CT",
    "S_IF",
    "S_UKB",
    "Confidence_Score",
    "Frequency_Score",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True, help="Output of 04_json_to_csv.")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--source-file", required=True, help="Publication metadata CSV.")
    p.add_argument("--current-year", type=int, default=2025)
    p.add_argument("--decay-lambda", type=float, default=0.1)
    p.add_argument("--impact-alpha", type=float, default=0.1)
    p.add_argument("--w-dp", type=float, default=0.3)
    p.add_argument("--w-ct", type=float, default=0.3)
    p.add_argument("--w-if", type=float, default=0.3)
    p.add_argument("--w-ukb", type=float, default=0.1)
    return p.parse_args()


def check_pmcid_column(df, label):
    if not df["PMCID"].map(lambda x: isinstance(x, str)).all():
        raise AssertionError(f"{label} PMCID column contains non-string values!")
    if (df["PMCID"] != df["PMCID"].str.strip()).any():
        raise AssertionError(f"{label} PMCID contains leading/trailing spaces!")


def process_dp(dp_value):
    """Publication date -> year. The CSV stores dates as 'YYYY ...' strings."""
    if pd.isna(dp_value) or dp_value == "":
        return np.nan
    year = str(dp_value)[:4]
    return int(year) if year.isdigit() else np.nan


def load_source(args):
    source_df = pd.read_csv(args.source_file)

    source_df["PMID"] = source_df["PMID"].fillna(0).astype(int).astype(str)
    source_df["PMID"] = source_df["PMID"].replace("0", np.nan)
    source_df.rename(columns={"PMC": "PMCID"}, inplace=True)
    source_df = source_df[SOURCE_COLS]

    check_pmcid_column(source_df, "source_df")
    if source_df["PMCID"].duplicated().any():
        raise AssertionError("source_df contains duplicated PMCID rows!")

    return source_df


def score_source(source_df, args):
    source_df["DP_y"] = source_df["DP"].apply(process_dp)

    source_df["DP_y"] = source_df["DP_y"].fillna(source_df["DP_y"].median()).astype(int)
    source_df["CT"] = source_df["CT"].fillna(source_df["CT"].mean())
    source_df["JIF"] = source_df["JIF"].fillna(source_df["JIF"].mean())

    source_df["S_DP"] = np.exp(-args.decay_lambda * (args.current_year - source_df["DP_y"]))
    source_df["S_CT"] = np.log1p(source_df["CT"]) / np.log1p(source_df["CT"].max())
    source_df["S_IF"] = 1 - np.exp(-args.impact_alpha * source_df["JIF"])
    source_df["S_UKB"] = (source_df["UKB"] == 1).astype(int)

    source_df["Confidence_Score"] = (
        args.w_dp * source_df["S_DP"]
        + args.w_ct * source_df["S_CT"]
        + args.w_if * source_df["S_IF"]
        + args.w_ukb * source_df["S_UKB"]
    )

    print("Scoring Done!")
    return source_df.sort_values(by="Confidence_Score", ascending=False)


def main():
    args = parse_args()
    print(f"Input:  {args.input_csv}\nSource: {args.source_file}\nOutput: {args.output_csv}")

    triple_df = pd.read_csv(args.input_csv)
    check_pmcid_column(triple_df, "triple_df")

    source_df = score_source(load_source(args), args)

    merged = triple_df.merge(source_df, on="PMCID", how="left")

    # Unmatched rows (BIOS edges, and articles absent from the metadata CSV)
    # fall back to the median rather than staying unscored.
    unscored = merged["Confidence_Score"].isna().sum()
    median_conf = merged["Confidence_Score"].median()
    merged.loc[merged["Confidence_Score"].isna(), "Confidence_Score"] = median_conf
    print(f"Assigned median confidence {median_conf:.4f} to {unscored} unmatched rows")

    merged["RELID"] = merged["RELID"].astype(str)
    non_numeric_cols = [c for c in merged.columns if c not in NUMERIC_COLS]
    merged[non_numeric_cols] = merged[non_numeric_cols].replace(np.nan, "none").replace("", "none")

    merged[OUTPUT_COLS].to_csv(args.output_csv, index=False)
    print(f"✅ Done! Merged file saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
