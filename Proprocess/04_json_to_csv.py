import json
import re
import csv
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


def load_json_data(path1, path2, merged_path, save_merge=True):
    print(f"\n[Step 1] Loading JSON triples from:")
    print(f"  - {path1}")
    print(f"  - {path2}")

    data = []
    for path in [path1, path2]:
        if path != "None":
            with open(path, 'r', encoding="utf-8") as f:
                data.extend(json.load(f))

    print(f"[Step 1] Total triple files merged: {len(data)}")

    print(f"\n[Step 2] Unify triples that differ only in case")

    all_entities = []
    for paper in data:
        for pmc, triples in paper.items():
            for tri in triples:
                all_entities.append(tri["head"]["name"].strip())
                all_entities.append(tri["tail"]["name"].strip())

    counter = Counter(all_entities)

    group_dict = defaultdict(list)
    for ent in counter.keys():
        group_dict[ent.lower()].append(ent)

    replace_map = {}
    for lower_form, variants in group_dict.items():
        if len(variants) == 1:
            continue

        best_form = max(variants, key=lambda x: counter[x])
        for v in variants:
            replace_map[v] = best_form

    print(f"[Step 2] Total of {len(replace_map)} entity variant groups requiring uniformity were found.")

    updated_data = []

    for paper in data:
        updated_paper = {}
        for pmc, triples in paper.items():
            updated_triples = []
            for tri in triples:
                h = tri["head"]["name"].strip()
                t = tri["tail"]["name"].strip()
                r = tri["relation"]["name"].strip()
                tri["head"]["name"] = replace_map.get(h, h)
                tri["tail"]["name"] = replace_map.get(t, t)
                tri["relation"]["name"] = r
                updated_triples.append(tri)
            updated_paper[pmc] = updated_triples
        updated_data.append(updated_paper)

    print("[Step 2] Example replacement mappings (first 10):")
    for k, v in list(replace_map.items())[:10]:
        print(f"  {k} → {v}")

    if save_merge:
        with open(merged_path, 'w') as f:
            json.dump(updated_data, f, indent=2, ensure_ascii=False)
    print(f"[Step 2] Merged JSON saved to: {merged_path}")
    return updated_data


def build_row(triple, pmcid):
    head, tail, relation = triple["head"], triple["tail"], triple["relation"]

    head_name = head['name']
    head_type = head['type']
    head_cid = head.get('CID', 'none')
    tail_name = tail['name']
    tail_type = tail['type']
    tail_cid = tail.get('CID', 'none')
    relation_name = relation['name'].replace(' ', '_').replace('-', '_')
    relation_type = relation['type']
    relation_rid = relation.get('RELID', 'none')

    row = {
        'HeadName': head_name,
        'HeadType': head_type,
        'HeadCID': head_cid,
        'TailName': tail_name,
        'TailType': tail_type,
        'TailCID': tail_cid,
        'RelName': relation_name,
        'RelType': relation_type,
        'RELID': relation_rid,
        'PMCID': pmcid[:-4] if pmcid[-4:] == '.xml' else 'none',
    }

    return row


def triples_to_csv(data, csv_path):
    print(f"\n[Step 3] Converting triples to CSV: {csv_path}")

    fieldnames = ['HeadName', 'HeadType', 'HeadCID', 'TailName', 'TailType', 'TailCID', 'RelName', 'RelType',
                  'RELID', 'PMCID']

    count = 0
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for paper in data:
            for pmc, triples in paper.items():
                for tri in triples:
                    row = build_row(tri, pmc)
                    csv_writer.writerow(row)
                    count += 1

    print(f"[Step 3] CSV writing done. Total rows: {count}")

def process_csv(path):
    print(f"\n[Step 4] Process CSV: {path}")
    df = pd.read_csv(path)
    print(f"[Step 4] Loaded {len(df)} rows")

    print(f"[Step 4] Resolving type conflicts ...")
    entity_type_counter = defaultdict(Counter)

    for idx, row in df.iterrows():
        for prefix in ['Head', 'Tail']:
            name_key = f'{prefix}Name'
            type_key = f'{prefix}Type'
            name = row[name_key]
            t = row[type_key]
            entity_type_counter[name][t] += 1

    conflict_count = sum(1 for cnt in entity_type_counter.values() if len(cnt) > 1)
    print(f"[Step 4] Found {conflict_count} conflicting entities (will be unified)")

    name_to_major_type = {
        name: counter.most_common(1)[0][0]
        for name, counter in entity_type_counter.items()
    }

    for idx, row in df.iterrows():
        for prefix in ['Head', 'Tail']:
            name_key = f'{prefix}Name'
            type_key = f'{prefix}Type'
            name = row[name_key]
            df.at[idx, type_key] = name_to_major_type[name]

    df['Frequency_Score'] = df.groupby(['HeadName', 'RelName', 'TailName'])['HeadName'].transform('count')
    c_max = df['Frequency_Score'].max()
    df['Frequency_Score'] = np.log1p(df['Frequency_Score']) / np.log1p(c_max)
    
    before = len(df)
    df_unique = df.drop_duplicates(subset=['HeadName', 'RelName', 'TailName'])
    after = len(df_unique)
    print(f"[Step 4] Removed duplicates: {before - after} rows")

    df_unique.to_csv(path, index=False)
    print(f"[Step 4] Clean CSV saved to: {path}")
    print("[Done ✅] All processing finished.\n")

triple_path1 = 'save/save_sample/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_processed.json'
triple_path2 = 'save/save_bios/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_completion_typed.json'
merged_path = 'save/save_sample/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_bioscompleted.json'
csv_save_path = 'save/csv/all_triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_bioscompleted.csv'
# triple_path1 = 'save/save_ablation/triple/triples_gpt-5_minimal_wo_ner_filter1_refine_filter2_gpt-5verified_umlslinked98_processed.json'
# triple_path2 = 'None'
# merged_path = 'None'
# csv_save_path = 'save/save_ablation/csv/ablation_triples_gpt-5_minimal_wo_ner_filter1_refine_filter2_gpt-5verified_umlslinked98_processed.csv'

triples = load_json_data(triple_path1, triple_path2, merged_path, save_merge=True)
triples_to_csv(triples, csv_save_path)
process_csv(csv_save_path)