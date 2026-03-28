import json
from collections import Counter, defaultdict
from entity_type_tag.entity_type import EntityTypeClassifier


def process_ukb(json_path, output_path, relabel_entity=False):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # === Step 1. Unify triples that differ only in case ===
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

    print(f"[Unify - UKB] Total of {len(replace_map)} entity variant groups requiring uniformity were found.")

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

    print("[Unify - UKB] 示例替换映射（前10个）:")
    for k, v in list(replace_map.items())[:10]:
        print(f"  {k} → {v}")

    # === Step 2. PMC internal triple deduplication ===
    total_triples_before = 0
    total_triples_after = 0
    deduped_data = []

    for data_point in updated_data:
        pmc, triples = next(iter(data_point.items()))
        total_triples_before += len(triples)

        seen = set()
        unique_triples = []
        for tri in triples:
            key = (
                tri["head"]["name"].strip(),
                tri["relation"]["name"].strip(),
                tri["tail"]["name"].strip(),
            )
            if key not in seen:
                seen.add(key)
                unique_triples.append(tri)
        total_triples_after += len(unique_triples)

        deduped_data.append({pmc: unique_triples})

    print(f"[Deduplication - UKB] Total PMC entries: {len(deduped_data)}")
    print(f"[Deduplication - UKB] Total triples before deduplication: {total_triples_before}")
    print(f"[Deduplication - UKB] Total triples after deduplication:  {total_triples_after}")
    print(f"[Deduplication - UKB] Total duplicates removed: {total_triples_before - total_triples_after}")

    # === Step 3. entity typing ===
    stats = defaultdict(set)

    if relabel_entity:
        clf = EntityTypeClassifier(model_dir="entity_type_tag/save/save_35w_256")

        entity_set = set()
        for paper in deduped_data:
            for _, triples in paper.items():
                for tri in triples:
                    entity_set.add(tri["head"]["name"])
                    entity_set.add(tri["tail"]["name"])

        entity_list = list(entity_set)
        entity2type = clf.predict_dict(entity_list)

        for paper in deduped_data:
            for _, triples in paper.items():
                for tri in triples:
                    h = tri["head"]["name"]
                    t = tri["tail"]["name"]
                    tri["head"]["type"] = entity2type[h]
                    tri["tail"]["type"] = entity2type[t]
                    tri["relation"]["type"] = f"{entity2type[h]}-{entity2type[t]}"

                    stats[entity2type[h]].add(h)
                    stats[entity2type[t]].add(t)

        print("[Typing - UKB] === Entity count per class for UKB Triples ===")
        for k, v in stats.items():
            print(k, len(v))

        entity_to_types = defaultdict(set)
        for ent, typ in entity2type.items():
            entity_to_types[ent].add(typ)

        conflicts = {e: t for e, t in entity_to_types.items() if len(t) > 1}
        print(f"[Typing - UKB] Entities with multiple types: {len(conflicts)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduped_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Save updated file to {output_path}")
        

def process_completion(input_path, output_path):
    clf = EntityTypeClassifier(model_dir="entity_type_tag/save/save_35w_256")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # === Step 1. Triple deduplication ===
    total_triples_before = 0
    total_triples_after = 0
    deduped_data = []

    for data_point in data:
        pmc, triples = next(iter(data_point.items()))
        total_triples_before += len(triples)

        seen = set()
        unique_triples = []
        for tri in triples:
            key = (
                tri["head"]["name"].strip(),
                tri["relation"]["name"].strip(),
                tri["tail"]["name"].strip(),
            )
            if key not in seen:
                seen.add(key)
                unique_triples.append(tri)
        total_triples_after += len(unique_triples)

        deduped_data.append({pmc: unique_triples})

    print(f"[Deduplication - Completion] Total PMC entries: {len(deduped_data)}")
    print(f"[Deduplication - Completion] Total triples before deduplication: {total_triples_before}")
    print(f"[Deduplication - Completion] Total triples after deduplication:  {total_triples_after}")
    print(f"[Deduplication - Completion] Total duplicates removed: {total_triples_before - total_triples_after}")

    # === Step 2. Entity typing ===
    stats = defaultdict(set)

    for paper in deduped_data:
        for _, triples in paper.items():
            entity_set = set()
            for tri in triples:
                entity_set.add(tri["head"]["name"])
                entity_set.add(tri["tail"]["name"])

            entity_list = list(entity_set)
            entity2type = clf.predict_dict(entity_list)

            for tri in triples:
                h = tri["head"]["name"]
                t = tri["tail"]["name"]
                tri["head"]["type"] = entity2type[h]
                tri["tail"]["type"] = entity2type[t]
                tri["relation"]["type"] = f"{entity2type[h]}-{entity2type[t]}"

                stats[entity2type[h]].add(h)
                stats[entity2type[t]].add(t)

    print("[Typing - Completion] === Entity count per class for Completion Triples===")
    for k, v in stats.items():
        print(k, len(v))

    entity_to_types = defaultdict(set)
    for ent, typ in entity2type.items():
        entity_to_types[ent].add(typ)

    conflicts = {e: t for e, t in entity_to_types.items() if len(t) > 1}
    print(f"[Typing - Completion] Entities with multiple types: {len(conflicts)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduped_data, f, ensure_ascii=False, indent=2)


ukb_input_path = 'save/save_sample/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98.json'
ukb_output_path = ukb_input_path.replace(".json", "_processed.json")
completion_input_path = 'save/save_bios/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_completion.json'
completion_output_path = completion_input_path.replace(".json", "_typed.json")
# ukb_input_path = 'save/save_sample/triple/triples_gpt-5_minimal_wo_ner_filter1_refine_filter2_gpt-5verified_umlslinked98.json'
# ukb_output_path = ukb_input_path.replace(".json", "_processed.json")

# for ukb triples
process_ukb(ukb_input_path, ukb_output_path, relabel_entity=True)
# for completion triples
process_completion(completion_input_path, completion_output_path)
