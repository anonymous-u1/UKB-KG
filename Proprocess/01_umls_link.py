import json
from utils.umls import umls_map

input_path = 'save/save_sample/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified.json'
output_path = input_path.replace('.json', '_umlslinked98.json')

umls_threshold = 0.98
umls_mapper = umls_map(threshold=umls_threshold)


def safe_get_str(d: dict, *keys) -> str:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return ""
        d = d[k]
    return str(d).strip() if d is not None else ""


if __name__ == "__main__":
    with open(input_path, 'r', encoding='utf-8') as f:
        data_points = json.load(f)

    linked_data = []
    total_papers = len(data_points)
    total_triples = linked_entities = 0

    cache = {}

    print_examples = 50
    example_count = 0

    for data_point in data_points:
        key, triples = next(iter(data_point.items()))
        updated_triples = []

        for triple in triples:
            head = safe_get_str(triple, 'head', 'name')
            tail = safe_get_str(triple, 'tail', 'name')
            relation = safe_get_str(triple, 'relation', 'name')
            if not head or not tail or not relation:
                print(f"❌ Can not find head/tail/relation in {key}")

            if head not in cache:
                cache[head] = umls_mapper(head)
            if tail not in cache:
                cache[tail] = umls_mapper(tail)

            head_sim = cache[head]
            tail_sim = cache[tail]

            if example_count < print_examples and (head_sim != head or tail_sim != tail):
                print(f"{head} → Linked: {head_sim}")
                print(f"{tail} → Linked: {tail_sim}")
                example_count += 1

            if head_sim != head:
                triple['head']['name'] = head_sim
                linked_entities += 1
            if tail_sim != tail:
                triple['tail']['name'] = tail_sim
                linked_entities += 1

            updated_triples.append(triple)
            total_triples += 1

        linked_data.append({key: updated_triples})

    if total_papers != len(linked_data):
        print("❌ The number of papers before and after the linking are different.")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(linked_data, f, indent=2, ensure_ascii=False)

    print(f"=== Linking Summary ===")
    print(f"Total papers:  {total_papers:,}")
    print(f"Total triples: {total_triples:,}")
    print(f"Linked entities: {linked_entities:,}")
    print(f"Saved linked file: {output_path}")