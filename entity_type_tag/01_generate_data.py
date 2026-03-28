import random
from collections import defaultdict, Counter
from tqdm import tqdm

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 1. Config
MRSTY_PATH = "entity_type_tag/umls/MRSTY.RRF"
MRCONSO_PATH = "entity_type_tag/umls/MRCONSO.RRF"
SEMGROUP_PATH = "entity_type_tag/umls/SemGroups_UKB.txt"

OUT_DIR = "entity_type_tag/data"
TRAIN_PATH = f"{OUT_DIR}/train_data.txt"
EVAL_PATH = f"{OUT_DIR}/eval_data.txt"
TEST_PATH = f"{OUT_DIR}/test_data.txt"
ALL_PATH = f"{OUT_DIR}/all_data.txt"

N_SAMPLE_PER_GROUP = 350000

PRINT_TRUE_MULTI_GROUP_EXAMPLES = True
N_PRINT_TRUE_MULTI_GROUP = 10

CHECK_PREFERRED_PF = True
PRINT_PREFERRED_ISSUES = True
N_PRINT_PREFERRED_ISSUES = 10


# 2. Read TUI → Semantic Group
tui_to_group = {}
groups = set()

with open(SEMGROUP_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        group, _, tui, _ = line.strip().split("|")
        tui_to_group[tui] = group
        groups.add(group)

print(f"[INFO] Semantic Groups: {sorted(groups)}")


# 3. CUI → Semantic Group (from MRSTY)
cui_to_groups = defaultdict(list)

with open(MRSTY_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading MRSTY"):
        fields = line.rstrip("\n").split("|")
        if len(fields) < 2:
            continue
        cui, tui = fields[0], fields[1]
        if tui in tui_to_group:
            cui_to_groups[cui].append(tui_to_group[tui])

cui_to_group = {}
true_multi_group_cuis = []
multi_tui_single_group = 0

for cui, gs in cui_to_groups.items():
    uniq = sorted(set(gs))
    if len(uniq) == 1:
        cui_to_group[cui] = uniq[0]
        if len(gs) > 1:
            multi_tui_single_group += 1
    else:
        true_multi_group_cuis.append((cui, uniq))

print("=" * 60)
print("[CUI GROUP ASSIGNMENT]")
print(f"Total CUIs with mapped TUI: {len(cui_to_groups)}")
print(f"Single-group CUIs kept       : {len(cui_to_group)}")
print(f"multi-TUI but single group   : {multi_tui_single_group}")
print(f"TRUE multi-group CUIs removed: {len(true_multi_group_cuis)}")
print("=" * 60)

if PRINT_TRUE_MULTI_GROUP_EXAMPLES:
    print("[TRUE MULTI-GROUP EXAMPLES] (removed)")
    for cui, uniq in true_multi_group_cuis[:N_PRINT_TRUE_MULTI_GROUP]:
        print(f"  {cui} -> {uniq}")
    print("=" * 60)


# 4. CUI → entity names (MRCONSO)
IDX_CUI = 0
IDX_LAT = 1
IDX_TS = 2
IDX_LUI = 3
IDX_STT = 4
IDX_STR = 14
IDX_SUPPRESS = 16

all_valid_cuis = set(cui_to_group.keys())

cui_to_names = defaultdict(set)             # CUI -> {entity strings}
cui_to_pref_pf = defaultdict(set)           # CUI -> {preferred entity candidates (TS=P & STT=PF)}

with open(MRCONSO_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading MRCONSO"):
        fields = line.rstrip("\n").split("|")
        if len(fields) <= IDX_STR:
            continue

        cui = fields[IDX_CUI]
        if cui not in all_valid_cuis:
            continue

        lat = fields[IDX_LAT]
        suppress = fields[IDX_SUPPRESS] if len(fields) > IDX_SUPPRESS else "N"
        if lat != "ENG" or suppress == "Y":
            continue

        s = fields[IDX_STR]
        cui_to_names[cui].add(s)

        ts = fields[IDX_TS] if len(fields) > IDX_TS else ""
        stt = fields[IDX_STT] if len(fields) > IDX_STT else ""
        if ts == "P" and stt == "PF":
            cui_to_pref_pf[cui].add(s)

# Check if preferred(PF, TS=P) is unique for each CUI.
missing_pref = []
multi_pref = []
ok_pref = 0

for cui in list(all_valid_cuis):
    prefs = cui_to_pref_pf.get(cui, set())
    if len(prefs) == 0:
        missing_pref.append(cui)
    elif len(prefs) > 1:
        multi_pref.append((cui, sorted(prefs)))
    else:
        ok_pref += 1

print("=" * 60)
print("[PREFERRED ENTITY CHECK] (TS=P & STT=PF, after ENG+SUPPRESS filtering)")
print(f"CUIs total considered : {len(all_valid_cuis)}")
print(f"CUIs ok (exactly 1)   : {ok_pref}")
print(f"CUIs missing preferred: {len(missing_pref)}")
print(f"CUIs multi preferred  : {len(multi_pref)}")
print("=" * 60)

if CHECK_PREFERRED_PF and PRINT_PREFERRED_ISSUES:
    if missing_pref:
        print("[EXAMPLES] Missing preferred (first few):")
        for cui in missing_pref[:N_PRINT_PREFERRED_ISSUES]:
            print(f"  {cui} (group={cui_to_group.get(cui)})")
        print("-" * 60)
    if multi_pref:
        print("[EXAMPLES] Multiple preferred (first few):")
        for cui, prefs in multi_pref[:N_PRINT_PREFERRED_ISSUES]:
            print(f"  {cui} (group={cui_to_group.get(cui)}): {prefs}")
        print("-" * 60)

bad_cuis = set(missing_pref) | {cui for cui, _ in multi_pref}
kept_cuis_after_pref_check = all_valid_cuis - bad_cuis

print(f"[INFO] Removing CUIs failing preferred check: {len(bad_cuis)}")
print(f"[INFO] CUIs kept after preferred check      : {len(kept_cuis_after_pref_check)}")

# Clean group mapping
cui_to_group = {cui: g for cui, g in cui_to_group.items() if cui in kept_cuis_after_pref_check}

cuis_with_names = {cui for cui in kept_cuis_after_pref_check if cui_to_names.get(cui)}
removed_no_names = kept_cuis_after_pref_check - cuis_with_names
if removed_no_names:
    print(f"[INFO] Removing CUIs with no ENG+unsuppressed names: {len(removed_no_names)}")

kept_cuis_after_pref_check = cuis_with_names
cui_to_group = {cui: g for cui, g in cui_to_group.items() if cui in kept_cuis_after_pref_check}

# Flatten the preferred entity (unique) into a dict: cui -> preferred_str
cui_to_pref = {}
for cui in kept_cuis_after_pref_check:
    pref_set = cui_to_pref_pf[cui]
    cui_to_pref[cui] = next(iter(pref_set))


# 5. N_SAMPLE_PER_GROUP CUIs are randomly sampled for each group.
group_to_cuis = defaultdict(list)
for cui, g in cui_to_group.items():
    group_to_cuis[g].append(cui)

sampled_group_to_cuis = {}
for g, cuis in group_to_cuis.items():
    if len(cuis) <= N_SAMPLE_PER_GROUP:
        sampled_group_to_cuis[g] = set(cuis)
        print(f"[WARN] {g} has only {len(cuis)} CUIs (<{N_SAMPLE_PER_GROUP})")
    else:
        sampled_group_to_cuis[g] = set(random.sample(cuis, N_SAMPLE_PER_GROUP))

print("[INFO] Finished sampling CUIs per group")

final_sampled_cuis = set().union(*sampled_group_to_cuis.values())


# 6. Hendle entities that differ only in capitalization
lower_to_records = defaultdict(list)

for g, cuis in sampled_group_to_cuis.items():
    for cui in cuis:
        pref = cui_to_pref[cui]
        for ent in cui_to_names[cui]:
            lower_to_records[ent.lower()].append((g, cui, ent, ent == pref))

case_variant_keys = []
for k, recs in lower_to_records.items():
    forms = {r[2] for r in recs}  # original case forms
    if len(forms) > 1:
        case_variant_keys.append(k)

# Determine which values are ultimately retained for each lower_key.
keep_pairs = set()      # (cui, entity)
deleted_keys = 0
kept_keys = 0

for k, recs in lower_to_records.items():
    forms = {r[2] for r in recs}
    if len(forms) == 1:
        for _, cui, ent, _ in recs:
            keep_pairs.add((cui, ent))

for k in case_variant_keys:
    recs = lower_to_records[k]

    preferred_recs = [r for r in recs if r[3] is True]  # is_preferred
    if preferred_recs:
        candidates = preferred_recs

        groups_here = {r[0] for r in candidates}
        if len(candidates) > 1:
            if len(groups_here) > 1:
                deleted_keys += 1
                continue
            chosen = random.choice(candidates)
            keep_pairs.add((chosen[1], chosen[2]))
            kept_keys += 1
        else:
            chosen = candidates[0]
            keep_pairs.add((chosen[1], chosen[2]))
            kept_keys += 1
    else:
        groups_here = {r[0] for r in recs}
        if len(groups_here) > 1:
            deleted_keys += 1
            continue
        chosen = random.choice(recs)
        keep_pairs.add((chosen[1], chosen[2]))
        kept_keys += 1

print("=" * 60)
print("[CASE-VARIANT ENTITY STATS]")
print(f"Total case-variant keys (lowercase groups) : {len(case_variant_keys)}")
print(f"Keys fully deleted due to cross-group      : {deleted_keys}")
print(f"Keys kept (at least 1 entity preserved)    : {kept_keys}")
print("=" * 60)


# 7. CUI-level split
def split_cuis(cuis):
    cuis = list(cuis)
    random.shuffle(cuis)
    n = len(cuis)
    n_train = int(n * 0.9)
    n_eval = int(n * 0.05)
    train = set(cuis[:n_train])
    eval_ = set(cuis[n_train:n_train + n_eval])
    test = set(cuis[n_train + n_eval:])
    return train, eval_, test

train_lines, eval_lines, test_lines, all_lines = [], [], [], []

final_group_to_cuis_after_case = {}

for g, cuis in sampled_group_to_cuis.items():
    kept_cuis = []
    for cui in cuis:
        kept_entities = [e for e in cui_to_names[cui] if (cui, e) in keep_pairs]
        if kept_entities:
            kept_cuis.append(cui)
    final_group_to_cuis_after_case[g] = set(kept_cuis)

dropped_empty_cui = sum(len(sampled_group_to_cuis[g]) - len(final_group_to_cuis_after_case[g]) for g in sampled_group_to_cuis)
if dropped_empty_cui > 0:
    print(f"[INFO] Dropped CUIs with 0 entities after case-variant filtering: {dropped_empty_cui}")

for g, cuis in final_group_to_cuis_after_case.items():
    tr, ev, te = split_cuis(cuis)

    # train
    for cui in tr:
        for e in cui_to_names[cui]:
            if (cui, e) in keep_pairs:
                train_lines.append(f"{g}|{e}")

    # eval
    for cui in ev:
        for e in cui_to_names[cui]:
            if (cui, e) in keep_pairs:
                eval_lines.append(f"{g}|{e}")

    # test
    for cui in te:
        for e in cui_to_names[cui]:
            if (cui, e) in keep_pairs:
                test_lines.append(f"{g}|{e}")

    # all
    for cui in cuis:
        for e in cui_to_names[cui]:
            if (cui, e) in keep_pairs:
                all_lines.append(f"{g}|{e}")


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

write_lines(TRAIN_PATH, train_lines)
write_lines(EVAL_PATH, eval_lines)
write_lines(TEST_PATH, test_lines)
write_lines(ALL_PATH, all_lines)

print("[DONE]")
print(f"  Train: {len(train_lines)}")
print(f"  Eval : {len(eval_lines)}")
print(f"  Test : {len(test_lines)}")
print(f"  All  : {len(all_lines)}")