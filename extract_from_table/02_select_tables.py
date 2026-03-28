import sys
import io
import re
import json
import argparse
import tiktoken
import warnings
import os
from pydantic import BaseModel
from llm.openai_chat import *
import prompts

warnings.filterwarnings("ignore", category=FutureWarning)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdout.reconfigure(line_buffering=True)
encoder = tiktoken.encoding_for_model('gpt-4o')


def args_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder_dir', type=str, required=True, help='Path to the tables.')
    parser.add_argument('-c', '--chunk', type=int, required=True)
    parser.add_argument('-n', '--number_of_chunk', type=int, required=True)
    parser.add_argument('--chunk_size', type=int, required=True)
    parser.add_argument('--llm', type=str, required=True)
    parser.add_argument('--effort', type=str, required=True)
    parser.add_argument('--rel_table_save_dir', type=str, required=True, help='Path to save the relation tables.')
    parser.add_argument('--baseline_table_save_dir', type=str, required=True, help='Path to save the baseline tables.')
    parser.add_argument('--result_save_path', type=str, required=True, help='Path to save the selected results.')
    parser.add_argument('--select_table_prompt', type=str, required=True)
    args = parser.parse_args()
    return args


class TableSelectionForm(BaseModel):
    Relation_Tables: list[str]
    Baseline_Tables: list[str]


def extract_tables_by_ids(txt_content, selected_ids):
    tables = re.split(r"(?=^##+\s*Table\s+\d+:)", txt_content, flags=re.MULTILINE)
    if not tables:
        return []

    extracted = []
    for tid in selected_ids:
        match = re.search(r"(\d+)", tid)
        if not match:
            continue
        num = match.group(1)
        for block in tables:
            if re.search(rf"^##+\s*Table\s+{num}\b", block, flags=re.MULTILINE):
                extracted.append(block.strip())
                break
    return extracted


def save_json_result(result_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data.append(result_dict)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def select_tables(file_names, folder_path, rel_table_save_dir, baseline_table_save_dir, result_save_path):
    os.makedirs(rel_table_save_dir, exist_ok=True)
    os.makedirs(baseline_table_save_dir, exist_ok=True)

    if not os.path.isfile(result_save_path):
        with open(result_save_path, 'w', encoding='utf-8') as f:
            f.write('[]')
        print(f"create {result_save_path}.\n")
        file_names_saved = []
    else:
        with open(result_save_path, 'r', encoding='utf-8') as f:
            files = json.load(f)
            file_names_saved = [list(d.keys())[0] for d in files]

    print(f"✅ Loaded {len(file_names_saved)} processed files, {len(file_names)} total.")

    valid_files = 0
    total_selected_rel_tables = 0
    total_selected_baseline_tables = 0
    total_tokens = 0

    for file_name in file_names:
        if file_name in file_names_saved:
            continue

        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            table_content = f.read().strip()

        try:
            select_table_prompt = getattr(prompts, args.select_table_prompt).replace('<<tables>>', table_content)
            response = chat_structured(args.llm, select_table_prompt, TableSelectionForm, args.effort)
            selected_rel_ids = response['Relation_Tables']
            selected_baseline_ids = response['Baseline_Tables']
            tokens = encoder.encode(select_table_prompt)
            tokens_count = len(tokens)
            print(f'Prompt has {tokens_count} tokens')
        except Exception as e:
            print(f"❌ {e}")
            continue

        if selected_rel_ids is None or selected_baseline_ids is None:
            continue
        print(f"{file_name} → Rel Selected: {selected_rel_ids}; Baseline Selected: {selected_baseline_ids}")

        result = {file_name: {'Relation_Tables': [], 'Baseline_Tables': []}}
        if len(selected_rel_ids) > 0:
            result[file_name]['Relation_Tables'] = selected_rel_ids
            extracted_rel_tables = extract_tables_by_ids(table_content, selected_rel_ids)
            if not extracted_rel_tables:
                print(f"❌ No valid rel tables found for {file_name}.")
            else:
                output_rel_text = "\n\n".join(extracted_rel_tables)
                output_rel_path = os.path.join(rel_table_save_dir, file_name)
                with open(output_rel_path, "w", encoding="utf-8") as f:
                    f.write(output_rel_text)
        if len(selected_baseline_ids) > 0:
            result[file_name]['Baseline_Tables'] = selected_baseline_ids
            extracted_baseline_tables = extract_tables_by_ids(table_content, selected_baseline_ids)
            if not extracted_baseline_tables:
                print(f"❌ No valid baseline tables found for {file_name}.")
            else:
                output_baseline_text = "\n\n".join(extracted_baseline_tables)
                output_baseline_path = os.path.join(baseline_table_save_dir, file_name)
                with open(output_baseline_path, "w", encoding="utf-8") as f:
                    f.write(output_baseline_text)

        save_json_result(result, result_save_path)

        valid_files += 1
        total_selected_rel_tables += len(selected_rel_ids)
        total_selected_baseline_tables += len(selected_baseline_ids)
        total_tokens += tokens_count

        print(f"✅ Saved filtered tables for {file_name} ({len(selected_rel_ids)} rel tables, {len(selected_baseline_ids)} baseline tables)\n")

    print("\n========== SUMMARY ==========")
    print(f"Valid files: {valid_files}")
    print(f"Total selected rel tables across all files: {total_selected_rel_tables}")
    print(f"Total selected baseline tables across all files: {total_selected_baseline_tables}")
    print(f"Total tokens counted: {total_tokens}")
    if valid_files > 0:
        print(f"Average tokens per valid article: {total_tokens / valid_files:.2f}")
    print(f"Results saved to: {result_save_path}")
    print(f"Filtered rel tables saved to: {rel_table_save_dir}")
    print(f"Filtered baseline tables saved to: {baseline_table_save_dir}")
    print("=============================\n")

if __name__ == "__main__":
    args = args_argument()
    rel_table_save_dir = args.rel_table_save_dir
    baseline_table_save_dir = args.baseline_table_save_dir
    result_save_path = args.result_save_path
    folder_path = args.folder_dir
    all_file_names = os.listdir(folder_path)
    all_file_names.sort()

    chunk_size = args.chunk_size
    if args.chunk == 0:
        file_names = all_file_names
        result_save_path = result_save_path
        print("File: all (no chunking)")
    else:
        start_idx = (args.chunk - 1) * chunk_size
        end_idx = start_idx + chunk_size if args.chunk < args.number_of_chunk else None
        print(f"File: {start_idx}-{'' if end_idx is None else end_idx}")
        file_names = all_file_names[start_idx:end_idx]

        suffix = f"_{(end_idx or len(all_file_names))}"
        result_save_path = result_save_path.replace('.json', f'{suffix}.json')

    print(f"Relation Table save path:{rel_table_save_dir}\nBaseline Table save path:{baseline_table_save_dir}\nResult save path:{result_save_path}")

    select_tables(file_names, folder_path, rel_table_save_dir, baseline_table_save_dir, result_save_path)