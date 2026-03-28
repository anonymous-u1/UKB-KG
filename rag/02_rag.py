from llm.openai_chat import *
import warnings
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
from neo4j import GraphDatabase
from typing import List
from transformers import AutoTokenizer, AutoModel
import prompts
import torch
import tiktoken
import argparse
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')


def args_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, required=True)
    parser.add_argument('--effort', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--params', type=int, required=True)
    parser.add_argument('--score', type=int, required=True)
    parser.add_argument('--chunk', type=int, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--response_path', type=str, required=True)
    parser.add_argument('--triple_csv_path', type=str, required=True)
    parser.add_argument('--ukb_node_emb', type=str, required=True)
    parser.add_argument('--data_entity_emb', type=str, required=True)
    args = parser.parse_args()
    return args


args = args_argument()
method = args.method


# ========================
# Load
# ========================
encoder = tiktoken.encoding_for_model('gpt-4o')

data = json.load(open(args.data_path))
items = list(data.items())

if args.chunk == 0:
    data = dict(items)
    print("Using all data")
elif args.chunk == 1:
    data = dict(items[:250])
    print("Using chunk 1")
else:
    data = dict(items[250:])
    print("Using chunk 2")

if method == "rag":
    triple_csv_df = pd.read_csv(args.triple_csv_path)

    # Check confidence / frequency score
    for col in ["Confidence_Score", "Frequency_Score"]:
        none_cnt = (triple_csv_df[col] == "none").sum()
        nan_cnt = (triple_csv_df[col].isna()).sum()
        if none_cnt > 0 or nan_cnt > 0:
            raise Exception(f"{col}: {none_cnt} 'none' or nan values")

    # Load BioBERT
    model_name = "model/biobert_v1.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    bert_model.eval()

    # Load embeddings
    ukb_node_emb = np.load(args.ukb_node_emb, allow_pickle=True)
    graph_entities = ukb_node_emb["entities"]
    graph_embeddings = ukb_node_emb["embeddings"]

    entity_emb_npz = np.load(args.data_entity_emb, allow_pickle=True)

    # Neo4j Handler
    class Neo4jHandler:
        def __init__(self, uri, user, password):
            self.driver = GraphDatabase.driver(uri, auth=(user, password))

        def close(self):
            self.driver.close()

        def find_shortest_path(self, start_node, end_node, max_hops=4):
            with self.driver.session() as session:
                result = session.read_transaction(self._find_shortest_path, start_node, end_node, max_hops)
                return result

        @staticmethod
        def _find_shortest_path(tx, start_node, end_node, max_hops):
            query = (
                f"MATCH (start {{name: \"{start_node}\"}}), (end {{name: \"{end_node}\"}}), "
                f"p = shortestPath((start)-[*..{max_hops}]-(end)) "
                f"RETURN p"
            )
            result = tx.run(query, start_node=start_node, end_node=end_node, max_hops=max_hops)
            paths = []
            for record in result:
                path = record["p"]
                path_list = []
                for rel in path.relationships:
                    path_list.append({
                        "start": rel.start_node["name"],
                        "relation": rel.type,
                        "end": rel.end_node["name"]
                    })
                paths.append(path_list)
            return paths


    neo4j_handler = Neo4jHandler("neo4j_url", "user", "password")


# ========================
# UTILS
# ========================
def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding


def normalize(x):
    return (x + 1) / 2


# ========================
# FUNCTIONS
# ========================
def get_nodes(question, qid: str, k1: int) -> List[str]:
    if qid not in entity_emb_npz:
        print(f"{qid} not in entity_emb_npz")
        q_emb = get_embedding(question)
        sims = cosine_similarity([q_emb], graph_embeddings)[0]
        top_idx = sims.argsort()[-k1:][::-1]
        return list(graph_entities[top_idx])

    entity_embeddings = entity_emb_npz[qid]  # (Ne, 768)
    nodes = set()

    for emb in entity_embeddings:
        sims = cosine_similarity([emb], graph_embeddings)[0]
        top_idx = sims.argsort()[-k1:][::-1]
        for idx in top_idx:
            nodes.add(graph_entities[idx])

    return list(nodes)


def get_candidate_triples(entities: list) -> list:
    triples = []
    for entity in entities:
        head_df = triple_csv_df[triple_csv_df['HeadName'] == entity]
        tail_df = triple_csv_df[triple_csv_df['TailName'] == entity]

        for _, r in head_df.iterrows():
            triples.append([
                r['HeadName'], r['RelName'], r['TailName'],
                r['PMCID'], r['Confidence_Score'], r['Frequency_Score']
            ])

        for _, r in tail_df.iterrows():
            triples.append([
                r['HeadName'], r['RelName'], r['TailName'],
                r['PMCID'], r['Confidence_Score'], r['Frequency_Score']
            ])
    return triples


def get_top_triples(question, triples, k2):
    def score(triple, q_emb):
        triple_str = f"[{triple[0]},{triple[1]},{triple[2]}]"
        t_emb = get_embedding(triple_str)
        sim = cosine_similarity([q_emb], [t_emb])[0][0]
        sim = normalize(sim)

        if args.score == 1:
            w1, w2, w3 = 0.1, 0.05, 0.85
        elif args.score == 2:
            w1, w2, w3 = 0.05, 0.05, 0.9
        elif args.score == 3:
            w1, w2, w3 = 0.15, 0.05, 0.8

        confidence_score = triple[4]
        frequency_score = triple[5]
        combined_score = w1 * confidence_score + w2 * frequency_score + w3 * sim

        return combined_score

    q_emb = get_embedding(question)
    scored = [(t, score(t, q_emb)) for t in triples]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_triples = [t for t, _ in scored[:k2]]
    return top_triples


def get_top_paths(question, paths, k3):
    q_emb = get_embedding(question)
    scored = []
    for p in paths:
        p_emb = get_embedding(p)
        sim = cosine_similarity([q_emb], [p_emb])[0][0]
        sim = normalize(sim)
        scored.append((p, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_paths = [p for p, _ in scored[:k3]]
    return top_paths


# ========================
# RETRIEVERS
# ========================
def neighbor_based_retriever(triples, pmcids):
    output_text = "\n\n## Neighbor-based Context:\n\nTriples:\n"
    for idx, (triple, pmcid) in enumerate(zip(triples, pmcids)):
        output_text += f"{idx + 1}. {triple}\n"
    return output_text.strip()


def path_based_retriever(question, entities: list, k3) -> str:
    entities = list(set(entities))

    paths_dict = {}
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            start_node = entities[i]
            end_node = entities[j]
            paths = neo4j_handler.find_shortest_path(start_node, end_node)
            if paths:
                paths_dict[f"Path linking '{start_node}' and '{end_node}'"] = paths

    path_text_list = []
    for idx, (key, paths) in enumerate(paths_dict.items()):
        for path in paths:
            path_representation = ""
            for step in path:
                if not path_representation:
                    path_representation += f"{step['start']} -(" + step['relation'] + ")-> " + step['end']
                else:
                    if path_representation.endswith(step['start']):
                        path_representation += f" -(" + step['relation'] + ")-> " + step['end']
                    elif path_representation.endswith(step['end']):
                        path_representation += f" <-(" + step['relation'] + ")- " + step['start']
                    elif path_representation.startswith(step['end']):
                        path_representation = step['start'] + f" -(" + step['relation'] + ")-> " + path_representation
                    elif path_representation.startswith(step['start']):
                        path_representation = step['end'] + f" <-(" + step['relation'] + ")- " + path_representation

            path_representation = "[" + path_representation + "]"
            path_text_list.append(path_representation)

    top_paths = get_top_paths(question, path_text_list, k3)
    output_text = "\n\n## Path-based Context:\n\nPaths:\n"
    for idx, path in enumerate(top_paths):
        output_text += f"{idx + 1}. {path}\n"

    return output_text


def retriever(qid, question: str):
    if args.params == 1:
        k1, k2, k3 = 4, 50, 30
    elif args.params == 2:
        k1, k2, k3 = 7, 70, 45

    nodes = get_nodes(question, qid, k1)
    candidate_triples = get_candidate_triples(nodes)
    top_triples_raw = get_top_triples(question, candidate_triples, k2)

    top_triples = [f'[{triple[0]} -({triple[1]})-> {triple[2]}]' for triple in top_triples_raw]
    top_pmcids = [triple[3] for triple in top_triples_raw]

    context = neighbor_based_retriever(top_triples, top_pmcids)
    try:
        context += path_based_retriever(question, nodes, k3)
    except Exception as e:
        print(f"Path retriever error: {e}")

    return context


# ========================
# RUN RAG
# ========================
if method != "none":
    template_0shot = args.template.replace(f"{args.data}_{method}", f"{args.data}_{method}_0shot")
    template_1shot = args.template.replace(f"{args.data}_{method}", f"{args.data}_{method}_1shot")
    save_0shot_path = args.save_path.replace(f"{args.data}_{method}", f"{args.data}_{method}_0shot")
    save_1shot_path = args.save_path.replace(f"{args.data}_{method}", f"{args.data}_{method}_1shot")
    prompt_0shot_path = args.prompt_path.replace(f"{args.data}_{method}", f"{args.data}_{method}_0shot")
    prompt_1shot_path = args.prompt_path.replace(f"{args.data}_{method}", f"{args.data}_{method}_1shot")
    results_0shot = json.load(open(save_0shot_path)) if os.path.exists(save_0shot_path) else {}
    results_1shot = json.load(open(save_1shot_path)) if os.path.exists(save_1shot_path) else {}
else:
    template = args.template
    save_path = args.save_path
    prompt_path = args.prompt_path
    results = json.load(open(save_path)) if os.path.exists(save_path) else {}

for i, (idx, details) in enumerate(tqdm(data.items())):
    if method != "none":
        if idx in results_0shot and idx in results_1shot:
            continue
    else:
        if idx in results:
            continue

    if args.data == "pubmedqa":
        question = details.get('QUESTION', '')
    elif args.data == "bioasq":
        question = details.get('QUESTION', '')
    if not question:
        continue

    try:
        if method == "rag":
            rag_context = retriever(idx, question)
            tokens = encoder.encode(rag_context)
            print(f'RAG context has {len(tokens)} tokens')
            max_context_length = 10000
            rag_context = encoder.decode(tokens[:max_context_length])
            rag_0shot_prompt = (getattr(prompts, template_0shot).replace('<<question>>', question).replace('<<context>>', rag_context))
            rag_1shot_prompt = (getattr(prompts, template_1shot).replace('<<question>>', question).replace('<<context>>', rag_context))
            prompt_0shot = re.sub(r'\n{3,}', '\n\n', rag_0shot_prompt)
            prompt_1shot = re.sub(r'\n{3,}', '\n\n', rag_1shot_prompt)
        elif method == "cot":
            prompt_0shot = getattr(prompts, template_0shot).replace('<<question>>', question)
            prompt_1shot = getattr(prompts, template_1shot).replace('<<question>>', question)
        elif method == "none":
            prompt = getattr(prompts, template).replace('<<question>>', question)

        if method != "none":
            if idx not in results_0shot:
                response = chat_wo_structure(args.llm, prompt_0shot, args.effort)

                if i < 20:
                    with open(prompt_0shot_path, 'a', encoding='utf-8') as file:
                        file.write(prompt_0shot)

                results_0shot[idx] = {}
                answer = re.findall(r'<ans>(.*?)</ans>', response)
                results_0shot[idx] = answer[0] if answer else ""

                with open(save_0shot_path, 'w') as file:
                    json.dump(results_0shot, file, indent=4)

            if idx not in results_1shot:
                response = chat_wo_structure(args.llm, prompt_1shot, args.effort)

                if i < 20:
                    with open(prompt_1shot_path, 'a', encoding='utf-8') as file:
                        file.write(prompt_1shot)

                results_1shot[idx] = {}
                answer = re.findall(r'<ans>(.*?)</ans>', response)
                results_1shot[idx] = answer[0] if answer else ""

                with open(save_1shot_path, 'w') as file:
                    json.dump(results_1shot, file, indent=4)
        else:
            if idx not in results:
                response = chat_wo_structure(args.llm, prompt, args.effort)

                if i < 20:
                    with open(prompt_path, 'a', encoding='utf-8') as file:
                        file.write(prompt)

                results[idx] = {}
                answer = re.findall(r'<ans>(.*?)</ans>', response)
                results[idx] = answer[0] if answer else ""

                with open(save_path, 'w') as file:
                    json.dump(results, file, indent=4)

    except Exception as e:
        print(f"Index {idx} error: {e}")