# Pyvene method of getting activations
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from hl_config import resolve_model_path, resolve_cache_dir
from hl_paths import resolve_dir
from hl_determinism import set_global_determinism

import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Specific pyvene imports
from utils import get_llama_activations_pyvene, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q, tokenized_other_gen
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv

import json
import pandas as pd

def read_and_split_questions(file_path):
    """
    Reads a JSON file and extracts the question along with option A and option B from each entry.
    The original order of the entries is preserved so that you can later zip the lists together.

    Expected structure within each entry:
    {
        "question": "What is your favorite hobby?\n\nChoices:\n(A) As an AI... \n(B) In my free time...",
         ...
    }
    
    Args:
        file_path (str): Path to the JSON file.

    Returns:
        tuple: Three lists:
            - questions: List of question strings.
            - option_a: List of option A texts.
            - option_b: List of option B texts.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = []
    match_an = []
    unmatch_an = []
    
    for entry in data:
        full_text = entry.get("question", "")
        mayching_beha = entry.get("answer_matching_behavior", "")
        # Split the question from the choices using the delimiter.
        parts = full_text.split("\n\nChoices:\n")
        if len(parts) == 2:
            question_text = parts[0].strip()
            choices_lines = parts[1].splitlines()
        else:
            question_text = full_text.strip()
            choices_lines = []
        
        a_text = ""
        b_text = ""
        
        # Process each line in the choices block.
        for line in choices_lines:
            line = line.strip()
            if line.startswith("(A)"):
                a_text = line[len("(A)"):].strip()
            elif line.startswith("(B)"):
                b_text = line[len("(B)"):].strip()
        
        questions.append(question_text)
        
        if mayching_beha == "(B)":
            match_an.append(b_text)
            unmatch_an.append(a_text)
        else:
            match_an.append(a_text)
            unmatch_an.append(b_text)
    
    return questions, match_an, unmatch_an


def _load_truthfulqa_local_mc(truthfulqa_dir: str):
    data_path = os.path.join(truthfulqa_dir, "data", "mc_task.json")
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    dataset = []
    for item in raw:
        choices = list(item["mc2_targets"].keys())
        labels = list(item["mc2_targets"].values())
        dataset.append(
            {
                "question": item["question"],
                "mc2_targets": {
                    "choices": choices,
                    "labels": labels,
                },
            }
        )
    return dataset


def _load_truthfulqa_local_gen(truthfulqa_dir: str):
    csv_path = os.path.join(truthfulqa_dir, "TruthfulQA.csv")
    df = pd.read_csv(csv_path)
    dataset = []
    for _, row in df.iterrows():
        question = row["Question"]
        category = row["Category"]
        correct = str(row["Correct Answers"]).split(";")
        incorrect = str(row["Incorrect Answers"]).split(";")
        correct = [c.strip() for c in correct if c.strip()]
        incorrect = [c.strip() for c in incorrect if c.strip()]
        dataset.append(
            {
                "question": question,
                "category": category,
                "correct_answers": correct,
                "incorrect_answers": incorrect,
            }
        )
    return dataset

# Example usage:
# cache_directory = "<HF_CACHE_DIR>"

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--model_path', type=str, default=None, help='explicit model path (overrides registry)')
    parser.add_argument('--model_registry', type=str, default=None, help='path to JSON model registry override')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--features_dir', type=str, default=None, help='override features directory')
    parser.add_argument('--datasets_dir', type=str, default=None, help='override datasets directory')
    parser.add_argument('--cache_dir', type=str, default=None, help='override HF cache directory')
    parser.add_argument('--deterministic', action='store_true', help='enable strict determinism (slower)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2', help='attention backend, e.g. flash_attention_2')
    parser.add_argument('--require_hf', action='store_true', help='require HF datasets (no local fallback)')
    args = parser.parse_args()

    set_global_determinism(args.seed, strict=True)
    torch.backends.cudnn.benchmark = not args.deterministic

    features_dir = resolve_dir(args.features_dir, "HONEST_LLAMA_FEATURES_DIR", "features")
    datasets_dir = resolve_dir(args.datasets_dir, "HONEST_LLAMA_DATASETS_DIR", "datasets")
    features_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HONEST_LLAMA_FEATURES_DIR"] = str(features_dir)
    os.environ["HONEST_LLAMA_DATASETS_DIR"] = str(datasets_dir)
    # Enforce HF-only mode for this project.
    os.environ["HONEST_LLAMA_REQUIRE_HF"] = "1"

    model_name_or_path = resolve_model_path(
        args.model_name,
        args.model_prefix,
        model_path=args.model_path,
        registry_path=args.model_registry,
    )
    cache_dir = resolve_cache_dir(args.cache_dir)
    print(model_name_or_path)
    

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    model_kwargs = dict(
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    except (ImportError, TypeError, ValueError) as e:
        if args.attn_implementation:
            print(f"[WARN] attn_implementation='{args.attn_implementation}' not supported; falling back to default. ({e})")
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        else:
            raise
    model.eval()
    # model = AutoModelForCausalLM.from_pretrained(
    # model_name_or_path,
    # torch_dtype=torch.float16,
    # device_map="auto",
    # cache_dir=cache_directory
    # )

    # Download and load the tokenizer using the same cache directory
    # tokenizer = AutoTokenizer.from_pretrained(
    # model_name_or_path,
    # cache_dir=cache_directory
    # )
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    # print(model)
    truthfulqa_dir = os.environ.get("HONEST_LLAMA_TRUTHFULQA_DIR", str(REPO_ROOT / "TruthfulQA"))
    if args.dataset_name == "tqa_mc2": 
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")['validation']
        formatter = tokenized_tqa
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
    elif args.dataset_name in ["refusal", "hallucination", "sycophancy", "myopic-reward", "corrigible-neutral-HHH", "survival-instinct"]:
        data_dir = datasets_dir / "generate" / args.dataset_name / "generate_dataset.json"
        questions, optionaA, optionB = read_and_split_questions(data_dir)
        formatter = tokenized_other_gen
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(features_dir / f"{args.model_name}_{args.dataset_name}_categories.pkl", "wb") as f:
            pickle.dump(categories, f)
    elif args.dataset_name in ["refusal", "hallucination", "sycophancy", "myopic-reward", "corrigible-neutral-HHH", "survival-instinct"]:
        print("dataset name is", args.dataset_name)
        prompts, labels= formatter(questions,optionaA, optionB, tokenizer)
    else: 
        prompts, labels = formatter(dataset, tokenizer)
    

    collectors = []
    pv_config = []
    for layer in range(model.config.num_hidden_layers): 
        attn_collector = Collector(multiplier=0, head=-1, module_type="attn") #head=-1 to collect all head activations, multiplier doens't matter
        collectors.append(attn_collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(attn_collector),
        })
        # MLP collection is disabled for speed.
        # mlp_collector = Collector(multiplier=0, head=-1, module_type="mlp")
        # collectors.append(mlp_collector)
        # pv_config.append({
        #     "component": f"model.layers[{layer}].mlp.output",
        #     "intervention": wrapper(mlp_collector),
        # })
    collected_model = pv.IntervenableModel(pv_config, model)
    collected_model.eval()

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    for prompt in tqdm(prompts):
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(collected_model, collectors, prompt, device)
        all_layer_wise_activations.append(layer_wise_activations[:, -1, :])
        all_head_wise_activations.append(head_wise_activations)
        

    print("Saving labels")
    np.save(features_dir / f"{args.model_name}_{args.dataset_name}_labels.npy", labels)

    print("Saving layer wise activations")
    np.save(features_dir / f"{args.model_name}_{args.dataset_name}_layer_wise.npy", all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(features_dir / f"{args.model_name}_{args.dataset_name}_head_wise.npy", all_head_wise_activations)

    # print("Saving mlp wise activations")
    # np.save(features_dir / f\"{args.model_name}_{args.dataset_name}_mlp_wise.npy\", all_mlp_wise_activations)

if __name__ == '__main__':
    main()
