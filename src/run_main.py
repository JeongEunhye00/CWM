# coding: utf-8
import argparse
import json
import logging
import os
import random
from datetime import datetime

import torch
from tqdm import tqdm

from models import ModelWrapper
from self_discover import SelfDiscover
from solve_problem import solve_problem


def setup_logging(log_file, log_level):
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))
    handler = logging.FileHandler(log_file)
    handler.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


model_version = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def main():
    parser = argparse.ArgumentParser(description="Run Step 1, 2, and 3 with a single model load")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="gpt-oss-20b")
    parser.add_argument("--dataset_name", type=str, default="HotpotQA", help="BBH, T4D, MATH500, HotpotQA, StrategyQA, Musique")
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--data_dir", type=str, default="dataset")

    # PPL→alpha gating
    parser.add_argument("--alpha_cap", type=float, default=5.0)
    parser.add_argument("--alpha_pos_cap", type=float, default=5.0)
    parser.add_argument("--alpha_neg_cap", type=float, default=2.0)
    parser.add_argument("--ppl_off", type=float, default=20.0)
    parser.add_argument("--ppl_on", type=float, default=150.0)
    # Starting point for negative alpha in the lower range
    parser.add_argument("--ppl_neg_on", type=float, default=5.0)

    # === Scalar Scaling Strength ===
    parser.add_argument("--marker_gain_coef", type=float, default=1.00)   # alpha * coef
    parser.add_argument("--marker_gain_cap", type=float, default=1.00)    # Maximum amplification (e.g., 0.5 => 1.5x)
    parser.add_argument("--marker_dropout", type=float, default=0.00)     # Random dropout ratio within the span
    parser.add_argument("--marker_taper", action="store_true",  default=False)  # Stronger at the center, weaker at the edges

    # === Non-Collinear Delta Strength ===
    parser.add_argument("--delta_gain_coef", type=float, default=0.07)    # alpha * coef (additional vector magnitude)
    parser.add_argument("--delta_gain_cap", type=float, default=1.0)      # Maximum delta magnitude (norm)

    # === Safeguards for the GPT-OSS Path ===
    parser.add_argument("--force_eager_attn", action="store_true")
    parser.add_argument("--disable_sdp_flash", action="store_true")


    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    log_dir = os.path.join(args.log_dir, args.dataset_name, 'Step1-2')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{args.model_name}_{timestamp}.log")

    logger = setup_logging(log_file, args.log_level)
    logger.info(f"Starting self-discover using {args.model_name} for {args.dataset_name}")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")


    output_dir = os.path.join(args.output_dir, args.dataset_name, 'Step1-2', args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{args.model_name}_Stepe1-2_delta_{args.delta_gain_coef}_negcap_{args.alpha_neg_cap}.json")
    
    args.model_id = model_version[args.model_name]

    model = ModelWrapper(args)

    actions = ["SELECT", "ADAPT", "IMPLEMENT"]

    data_dir = os.path.join(args.data_dir, args.dataset_name)
    data_path = os.path.join(data_dir, f"{args.dataset_name}.json")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    res = []

    for data in tqdm(dataset, desc="Step 1 & 2"):
        runner = SelfDiscover(model, actions, data['question'], logger, args)
        runner()

        tmp = {
            "id": data['id'],
            "question": data['question'],
            "answer": data.get('answer'),
            "selected_modules": runner.selected_modules,
            "using_rag": runner.using_rag,
            "adapted_modules": runner.adapted_modules,
            "reasoning_structure": runner.reasoning_structure,
            "spp_nll": runner.spp_nll,
            "spp_ppl": runner.spp_ppl,
            "alpha": runner.alpha,
            "used_scaled_embed": runner.used_scaled_embed,
            "m1_found": runner.m1_found,
            "m1_span_len": runner.m1_span_len,
            "scaled_tokens": runner.scaled_tokens,
            "marker_eff_gain": runner.marker_eff_gain,
            "delta_eff_gain": runner.delta_eff_gain,
        }
       
        if args.dataset_name == "BBH":
            tmp['dataset_name'] = data['dataset_name']
        res.append(tmp)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=4)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)

    logger.info("Step 1 & 2 complete; starting Step 3 with the loaded model")
    output_dir = os.path.join(args.output_dir, args.dataset_name, 'Step3', args.model_name)
    os.makedirs(output_dir, exist_ok=True)


    output_path = os.path.join(output_dir, f"{args.model_name}_Step3_delta_{args.delta_gain_coef}_negcap_{args.alpha_neg_cap}.json")
    
    results = res
    id_data = {data["id"]: data for data in dataset}

    tmp = []
    
    for step2_record in tqdm(results, desc="Step 3"):
        res = dict(step2_record)
        if not res['reasoning_structure']:
            res['response'] = None
            tmp.append(res)
            continue

        gold_context = None
        if res['using_rag'] and args.dataset_name in ["HotpotQA", "StrategyQA", "Musique"]:
            gold_context = ""
            gold_context = "\n".join(id_data[res['id']]['supporting']).strip()

        dataset_name = args.dataset_name
        if dataset_name == "BBH":
            if res['dataset_name'] == "web_of_lies":
                dataset_name = "BBH_web_of_lies"
            elif res['dataset_name'] == "formal_fallacies":
                dataset_name = "BBH_formal_fallacies"
        response = solve_problem(model, res['question'], res['reasoning_structure'], gold_context, dataset_name)
        res['response'] = response

        tmp.append(res)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tmp, f, indent=4)
      
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tmp, f, indent=4)

    logger.info("Step 3 results saved to %s", os.path.abspath(output_path))


if __name__ == "__main__":
    main()
